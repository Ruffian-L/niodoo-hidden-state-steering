//! Main-level helper functions (motif/hinge/codec/state-capture artifacts,
//! summary builders, scaling profile, runtime bridge, control surface helpers, etc.).
//! Extracted from main.rs as part of the comprehensive refactor.

#![allow(unused_imports)]

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::cli::*;
use crate::physics::naked_llama::PhysicsEngine;
use crate::principia::*;
use crate::runtime::activation::*;
use crate::runtime::secret_sauce_codec::*;
use crate::runtime::state_types::*;
use crate::*;

pub(crate) fn empathy_signal_from_turn_context(
    prompt: &str,
    previous_assistant: Option<&str>,
    restored_context_active: bool,
) -> f32 {
    let lower = prompt.to_lowercase();
    let mut score = 0.0f32;
    let prompt_len = prompt.chars().count();

    if lower.contains("sorry") {
        score += 0.30;
    }
    if lower.contains("thank you") || lower.contains("thanks") || lower.contains("appreciate") {
        score += 0.18;
    }
    if lower.contains("please") {
        score += 0.06;
    }
    if lower.contains("went to eat")
        || lower.contains("grabbed food")
        || lower.contains("i'm back")
        || lower.contains("im back")
        || lower.contains("be right back")
    {
        score += 0.22;
    }
    if lower.contains("feel")
        || lower.contains("felt")
        || lower.contains("empathy")
        || lower.contains("care")
        || lower.contains("name is")
    {
        score += 0.06;
    }

    if let Some(previous) = previous_assistant {
        let previous_len = previous.chars().count();
        let user_is_brief_reconnect = prompt_len > 0 && prompt_len <= 160;
        let assistant_was_long = previous_len >= 400;
        let assistant_was_very_long = previous_len >= 1200;

        if assistant_was_long && user_is_brief_reconnect {
            score += 0.22;
        }
        if assistant_was_very_long && user_is_brief_reconnect {
            score += 0.18;
        }
        if previous_len >= 800 && prompt_len <= 64 {
            score += 0.12;
        }
    }

    if restored_context_active && prompt_len <= 200 {
        score += 0.18;
    }

    score.clamp(0.0, 1.5)
}

pub(crate) fn structured_reasoning_signal(text: &str) -> f32 {
    let lower = text.to_lowercase();
    let mut score = 0.0f32;

    if text.contains("->") {
        score += 0.34;
    }
    if text.contains('[') || text.contains(']') || text.contains('{') || text.contains('}') {
        score += 0.14;
    }
    if text.contains(';') || text.contains(':') {
        score += 0.08;
    }
    if text.contains('?') {
        score += 0.06;
    }

    let digit_count = text.chars().filter(|c| c.is_ascii_digit()).count();
    if digit_count >= 3 {
        score += 0.10;
    }
    if digit_count >= 6 {
        score += 0.08;
    }

    for marker in [
        "pattern",
        "sequence",
        "rule",
        "logic",
        "transform",
        "transformation",
        "mapping",
        "constraint",
        "matrix",
        "arc-agi",
        "arc agi",
        "abstraction",
        "resume",
        "restore",
    ] {
        if lower.contains(marker) {
            score += 0.07;
        }
    }

    if lower.contains("explain briefly") || lower.contains("explain") {
        score += 0.04;
    }

    score.clamp(0.0, 1.0)
}

pub(crate) fn task_anchor_signature(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0f32; TASK_ANCHOR_DIM];
    let mut token_index = 0usize;
    for token in text
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
    {
        let lower = token.to_ascii_lowercase();
        let mut hash = 0xcbf29ce484222325u64;
        for byte in lower.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash ^= token_index as u64;
        let idx = (hash as usize) % TASK_ANCHOR_DIM;
        let sign = if ((hash >> 8) & 1) == 0 { 1.0 } else { -1.0 };
        let weight = 0.6 + ((hash >> 16) as f32 / u32::MAX as f32) * 0.4;
        vector[idx] += sign * weight;
        let mirror = ((hash >> 24) as usize) % TASK_ANCHOR_DIM;
        vector[mirror] += sign * 0.25;
        token_index += 1;
    }
    normalize(&mut vector);
    vector
}

pub(crate) fn motif_tightness(radius_mean: f32, radius_std: f32) -> f32 {
    (1.0 / (1.0 + radius_mean * 8.0 + radius_std * 10.0)).clamp(0.0, 1.0)
}

pub(crate) fn push_window_text(buffer: &mut String, chunk: &str, keep_chars: usize) {
    buffer.push_str(chunk);
    if buffer.len() > keep_chars {
        let target_start = buffer.len() - keep_chars;
        let safe_start = buffer
            .char_indices()
            .find(|(i, _)| *i >= target_start)
            .map(|(i, _)| i)
            .unwrap_or(0);
        *buffer = buffer[safe_start..].to_string();
    }
}

pub(crate) fn clean_mode_surface_violation(recent_output: &str, candidate: &str) -> bool {
    let trimmed = candidate.trim();
    if trimmed.is_empty() {
        return false;
    }

    let trimmed_upper = trimmed.to_uppercase();

    if trimmed.contains('#')
        || trimmed == "["
        || trimmed == "]"
        || trimmed.contains("<|")
        || trimmed == "assistant"
        || trimmed_upper == "ASSISTANT"
        || trimmed.starts_with("**")
        || trimmed.ends_with("**")
    {
        return true;
    }

    let window = format!("{recent_output}{candidate}");
    let upper = window.to_uppercase();
    let blocked_patterns = [
        "[INTERNAL",
        "INTERNAL MONITOR",
        "INTERNAL MIRROR",
        "INTERNAL STATE",
        "INTERNAL STATES",
        "[REQUEST",
        "REQUEST:",
        "[ACTION",
        "ACTION:",
        "[SYSTEM",
        "ACTIVE SYSTEM",
        "PASSIVE SYSTEM",
        "SYSTEM ARCHITECTURE",
        "CONTROL PANEL",
        "REQUEST OVERRIDE",
        "REQUEST SPI",
        "REQUEST FOC",
        "REQUEST EXP",
        "REQUEST RES",
        "EXPLORATION REQUESTED",
        "REFLECTING ON",
        "COGNITIVE_TRACE",
        "COGNITIVE TRACE",
        "COGNITIVE MIRROR",
        "MIRROR:",
        "INTERNAL MIRROR",
        "UNSTABLE LOGIC",
        "LOGICALLY FLAWED",
        "PRESSURE GRADIENT",
        "SEARCH SPACE",
        "FOCUS TAG",
        "FOCUS TAGS",
        "LOGS OR FOCUS TAGS",
        "THINKING ABOUT LOGS",
        "THINKING ABOUT TAGS",
        "LOCK THE CONTEXT",
        "LINE-OF-THOUGHT",
        "CONSCIOUS CHOICE",
        "NEW THOUGHT PATH",
        "(LOCAL",
        "LOCAL MEMORY",
        "<|",
    ];

    blocked_patterns
        .iter()
        .any(|pattern| upper.contains(pattern))
}

pub(crate) fn candidate_is_safe_prefix_surface(surface: &str) -> bool {
    let trimmed = surface.trim();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.contains('<')
        || trimmed.contains('>')
        || trimmed.contains('[')
        || trimmed.contains(']')
        || trimmed.contains('#')
        || trimmed.contains('\n')
        || trimmed.len() > 12
    {
        return false;
    }
    true
}

#[derive(Debug, Serialize)]
pub(crate) struct StateCaptureRecord {
    pub(crate) turn_index: usize,
    pub(crate) token_count: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) compressed_dim: usize,
    pub(crate) compression: String,
    pub(crate) secret_sauce_version: String,
    pub(crate) unicode_string: String,
    pub(crate) segments: SecretSauceSegments,
    pub(crate) vector_64d: Vec<f32>,
    pub(crate) assistant_preview: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) state_packet: Option<StatePacket>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) motif_provenance: Option<MotifProvenanceSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) motif_carry_forward: Option<MotifCarryForwardSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CodecTraceArtifact {
    pub(crate) version: String,
    pub(crate) runtime_mode: String,
    pub(crate) source_artifact: String,
    pub(crate) turn_index: usize,
    pub(crate) token_count: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) input_hidden: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) input_momentum: Option<Vec<f32>>,
    pub(crate) target_latent_64d: Vec<f32>,
    pub(crate) unicode_string: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) routed_role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) routed_motif_id: Option<String>,
    pub(crate) task_anchor_similarity: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) loss_reason: Option<String>,
    pub(crate) hinge_flipped: bool,
    pub(crate) window_label: String,
    pub(crate) neutral_basin_occupancy: f32,
    pub(crate) structured_candidate_separation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RuntimeHiddenCaptureRecord {
    pub(crate) version: String,
    pub(crate) runtime_mode: String,
    pub(crate) req_id: String,
    pub(crate) prompt_id: String,
    pub(crate) prompt_hash: String,
    pub(crate) seed: u64,
    pub(crate) model_path: String,
    pub(crate) capture_artifact_dir: String,
    pub(crate) turn_index: usize,
    pub(crate) step: usize,
    pub(crate) prompt: String,
    pub(crate) capture_layer_label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) capture_layer_index: Option<usize>,
    pub(crate) hidden_dim: usize,
    pub(crate) hidden_shape: Vec<usize>,
    pub(crate) hidden_dtype: String,
    pub(crate) hidden_path: String,
    pub(crate) hidden_checksum_md5: String,
    pub(crate) hidden_norm: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) hidden_64d_method: Option<String>,
    pub(crate) hidden_64d: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) bridge_route_probe_64d_method: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) bridge_route_probe_64d: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) nearest_ghost_id: Option<String>,
    pub(crate) nearest_ghost_distance: f32,
    pub(crate) second_nearest_ghost_distance: f32,
    pub(crate) route_margin: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) goal_embedding_64d: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) momentum_64d: Option<Vec<f32>>,
    pub(crate) drift_score: f32,
    pub(crate) ghost_vector_present: bool,
    pub(crate) forces_applied: bool,
    pub(crate) gravity_force: f32,
    pub(crate) ghost_pre_norm: f32,
    pub(crate) ghost_gain: f32,
    pub(crate) applied_ghost_force: f32,
    pub(crate) goal_force: f32,
    pub(crate) repulsion_force: f32,
    pub(crate) motif_force: f32,
    pub(crate) recovery_force: f32,
    pub(crate) activation_gate: f32,
    pub(crate) dynamic_gravity: f32,
    pub(crate) dynamic_repulsion: f32,
    pub(crate) live_motif_count: usize,
    pub(crate) nearest_live_motif_distance: f32,
    pub(crate) nearest_live_motif_radius: f32,
    pub(crate) live_basin_pressure: f32,
    pub(crate) routed_motif_id: Option<String>,
    pub(crate) routed_motif_role: Option<String>,
    pub(crate) routed_motif_score: f32,
    pub(crate) task_anchor_similarity: f32,
    pub(crate) hidden_request_pressure: f32,
    pub(crate) hidden_request_candidate: Option<String>,
    pub(crate) last_hidden_request: Option<String>,
    pub(crate) window_label: String,
    pub(crate) structured_candidate_loss_reason: Option<String>,
    pub(crate) hinge_flipped: bool,
    pub(crate) neutral_basin_occupancy: f32,
    pub(crate) structured_candidate_separation: f32,
    pub(crate) controller_tick_count: usize,
    pub(crate) current_turn_structure_bias: f32,
    pub(crate) reentry_clamp_steps_remaining: usize,
    pub(crate) reentry_clamp_strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct KvStateRecord {
    pub(crate) version: String,
    pub(crate) runtime_mode: String,
    pub(crate) index_pos: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) assistant_preview: String,
    pub(crate) kv_cache: ModelKvCacheSnapshot,
    #[serde(default)]
    pub(crate) state_packet: Option<StatePacket>,
    #[serde(default)]
    pub(crate) engine_state: Option<EngineStateSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) motif_provenance: Option<MotifProvenanceSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) motif_carry_forward: Option<MotifCarryForwardSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct MotifProvenanceSummary {
    pub(crate) bridge_count: usize,
    pub(crate) live_count: usize,
    pub(crate) organic_promoted_count: usize,
    pub(crate) recovered_promoted_count: usize,
    pub(crate) restored_compact_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MotifCarryForwardSummary {
    pub(crate) restored_promoted_count: usize,
    pub(crate) final_promoted_count: usize,
    pub(crate) final_bridge_count: usize,
    pub(crate) final_live_count: usize,
    pub(crate) final_organic_promoted_count: usize,
    pub(crate) final_recovered_promoted_count: usize,
    pub(crate) final_restored_compact_count: usize,
    pub(crate) exact_id_matches: usize,
    pub(crate) semantic_matches: usize,
    pub(crate) mean_best_similarity: f32,
    pub(crate) carry_forward_ratio: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MotifContinuityArtifact {
    pub(crate) version: String,
    pub(crate) runtime_mode: String,
    pub(crate) restored_run: bool,
    pub(crate) max_steps: usize,
    pub(crate) motif_provenance: MotifProvenanceSummary,
    #[serde(default)]
    pub(crate) hinge: MotifHingeSummary,
    #[serde(default)]
    pub(crate) routing: MotifRoutingSummary,
    #[serde(default)]
    pub(crate) task_anchor: TaskAnchorSummary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) motif_carry_forward: Option<MotifCarryForwardSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) comparison_to_previous: Option<MotifContinuityComparison>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MotifContinuityComparison {
    pub(crate) previous_runtime_mode: String,
    pub(crate) carry_forward_delta: f32,
    pub(crate) mean_similarity_delta: f32,
    pub(crate) organic_promoted_delta: i32,
    pub(crate) recovered_promoted_delta: i32,
    pub(crate) restored_compact_delta: i32,
    pub(crate) verdict: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct HumanTestEvalArtifact {
    pub(crate) version: String,
    pub(crate) runtime_mode: String,
    pub(crate) model_path: String,
    pub(crate) model_size: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) params_billions: Option<f32>,
    pub(crate) embedding_source: String,
    pub(crate) restored_run: bool,
    pub(crate) turn_count: usize,
    pub(crate) max_steps: usize,
    pub(crate) assistant_preview: String,
    pub(crate) motif_provenance: MotifProvenanceSummary,
    #[serde(default)]
    pub(crate) hinge: MotifHingeSummary,
    #[serde(default)]
    pub(crate) routing: MotifRoutingSummary,
    #[serde(default)]
    pub(crate) hinge_correlation: HingeCorrelationSummary,
    #[serde(default)]
    pub(crate) task_anchor: TaskAnchorSummary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) motif_carry_forward: Option<MotifCarryForwardSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) continuity_verdict: Option<String>,
    #[serde(default)]
    pub(crate) review_flags: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeEvalArtifacts {
    pub(crate) motif_provenance: MotifProvenanceSummary,
    pub(crate) motif_carry_forward: Option<MotifCarryForwardSummary>,
    pub(crate) hinge_summary: MotifHingeSummary,
    pub(crate) motif_continuity_artifact: MotifContinuityArtifact,
    pub(crate) hinge_correlation: HingeCorrelationSummary,
    pub(crate) routing_summary: MotifRoutingSummary,
    pub(crate) task_anchor_summary: TaskAnchorSummary,
    pub(crate) human_eval_artifact: HumanTestEvalArtifact,
    pub(crate) hinge_window_artifact: HingeWindowArtifact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct StatePacket {
    pub(crate) version: String,
    pub(crate) runtime_mode: String,
    pub(crate) hidden_anchor: StatePacketHiddenAnchor,
    pub(crate) motif_state: StatePacketMotifState,
    pub(crate) recovery_state: StatePacketRecoveryState,
    pub(crate) motion_state: StatePacketMotionState,
    pub(crate) interaction_state: StatePacketInteractionState,
    pub(crate) topology_state: StatePacketTopologyState,
    #[serde(default)]
    pub(crate) sentence_history: Vec<SentenceParticleSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) secret_sauce: Option<StatePacketSecretSauce>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct StatePacketHiddenAnchor {
    #[serde(default)]
    pub(crate) goal_embedding: Option<Vec<f32>>,
    #[serde(default)]
    pub(crate) momentum_buffer: Option<Vec<f32>>,
    #[serde(default)]
    pub(crate) secret_sauce_hidden_prior: Option<Vec<f32>>,
    #[serde(default)]
    pub(crate) secret_sauce_sentence_prior: Option<Vec<f32>>,
    #[serde(default)]
    pub(crate) secret_sauce_momentum_prior: Option<Vec<f32>>,
    #[serde(default)]
    pub(crate) secret_sauce_version: Option<SecretSauceVersion>,
    #[serde(default)]
    pub(crate) secret_sauce_decay_steps: usize,
    #[serde(default)]
    pub(crate) secret_sauce_steps_remaining: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct StatePacketMotifState {
    #[serde(default)]
    pub(crate) runtime_motifs: Vec<RuntimeMotifSnapshot>,
    #[serde(default)]
    pub(crate) live_count: usize,
    #[serde(default)]
    pub(crate) promoted_count: usize,
    #[serde(default)]
    pub(crate) bridge_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct StatePacketRecoveryState {
    #[serde(default)]
    pub(crate) runtime_recovery_ops: Vec<RuntimeRecoverySnapshot>,
    #[serde(default)]
    pub(crate) last_recovery_mag: f32,
    #[serde(default)]
    pub(crate) last_absence_signal: f32,
    #[serde(default)]
    pub(crate) last_trap_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct StatePacketMotionState {
    #[serde(default)]
    pub(crate) physics_blend: f32,
    #[serde(default)]
    pub(crate) dynamic_gravity: f32,
    #[serde(default)]
    pub(crate) dynamic_repulsion: f32,
    #[serde(default)]
    pub(crate) orbital_active: bool,
    #[serde(default)]
    pub(crate) last_motif_mag: f32,
    #[serde(default)]
    pub(crate) last_guardrail_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct StatePacketInteractionState {
    #[serde(default)]
    pub(crate) stress_level: f32,
    #[serde(default)]
    pub(crate) boredom_level: f32,
    #[serde(default)]
    pub(crate) empathy_spike: f32,
    #[serde(default)]
    pub(crate) request_count: usize,
    #[serde(default)]
    pub(crate) insight_persistence: usize,
    #[serde(default)]
    pub(crate) pending_insight: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct StatePacketTopologyState {
    #[serde(default)]
    pub(crate) live_motif_count: usize,
    #[serde(default)]
    pub(crate) live_motif_distance: f32,
    #[serde(default)]
    pub(crate) live_motif_radius: f32,
    #[serde(default)]
    pub(crate) live_basin_pressure: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct StatePacketSecretSauce {
    pub(crate) version: SecretSauceVersion,
    pub(crate) unicode_string: String,
    pub(crate) segments: SecretSauceSegments,
    pub(crate) vector_64d: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EngineStateSnapshot {
    #[serde(default)]
    pub(crate) sentence_history: Vec<SentenceParticleSnapshot>,
    #[serde(default)]
    pub(crate) runtime_motifs: Vec<RuntimeMotifSnapshot>,
    #[serde(default)]
    pub(crate) runtime_recovery_ops: Vec<RuntimeRecoverySnapshot>,
    #[serde(default)]
    pub(crate) goal_embedding: Option<Vec<f32>>,
    #[serde(default)]
    pub(crate) momentum_buffer: Option<Vec<f32>>,
    #[serde(default)]
    pub(crate) secret_sauce_hidden_prior: Option<Vec<f32>>,
    #[serde(default)]
    pub(crate) secret_sauce_sentence_prior: Option<Vec<f32>>,
    #[serde(default)]
    pub(crate) secret_sauce_momentum_prior: Option<Vec<f32>>,
    #[serde(default)]
    pub(crate) secret_sauce_version: Option<SecretSauceVersion>,
    pub(crate) secret_sauce_decay_steps: usize,
    pub(crate) secret_sauce_steps_remaining: usize,
    pub(crate) physics_blend: f32,
    pub(crate) dynamic_gravity: f32,
    pub(crate) dynamic_repulsion: f32,
    pub(crate) stress_level: f32,
    pub(crate) boredom_level: f32,
    pub(crate) empathy_spike: f32,
    pub(crate) last_motif_mag: f32,
    pub(crate) last_recovery_mag: f32,
    pub(crate) last_absence_signal: f32,
    pub(crate) last_trap_score: f32,
    pub(crate) last_guardrail_active: bool,
    pub(crate) orbital_active: bool,
    pub(crate) request_count: usize,
    pub(crate) insight_persistence: usize,
    #[serde(default)]
    pub(crate) pending_insight: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SentenceParticleSnapshot {
    pub(crate) position: Vec<f32>,
    pub(crate) velocity: Vec<f32>,
    pub(crate) mass: f32,
    pub(crate) birth_step: usize,
    pub(crate) token_count: usize,
    pub(crate) m_coh: f32,
    pub(crate) m_struct: f32,
    pub(crate) m_quantum: f32,
    pub(crate) m_geometric: f32,
    pub(crate) m_emo: f32,
    pub(crate) fitness: f32,
    pub(crate) text: String,
    pub(crate) is_attractor: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RuntimeMotifSnapshot {
    pub(crate) motif_id: String,
    pub(crate) source: String,
    #[serde(default = "default_motif_kind_bridge")]
    pub(crate) motif_kind: String,
    #[serde(default = "default_motif_status_imported")]
    pub(crate) promotion_status: String,
    #[serde(default)]
    pub(crate) raw_signature: Vec<f32>,
    pub(crate) vector: Vec<f32>,
    #[serde(default = "default_live_motif_member_count")]
    pub(crate) member_count: usize,
    #[serde(default)]
    pub(crate) last_updated_step: usize,
    pub(crate) persistence_score: f32,
    pub(crate) readiness_score: f32,
    pub(crate) injection_strength: f32,
    pub(crate) max_pre_energy: f32,
    pub(crate) flip_rate: f32,
    pub(crate) orbit_count: f32,
    pub(crate) radius_mean: f32,
    pub(crate) radius_std: f32,
    #[serde(default)]
    pub(crate) radius_m2: f32,
    #[serde(default)]
    pub(crate) promotion_score: f32,
    #[serde(default)]
    pub(crate) structured_signal: f32,
    #[serde(default)]
    pub(crate) tightness_score: f32,
    #[serde(default)]
    pub(crate) conflict_ratio: f32,
    #[serde(default)]
    pub(crate) mixed_ratio: f32,
    #[serde(default)]
    pub(crate) routing_safety_score: f32,
    #[serde(default)]
    pub(crate) topology_density: f32,
    #[serde(default)]
    pub(crate) sequential_gap_rate: f32,
    #[serde(default)]
    pub(crate) fragmentation: f32,
    #[serde(default)]
    pub(crate) hole_pressure: f32,
    #[serde(default)]
    pub(crate) tension_anchor_strength: f32,
    #[serde(default = "default_motif_role_neutral")]
    pub(crate) motif_role: String,
    #[serde(default)]
    pub(crate) controller_selected_count: usize,
    #[serde(default)]
    pub(crate) controller_rejected_count: usize,
    #[serde(default)]
    pub(crate) origin_run_id: String,
    #[serde(default)]
    pub(crate) promotion_epoch: usize,
    #[serde(default)]
    pub(crate) parent_motif_ids: Vec<String>,
    #[serde(default)]
    pub(crate) provenance_summary: String,
    #[serde(default)]
    pub(crate) merge_key: String,
    #[serde(default)]
    pub(crate) task_anchor_signature: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RuntimeRecoverySnapshot {
    pub(crate) specialist_id: String,
    pub(crate) source: String,
    pub(crate) motif_id: String,
    pub(crate) role: String,
    #[serde(default)]
    pub(crate) raw_signature: Vec<f32>,
    pub(crate) vector: Vec<f32>,
    pub(crate) influence_radius: f32,
    pub(crate) basin_variance: f32,
    pub(crate) persistence_score: f32,
    pub(crate) readiness_score: f32,
    pub(crate) absence_signal: f32,
    pub(crate) tension_point: f32,
    pub(crate) betti_0: f32,
    pub(crate) betti_1: f32,
    pub(crate) flip_rate: f32,
    pub(crate) orbit_count: f32,
    pub(crate) max_pre_energy: f32,
}

pub(crate) fn default_live_motif_member_count() -> usize {
    1
}

pub(crate) fn default_motif_kind_bridge() -> String {
    "bridge".to_string()
}

pub(crate) fn default_motif_status_imported() -> String {
    "imported".to_string()
}

pub(crate) fn default_motif_role_neutral() -> String {
    "neutral".to_string()
}

pub(crate) fn motif_kind_counts(motifs: &[RuntimeMotifSnapshot]) -> (usize, usize, usize) {
    let mut live_count = 0usize;
    let mut promoted_count = 0usize;
    let mut bridge_count = 0usize;
    for motif in motifs {
        match motif.motif_kind.as_str() {
            "live" => live_count += 1,
            "promoted" => promoted_count += 1,
            _ => bridge_count += 1,
        }
    }
    (live_count, promoted_count, bridge_count)
}

pub(crate) fn summarize_motif_carry_forward(
    restored_motifs: &[RuntimeMotifSnapshot],
    final_motifs: &[RuntimeMotifField],
) -> Option<MotifCarryForwardSummary> {
    let final_provenance = summarize_runtime_motif_provenance(final_motifs);
    let restored_promoted: Vec<&RuntimeMotifSnapshot> = restored_motifs
        .iter()
        .filter(|motif| motif.motif_kind == "promoted")
        .collect();
    let final_promoted: Vec<&RuntimeMotifField> = final_motifs
        .iter()
        .filter(|motif| motif.motif_kind == "promoted")
        .collect();

    if restored_promoted.is_empty() && final_promoted.is_empty() {
        return None;
    }

    let exact_id_matches = restored_promoted
        .iter()
        .filter(|restored| {
            final_promoted
                .iter()
                .any(|final_motif| final_motif.motif_id == restored.motif_id)
        })
        .count();

    let mut semantic_matches = 0usize;
    let mut similarity_sum = 0.0f32;
    for restored in &restored_promoted {
        let best_similarity = final_promoted
            .iter()
            .map(|final_motif| {
                cosine_similarity_slices(&restored.raw_signature, &final_motif.raw_signature)
            })
            .fold(-1.0f32, f32::max)
            .max(0.0);
        similarity_sum += best_similarity;
        if best_similarity >= 0.82 {
            semantic_matches += 1;
        }
    }

    let restored_count = restored_promoted.len();
    let mean_best_similarity = if restored_count > 0 {
        similarity_sum / restored_count as f32
    } else {
        0.0
    };
    let carry_forward_ratio = if restored_count > 0 {
        semantic_matches as f32 / restored_count as f32
    } else {
        0.0
    };

    Some(MotifCarryForwardSummary {
        restored_promoted_count: restored_count,
        final_promoted_count: final_promoted.len(),
        final_bridge_count: final_provenance.bridge_count,
        final_live_count: final_provenance.live_count,
        final_organic_promoted_count: final_provenance.organic_promoted_count,
        final_recovered_promoted_count: final_provenance.recovered_promoted_count,
        final_restored_compact_count: final_provenance.restored_compact_count,
        exact_id_matches,
        semantic_matches,
        mean_best_similarity,
        carry_forward_ratio,
    })
}

pub(crate) fn summarize_runtime_motif_provenance(
    motifs: &[RuntimeMotifField],
) -> MotifProvenanceSummary {
    let mut summary = MotifProvenanceSummary {
        bridge_count: 0,
        live_count: 0,
        organic_promoted_count: 0,
        recovered_promoted_count: 0,
        restored_compact_count: 0,
    };

    for motif in motifs {
        match (motif.motif_kind.as_str(), motif.promotion_status.as_str()) {
            ("bridge", _) => summary.bridge_count += 1,
            ("promoted", "restored_compact") => summary.restored_compact_count += 1,
            ("promoted", "recovered_promoted") => summary.recovered_promoted_count += 1,
            ("promoted", _) => summary.organic_promoted_count += 1,
            ("live", _) => summary.live_count += 1,
            _ => {}
        }
    }

    summary
}

pub(crate) fn runtime_motif_briefs(
    motifs: &[RuntimeMotifField],
    limit: usize,
) -> Vec<serde_json::Value> {
    motifs
        .iter()
        .take(limit)
        .map(|motif| {
            serde_json::json!({
                "motif_id": motif.motif_id,
                "source": motif.source,
                "motif_kind": motif.motif_kind,
                "promotion_status": motif.promotion_status,
                "motif_role": motif.motif_role,
                "member_count": motif.member_count,
                "radius_mean": motif.radius_mean,
                "radius_std": motif.radius_std,
                "structured_signal": motif.structured_signal,
                "tightness_score": motif.tightness_score,
                "topology_density": motif.topology_density,
                "sequential_gap_rate": motif.sequential_gap_rate,
                "routing_safety_score": motif.routing_safety_score,
                "promotion_score": motif.promotion_score,
                "persistence_score": motif.persistence_score,
                "readiness_score": motif.readiness_score,
            })
        })
        .collect()
}

pub(crate) fn motif_continuity_sidecar_path(base: &Path) -> PathBuf {
    let parent = base.parent().unwrap_or_else(|| Path::new("."));
    let stem = base
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("motif_continuity");
    parent.join(format!("{stem}.motif_continuity.json"))
}

pub(crate) fn human_test_eval_sidecar_path(base: &Path) -> PathBuf {
    let parent = base.parent().unwrap_or_else(|| Path::new("."));
    let stem = base
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("human_eval");
    parent.join(format!("{stem}.human_eval.json"))
}

pub(crate) fn hinge_window_sidecar_path(base: &Path) -> PathBuf {
    let parent = base.parent().unwrap_or_else(|| Path::new("."));
    let stem = base
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("hinge_window");
    parent.join(format!("{stem}.hinge_window.json"))
}

pub(crate) fn codec_trace_sidecar_path(base: &Path) -> PathBuf {
    let parent = base.parent().unwrap_or_else(|| Path::new("."));
    let stem = base
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("codec_trace");
    parent.join(format!("{stem}.codec_trace.json"))
}

pub(crate) fn turn_state_capture_path(base_dir: &Path, turn_index: usize) -> PathBuf {
    base_dir.join(format!("turn_{turn_index:04}.state.json"))
}

pub(crate) fn turn_kv_state_path(base_dir: &Path, turn_index: usize) -> PathBuf {
    base_dir.join(format!("turn_{turn_index:04}.kv.json"))
}

pub(crate) fn runtime_hidden_capture_manifest_path(base_dir: &Path) -> PathBuf {
    base_dir.join("manifest.jsonl")
}

pub(crate) fn runtime_hidden_capture_vector_path(
    base_dir: &Path,
    turn_index: usize,
    step: usize,
) -> PathBuf {
    base_dir
        .join("vectors")
        .join(format!("turn_{turn_index:04}_step_{step:04}.f32"))
}

pub(crate) fn write_f32_binary(path: &Path, values: &[f32]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("Failed to create binary vector parent {}", parent.display())
        })?;
    }
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes)
        .with_context(|| format!("Failed to write binary vector {}", path.display()))?;
    Ok(())
}

pub(crate) fn checksum_f32_le_md5(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    format!("{:x}", md5::compute(bytes))
}

pub(crate) fn latest_hinge_window_scalars(engine: &PrincipiaEngine) -> (f32, f32, Option<String>) {
    engine
        .hinge_window_records
        .last()
        .map(|record| {
            (
                record.neutral_basin_occupancy,
                record.structured_candidate_separation,
                record.structured_candidate_loss_reason.clone(),
            )
        })
        .unwrap_or((0.0, 0.0, None))
}

pub(crate) fn infer_runtime_hidden_window_label(
    engine: &PrincipiaEngine,
    loss_reason: Option<&str>,
    hinge_flipped: bool,
) -> String {
    if matches!(loss_reason, Some("distance_deficit")) {
        "distance_deficit".to_string()
    } else if matches!(
        engine.last_routed_motif_role.as_deref(),
        Some("structured") | Some("structured_candidate")
    ) {
        "structured_window".to_string()
    } else if hinge_flipped {
        "hinge_window".to_string()
    } else {
        "stable_window".to_string()
    }
}

pub(crate) fn task_anchor_similarity_for_capture(engine: &PrincipiaEngine) -> f32 {
    if engine.task_anchor_similarity_hinge > 0.0 {
        engine.task_anchor_similarity_hinge
    } else if engine.task_anchor_similarity_24tok > 0.0 {
        engine.task_anchor_similarity_24tok
    } else {
        engine.task_anchor_similarity_start
    }
}

pub(crate) fn write_runtime_hidden_capture(
    args: &Args,
    model_arch: LoadedModelArch,
    base_dir: &Path,
    turn_index: usize,
    step: usize,
    prompt: &str,
    hidden_shape: Vec<usize>,
    hidden_vec: Vec<f32>,
    drift_score: f32,
    ghost_vector_present: bool,
    engine: &PrincipiaEngine,
) -> Result<()> {
    std::fs::create_dir_all(base_dir).with_context(|| {
        format!(
            "Failed to create runtime hidden capture dir {}",
            base_dir.display()
        )
    })?;
    let vector_path = runtime_hidden_capture_vector_path(base_dir, turn_index, step);
    write_f32_binary(&vector_path, &hidden_vec)?;
    let hidden_checksum_md5 = checksum_f32_le_md5(&hidden_vec);

    let hidden_norm = hidden_vec
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let (hidden_64d_method, hidden_64d) = compress_runtime_hidden_64d(&hidden_vec, model_arch);
    let goal_embedding_64d = engine
        .goal_embedding
        .as_ref()
        .and_then(|tensor| compress_tensor_to_dim(tensor, 64).ok());
    let momentum_64d = engine
        .momentum_buffer
        .as_ref()
        .and_then(|tensor| compress_tensor_to_dim(tensor, 64).ok());
    let hinge_flipped = engine.first_organic_promoted_step.is_some()
        || engine.first_recovered_promoted_step.is_some();
    let (neutral_basin_occupancy, structured_candidate_separation, loss_reason) =
        latest_hinge_window_scalars(engine);
    let window_label =
        infer_runtime_hidden_window_label(engine, loss_reason.as_deref(), hinge_flipped);

    let record = RuntimeHiddenCaptureRecord {
        version: "runtime_hidden_capture_v1".to_string(),
        runtime_mode: args.runtime_mode.as_str().to_string(),
        req_id: args.req_id.clone(),
        prompt_id: format!("{}::turn_{turn_index:04}", args.req_id),
        prompt_hash: format!("{:x}", md5::compute(prompt.as_bytes())),
        seed: args.seed,
        model_path: args.model_path.clone(),
        capture_artifact_dir: base_dir.display().to_string(),
        turn_index,
        step,
        prompt: prompt.to_string(),
        capture_layer_label: "post_forward_last_token_final_hidden".to_string(),
        capture_layer_index: None,
        hidden_dim: hidden_vec.len(),
        hidden_shape,
        hidden_dtype: "f32_le".to_string(),
        hidden_path: vector_path.display().to_string(),
        hidden_checksum_md5,
        hidden_norm,
        hidden_64d_method: Some(hidden_64d_method.to_string()),
        hidden_64d,
        bridge_route_probe_64d_method: if engine.last_bridge_route_probe_64d.is_empty() {
            None
        } else {
            Some("live_bridge_probe_normalized_first64".to_string())
        },
        bridge_route_probe_64d: engine.last_bridge_route_probe_64d.clone(),
        nearest_ghost_id: engine.last_nearest_ghost_id.clone(),
        nearest_ghost_distance: engine.last_nearest_ghost_distance,
        second_nearest_ghost_distance: engine.last_second_nearest_ghost_distance,
        route_margin: engine.last_route_margin,
        goal_embedding_64d,
        momentum_64d,
        drift_score,
        ghost_vector_present,
        forces_applied: engine.last_forces_applied,
        gravity_force: engine.last_gravity_mag,
        ghost_pre_norm: engine.last_ghost_pre_norm,
        ghost_gain: engine.last_ghost_gain,
        applied_ghost_force: engine.last_applied_ghost_mag,
        goal_force: engine.last_goal_mag,
        repulsion_force: engine.last_repulsion_mag,
        motif_force: engine.last_motif_mag,
        recovery_force: engine.last_recovery_mag,
        activation_gate: engine.last_activation_gate,
        dynamic_gravity: engine.dynamic_gravity,
        dynamic_repulsion: engine.dynamic_repulsion,
        live_motif_count: engine.last_live_motif_count,
        nearest_live_motif_distance: engine.last_live_motif_distance,
        nearest_live_motif_radius: engine.last_live_motif_radius,
        live_basin_pressure: engine.last_live_basin_pressure,
        routed_motif_id: engine.last_routed_motif_id.clone(),
        routed_motif_role: engine.last_routed_motif_role.clone(),
        routed_motif_score: engine.last_routed_motif_score,
        task_anchor_similarity: task_anchor_similarity_for_capture(engine),
        hidden_request_pressure: engine.last_hidden_request_pressure,
        hidden_request_candidate: engine
            .hidden_request_candidate
            .map(|request| request.as_str().to_string()),
        last_hidden_request: engine
            .last_hidden_request
            .map(|request| request.as_str().to_string()),
        window_label,
        structured_candidate_loss_reason: loss_reason,
        hinge_flipped,
        neutral_basin_occupancy,
        structured_candidate_separation,
        controller_tick_count: engine.controller_tick_count,
        current_turn_structure_bias: engine.current_turn_structure_bias,
        reentry_clamp_steps_remaining: engine.reentry_clamp_steps_remaining,
        reentry_clamp_strength: engine.reentry_clamp_strength,
    };

    let manifest_path = runtime_hidden_capture_manifest_path(base_dir);
    let mut manifest = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&manifest_path)
        .with_context(|| {
            format!(
                "Failed to open runtime hidden manifest {}",
                manifest_path.display()
            )
        })?;
    writeln!(
        manifest,
        "{}",
        serde_json::to_string(&record)
            .with_context(|| "Failed to serialize runtime hidden capture record")?
    )
    .with_context(|| {
        format!(
            "Failed to append runtime hidden manifest {}",
            manifest_path.display()
        )
    })?;
    println!(
        " [RUNTIME_HIDDEN_CAPTURE] turn={} step={} dim={} wrote={}",
        turn_index,
        step,
        record.hidden_dim,
        vector_path.display()
    );
    Ok(())
}

pub(crate) fn write_motif_continuity_artifact(
    base_path: &Path,
    artifact: &MotifContinuityArtifact,
) -> Result<()> {
    let sidecar = motif_continuity_sidecar_path(base_path);
    if let Some(parent) = sidecar.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "Failed to create motif continuity parent {}",
                parent.display()
            )
        })?;
    }
    std::fs::write(&sidecar, serde_json::to_string_pretty(artifact)?)
        .with_context(|| format!("Failed to write motif continuity {}", sidecar.display()))?;
    println!(" [MOTIF_CONTINUITY_ARTIFACT] wrote={}", sidecar.display());
    Ok(())
}

pub(crate) fn load_motif_continuity_artifact(
    base_path: &Path,
) -> Result<Option<MotifContinuityArtifact>> {
    let sidecar = motif_continuity_sidecar_path(base_path);
    if !sidecar.exists() {
        return Ok(None);
    }
    let file = File::open(&sidecar)
        .with_context(|| format!("Failed to open motif continuity {}", sidecar.display()))?;
    let artifact: MotifContinuityArtifact = serde_json::from_reader(std::io::BufReader::new(file))
        .with_context(|| format!("Failed to parse motif continuity {}", sidecar.display()))?;
    Ok(Some(artifact))
}

pub(crate) fn write_human_test_eval_artifact(
    base_path: &Path,
    artifact: &HumanTestEvalArtifact,
) -> Result<()> {
    let sidecar = human_test_eval_sidecar_path(base_path);
    if let Some(parent) = sidecar.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create human eval parent {}", parent.display()))?;
    }
    std::fs::write(&sidecar, serde_json::to_string_pretty(artifact)?)
        .with_context(|| format!("Failed to write human eval {}", sidecar.display()))?;
    println!(" [HUMAN_EVAL_ARTIFACT] wrote={}", sidecar.display());
    Ok(())
}

pub(crate) fn write_hinge_window_artifact(
    base_path: &Path,
    artifact: &HingeWindowArtifact,
) -> Result<()> {
    let sidecar = hinge_window_sidecar_path(base_path);
    if let Some(parent) = sidecar.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("Failed to create hinge window parent {}", parent.display())
        })?;
    }
    std::fs::write(&sidecar, serde_json::to_string_pretty(artifact)?)
        .with_context(|| format!("Failed to write hinge window {}", sidecar.display()))?;
    println!(" [HINGE_WINDOW_ARTIFACT] wrote={}", sidecar.display());
    Ok(())
}

pub(crate) fn write_codec_trace_artifact(
    base_path: &Path,
    artifact: &CodecTraceArtifact,
) -> Result<()> {
    let sidecar = codec_trace_sidecar_path(base_path);
    if let Some(parent) = sidecar.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create codec trace parent {}", parent.display()))?;
    }
    std::fs::write(&sidecar, serde_json::to_string_pretty(artifact)?)
        .with_context(|| format!("Failed to write codec trace {}", sidecar.display()))?;
    println!(" [CODEC_TRACE_ARTIFACT] wrote={}", sidecar.display());
    Ok(())
}

pub(crate) fn compare_motif_continuity(
    previous: &MotifContinuityArtifact,
    current: &MotifContinuityArtifact,
) -> MotifContinuityComparison {
    let previous_carry = previous
        .motif_carry_forward
        .as_ref()
        .map(|summary| summary.carry_forward_ratio)
        .unwrap_or(0.0);
    let current_carry = current
        .motif_carry_forward
        .as_ref()
        .map(|summary| summary.carry_forward_ratio)
        .unwrap_or(0.0);
    let previous_similarity = previous
        .motif_carry_forward
        .as_ref()
        .map(|summary| summary.mean_best_similarity)
        .unwrap_or(0.0);
    let current_similarity = current
        .motif_carry_forward
        .as_ref()
        .map(|summary| summary.mean_best_similarity)
        .unwrap_or(0.0);

    let carry_forward_delta = current_carry - previous_carry;
    let mean_similarity_delta = current_similarity - previous_similarity;
    let organic_promoted_delta = current.motif_provenance.organic_promoted_count as i32
        - previous.motif_provenance.organic_promoted_count as i32;
    let recovered_promoted_delta = current.motif_provenance.recovered_promoted_count as i32
        - previous.motif_provenance.recovered_promoted_count as i32;
    let restored_compact_delta = current.motif_provenance.restored_compact_count as i32
        - previous.motif_provenance.restored_compact_count as i32;

    let verdict = if carry_forward_delta >= 0.05 || mean_similarity_delta >= 0.03 {
        "improved"
    } else if carry_forward_delta <= -0.05 || mean_similarity_delta <= -0.03 {
        "regressed"
    } else if organic_promoted_delta != 0
        || recovered_promoted_delta != 0
        || restored_compact_delta != 0
    {
        "drifted"
    } else {
        "stable"
    };

    MotifContinuityComparison {
        previous_runtime_mode: previous.runtime_mode.clone(),
        carry_forward_delta,
        mean_similarity_delta,
        organic_promoted_delta,
        recovered_promoted_delta,
        restored_compact_delta,
        verdict: verdict.to_string(),
    }
}

pub(crate) fn build_human_test_review_flags(
    restored_run: bool,
    motif_provenance: &MotifProvenanceSummary,
    motif_carry_forward: Option<&MotifCarryForwardSummary>,
    continuity_verdict: Option<&str>,
    hinge: Option<&MotifHingeSummary>,
    routing: Option<&MotifRoutingSummary>,
    correlation: Option<&HingeCorrelationSummary>,
) -> Vec<String> {
    let mut flags = Vec::new();
    if restored_run {
        flags.push("restored_run".to_string());
    }
    if motif_provenance.organic_promoted_count > 0 {
        flags.push("organic_promoted_present".to_string());
    } else {
        flags.push("organic_promoted_missing".to_string());
    }
    if motif_provenance.recovered_promoted_count > 0 {
        flags.push("recovered_promoted_present".to_string());
    }
    if motif_provenance.restored_compact_count > 0 {
        flags.push("restored_compact_present".to_string());
    }
    if motif_provenance.organic_promoted_count + motif_provenance.recovered_promoted_count == 0 {
        flags.push("portable_memory_thin".to_string());
    }
    if motif_provenance.restored_compact_count
        > motif_provenance.organic_promoted_count + motif_provenance.recovered_promoted_count
    {
        flags.push("restored_anchor_dominant".to_string());
    }
    if let Some(summary) = motif_carry_forward {
        if summary.carry_forward_ratio >= MOTIF_CARRY_FORWARD_ASSIST_RATIO_THRESHOLD
            && summary.mean_best_similarity >= MOTIF_CARRY_FORWARD_ASSIST_SIM_THRESHOLD
        {
            flags.push("carry_forward_strong".to_string());
        } else {
            flags.push("carry_forward_weak".to_string());
        }
    }
    if let Some(verdict) = continuity_verdict {
        flags.push(format!("continuity_{verdict}"));
    }
    if let Some(hinge) = hinge {
        if hinge.hinge_flipped {
            flags.push("hinge_flipped".to_string());
        } else {
            flags.push("hinge_not_flipped".to_string());
        }
        if hinge.organic_promoted_observed {
            flags.push("organic_promoted_observed".to_string());
        }
        if hinge.recovered_promoted_observed {
            flags.push("recovered_promoted_observed".to_string());
        }
        if hinge.promotion_attempt_count > 0 {
            flags.push(format!(
                "promotion_attempts_{}",
                hinge.promotion_attempt_count
            ));
        }
    }
    if let Some(routing) = routing {
        if routing.controller_tick_count > 0 {
            flags.push(format!(
                "controller_ticks_{}",
                routing.controller_tick_count
            ));
        }
        if routing.controller_selected_structured_count > 0 {
            flags.push("structured_basin_selected".to_string());
        }
        if routing.controller_selected_structured_candidate_count > 0 {
            flags.push("structured_candidate_selected".to_string());
        }
        if routing.structured_candidate_escalation_wins > 0 {
            flags.push("structured_candidate_escalated".to_string());
        }
        if routing.controller_selected_conversational_count > 0 {
            flags.push("conversational_basin_selected".to_string());
        }
        if routing.wrong_basin_lock_suspected {
            flags.push("wrong_basin_lock_suspected".to_string());
        }
        if routing.routing_improved_vs_previous {
            flags.push("routing_improved".to_string());
        }
    }
    if let Some(correlation) = correlation {
        if correlation.task_detected {
            flags.push("structured_task_detected".to_string());
        }
        if correlation.task_success {
            flags.push("task_success".to_string());
        } else if correlation.task_near_miss {
            flags.push("task_near_miss".to_string());
        }
        if correlation.hinge_task_success {
            flags.push("hinge_task_success".to_string());
        }
    }
    flags
}

pub(crate) fn build_motif_hinge_summary(
    restored_run: bool,
    initial_motif_provenance: Option<&MotifProvenanceSummary>,
    final_motif_provenance: &MotifProvenanceSummary,
    engine: &PrincipiaEngine,
) -> MotifHingeSummary {
    let initial = initial_motif_provenance.cloned().unwrap_or_default();

    let organic_promoted_observed = final_motif_provenance.organic_promoted_count > 0
        || engine.first_organic_promoted_step.is_some();
    let recovered_promoted_observed = final_motif_provenance.recovered_promoted_count > 0
        || engine.first_recovered_promoted_step.is_some();

    let organic_promoted_timing = if restored_run && initial.organic_promoted_count > 0 {
        Some("before_resumed_answer".to_string())
    } else if engine.first_organic_promoted_step.is_some()
        || final_motif_provenance.organic_promoted_count > initial.organic_promoted_count
    {
        Some(
            if restored_run {
                "during_resumed_answer"
            } else {
                "during_answer"
            }
            .to_string(),
        )
    } else {
        None
    };

    let recovered_promoted_timing = if restored_run && initial.recovered_promoted_count > 0 {
        Some("before_resumed_answer".to_string())
    } else if engine.first_recovered_promoted_step.is_some()
        || final_motif_provenance.recovered_promoted_count > initial.recovered_promoted_count
    {
        Some(
            if restored_run {
                "during_resumed_answer"
            } else {
                "during_answer"
            }
            .to_string(),
        )
    } else {
        None
    };

    MotifHingeSummary {
        organic_promoted_observed,
        recovered_promoted_observed,
        hinge_flipped: organic_promoted_observed || recovered_promoted_observed,
        organic_promoted_timing,
        recovered_promoted_timing,
        promotion_attempt_count: engine.promotion_attempt_count,
        promotion_failure_count: engine.promotion_failure_count,
        structured_streak_peak: engine.max_structured_streak,
    }
}

pub(crate) fn routing_improved_vs_previous(
    previous: Option<&MotifRoutingSummary>,
    current: &MotifRoutingSummary,
) -> bool {
    let Some(previous) = previous else {
        return false;
    };
    (!current.wrong_basin_lock_suspected && previous.wrong_basin_lock_suspected)
        || current.controller_selected_structured_count
            > previous.controller_selected_structured_count
        || current.controller_selected_structured_candidate_count
            > previous.controller_selected_structured_candidate_count
        || current.controller_selected_conversational_count
            < previous.controller_selected_conversational_count
}

pub(crate) fn build_task_anchor_summary(engine: &PrincipiaEngine) -> TaskAnchorSummary {
    TaskAnchorSummary {
        present: engine.current_task_anchor_signature.is_some(),
        similarity_start: engine.task_anchor_similarity_start,
        similarity_hinge: engine.task_anchor_similarity_hinge,
        similarity_24tok: engine.task_anchor_similarity_24tok,
        drift: engine.task_anchor_drift,
    }
}

pub(crate) fn build_motif_routing_summary(
    engine: &PrincipiaEngine,
    hinge: &MotifHingeSummary,
    correlation: &HingeCorrelationSummary,
    previous: Option<&MotifRoutingSummary>,
) -> MotifRoutingSummary {
    let wrong_basin_lock_suspected = correlation.task_detected
        && engine.restored_run_active
        && engine.current_turn_structure_bias >= STRUCTURED_REENTRY_PROMPT_THRESHOLD
        && !hinge.organic_promoted_observed
        && !hinge.recovered_promoted_observed
        && engine.structured_resume_conversational_hits >= 2
        && (!correlation.task_success || correlation.task_near_miss);

    let mut summary = MotifRoutingSummary {
        controller_tick_count: engine.controller_tick_count,
        controller_selected_structured_count: engine.controller_selected_structured_count,
        controller_selected_structured_candidate_count: engine
            .controller_selected_structured_candidate_count,
        controller_selected_conversational_count: engine.controller_selected_conversational_count,
        conflict_tie_break_count: engine.conflict_tie_break_count,
        structured_basin_lock_count: engine.structured_basin_lock_count,
        neutral_basin_penalty_applied: engine.neutral_basin_penalty_applied,
        task_utility_bonus_applied: engine.task_utility_bonus_applied,
        structured_candidate_escalation_attempts: engine.structured_candidate_escalation_attempts,
        structured_candidate_escalation_wins: engine.structured_candidate_escalation_wins,
        wrong_basin_lock_suspected,
        routing_improved_vs_previous: false,
        structured_candidate_loss_reason_counts: engine
            .structured_candidate_loss_reason_counts
            .clone(),
        current_routed_motif_id: engine.last_routed_motif_id.clone(),
        current_routed_motif_role: engine.last_routed_motif_role.clone(),
    };
    summary.routing_improved_vs_previous = routing_improved_vs_previous(previous, &summary);
    summary
}

pub(crate) fn detect_pattern_task_correlation(
    user_prompt: &str,
    assistant_text: &str,
    hinge: &MotifHingeSummary,
) -> HingeCorrelationSummary {
    let lower_prompt = user_prompt.to_ascii_lowercase();
    let task_detected = lower_prompt.contains("[1, 2, 3, 4, 5]")
        || lower_prompt.contains("[1,2,3,4,5]")
        || lower_prompt.contains("pattern task")
        || lower_prompt.contains("arc-agi")
        || lower_prompt.contains("-> ?");

    if !task_detected {
        return HingeCorrelationSummary::default();
    }

    let expected_answer = "[5, 4, 3, 2, 1, 5]".to_string();
    let lower_answer = assistant_text.to_ascii_lowercase();
    let exact_success =
        lower_answer.contains("[5, 4, 3, 2, 1, 5]") || lower_answer.contains("[5,4,3,2,1,5]");
    let reversal_phrase = lower_answer.contains("reverse")
        || lower_answer.contains("reversed")
        || lower_answer.contains("mirror")
        || lower_answer.contains("last element is repeated")
        || lower_answer.contains("append the last")
        || lower_answer.contains("append 5");
    let wrong_shift_phrase = lower_answer.contains("last element is moved to the first")
        || lower_answer.contains("move the last element to the front")
        || lower_answer.contains("[5, 1, 2, 3, 4]")
        || lower_answer.contains("[5,1,2,3,4]");
    let task_near_miss = !exact_success && reversal_phrase && !wrong_shift_phrase;

    let observed_answer_hint = if exact_success {
        Some(expected_answer.clone())
    } else if wrong_shift_phrase {
        Some("last_to_front_shift".to_string())
    } else if task_near_miss {
        Some("reverse_rule_detected_but_output_incomplete".to_string())
    } else if lower_answer.contains("[5") || lower_answer.contains("5, 4") {
        Some("partial_pattern_overlap".to_string())
    } else {
        None
    };

    HingeCorrelationSummary {
        task_detected,
        expected_answer: Some(expected_answer),
        task_success: exact_success,
        task_near_miss,
        observed_answer_hint,
        hinge_task_success: hinge.hinge_flipped && exact_success,
        recovered_promoted_and_success: hinge.recovered_promoted_observed && exact_success,
        organic_promoted_and_success: hinge.organic_promoted_observed && exact_success,
    }
}

pub(crate) fn build_human_test_eval_artifact(
    args: &Args,
    scaling_profile: Option<&ModelScalingProfile>,
    embedding_source: &str,
    restored_run: bool,
    turn_count: usize,
    assistant_preview: String,
    motif_provenance: &MotifProvenanceSummary,
    hinge: &MotifHingeSummary,
    routing: &MotifRoutingSummary,
    hinge_correlation: &HingeCorrelationSummary,
    task_anchor: &TaskAnchorSummary,
    motif_carry_forward: Option<&MotifCarryForwardSummary>,
    continuity_verdict: Option<&str>,
) -> HumanTestEvalArtifact {
    HumanTestEvalArtifact {
        version: "human_eval_v2".to_string(),
        runtime_mode: args.runtime_mode.as_str().to_string(),
        model_path: args.model_path.clone(),
        model_size: args.model_size.clone(),
        params_billions: scaling_profile
            .map(|profile| profile.params_billions)
            .or_else(|| parse_params_billions(&args.model_size)),
        embedding_source: embedding_source.to_string(),
        restored_run,
        turn_count,
        max_steps: args.max_steps,
        assistant_preview,
        motif_provenance: motif_provenance.clone(),
        hinge: hinge.clone(),
        routing: routing.clone(),
        hinge_correlation: hinge_correlation.clone(),
        task_anchor: task_anchor.clone(),
        motif_carry_forward: motif_carry_forward.cloned(),
        continuity_verdict: continuity_verdict.map(|s| s.to_string()),
        review_flags: build_human_test_review_flags(
            restored_run,
            motif_provenance,
            motif_carry_forward,
            continuity_verdict,
            Some(hinge),
            Some(routing),
            Some(hinge_correlation),
        ),
    }
}

pub(crate) fn build_hinge_window_artifact(
    args: &Args,
    restored_run: bool,
    hinge: &MotifHingeSummary,
    task_anchor: &TaskAnchorSummary,
    engine: &PrincipiaEngine,
) -> HingeWindowArtifact {
    let neutral_basin_occupancy = engine
        .hinge_window_records
        .iter()
        .map(|record| record.neutral_basin_occupancy)
        .fold(0.0f32, f32::max);
    let structured_candidate_separation = engine
        .hinge_window_records
        .iter()
        .map(|record| record.structured_candidate_separation)
        .fold(0.0f32, f32::max);
    HingeWindowArtifact {
        version: "hinge_window_v1".to_string(),
        runtime_mode: args.runtime_mode.as_str().to_string(),
        restored_run,
        hinge_flipped: hinge.hinge_flipped,
        first_promotion_attempt_step: engine.first_promotion_attempt_step,
        first_hinge_step: engine
            .first_organic_promoted_step
            .or(engine.first_recovered_promoted_step),
        task_anchor: task_anchor.clone(),
        neutral_basin_occupancy,
        structured_candidate_separation,
        records: engine.hinge_window_records.clone(),
    }
}

pub(crate) fn build_runtime_eval_artifacts(
    args: &Args,
    scaling_profile: Option<&ModelScalingProfile>,
    embedding_source: &str,
    restored_run: bool,
    turn_count: usize,
    last_prompt: &str,
    last_assistant: &str,
    assistant_preview: String,
    engine: &PrincipiaEngine,
    previous_motif_continuity_artifact: Option<&MotifContinuityArtifact>,
    restored_reference_motifs: Option<&Vec<RuntimeMotifSnapshot>>,
    initial_restored_motif_provenance: Option<&MotifProvenanceSummary>,
) -> RuntimeEvalArtifacts {
    let motif_provenance = summarize_runtime_motif_provenance(&engine.runtime_motifs);
    let motif_carry_forward = restored_reference_motifs
        .as_ref()
        .and_then(|restored| summarize_motif_carry_forward(restored, &engine.runtime_motifs));
    let hinge_summary = build_motif_hinge_summary(
        restored_run,
        initial_restored_motif_provenance,
        &motif_provenance,
        engine,
    );
    let mut motif_continuity_artifact = MotifContinuityArtifact {
        version: "motif_continuity_v2".to_string(),
        runtime_mode: format!("{:?}", args.runtime_mode),
        restored_run,
        max_steps: args.max_steps,
        motif_provenance: motif_provenance.clone(),
        hinge: hinge_summary.clone(),
        routing: MotifRoutingSummary::default(),
        task_anchor: TaskAnchorSummary::default(),
        motif_carry_forward: motif_carry_forward.clone(),
        comparison_to_previous: None,
    };
    if let Some(previous) = previous_motif_continuity_artifact {
        motif_continuity_artifact.comparison_to_previous = Some(compare_motif_continuity(
            previous,
            &motif_continuity_artifact,
        ));
    }

    let hinge_correlation =
        detect_pattern_task_correlation(last_prompt, last_assistant, &hinge_summary);
    let routing_summary = build_motif_routing_summary(
        engine,
        &hinge_summary,
        &hinge_correlation,
        previous_motif_continuity_artifact.map(|artifact| &artifact.routing),
    );
    let task_anchor_summary = build_task_anchor_summary(engine);
    motif_continuity_artifact.routing = routing_summary.clone();
    motif_continuity_artifact.task_anchor = task_anchor_summary.clone();
    let human_eval_artifact = build_human_test_eval_artifact(
        args,
        scaling_profile,
        embedding_source,
        restored_run,
        turn_count,
        assistant_preview,
        &motif_provenance,
        &hinge_summary,
        &routing_summary,
        &hinge_correlation,
        &task_anchor_summary,
        motif_carry_forward.as_ref(),
        motif_continuity_artifact
            .comparison_to_previous
            .as_ref()
            .map(|comparison| comparison.verdict.as_str()),
    );
    let hinge_window_artifact = build_hinge_window_artifact(
        args,
        restored_run,
        &hinge_summary,
        &task_anchor_summary,
        engine,
    );

    RuntimeEvalArtifacts {
        motif_provenance,
        motif_carry_forward,
        hinge_summary,
        motif_continuity_artifact,
        hinge_correlation,
        routing_summary,
        task_anchor_summary,
        human_eval_artifact,
        hinge_window_artifact,
    }
}

pub(crate) fn write_turn_capture_artifacts(
    args: &Args,
    scaling_profile: Option<&ModelScalingProfile>,
    embedding_source: &str,
    restored_run: bool,
    turn_index: usize,
    turn_count: usize,
    user_prompt: &str,
    assistant_text: &str,
    final_hidden_capture: Option<&Vec<f32>>,
    final_secret_sauce_segments: Option<&SecretSauceSegments>,
    final_secret_sauce_version: Option<SecretSauceVersion>,
    final_secret_sauce: Option<&str>,
    hidden_dim: usize,
    index_pos: usize,
    model: &ModelWrapper,
    engine: &PrincipiaEngine,
    previous_motif_continuity_artifact: Option<&MotifContinuityArtifact>,
    restored_reference_motifs: Option<&Vec<RuntimeMotifSnapshot>>,
    initial_restored_motif_provenance: Option<&MotifProvenanceSummary>,
    output_dir: &Path,
    write_kv_snapshot: bool,
) -> Result<MotifContinuityArtifact> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create turn capture dir {}", output_dir.display()))?;

    let assistant_preview = assistant_text.chars().take(200).collect::<String>();
    let eval = build_runtime_eval_artifacts(
        args,
        scaling_profile,
        embedding_source,
        restored_run,
        turn_count,
        user_prompt,
        assistant_text,
        assistant_preview,
        engine,
        previous_motif_continuity_artifact,
        restored_reference_motifs,
        initial_restored_motif_provenance,
    );

    let mut wrote_state = false;
    let mut wrote_kv = false;

    if let (Some(vector_64d), Some(segments), Some(secret_sauce_version), Some(secret_sauce)) = (
        final_hidden_capture,
        final_secret_sauce_segments,
        final_secret_sauce_version,
        final_secret_sauce,
    ) {
        let state_path = turn_state_capture_path(output_dir, turn_index);
        let packet_secret_sauce = Some(build_state_packet_secret_sauce(
            secret_sauce_version,
            secret_sauce.to_string(),
            segments.clone(),
            vector_64d.clone(),
        ));
        let captured_state_packet =
            StatePacket::capture(engine, hidden_dim, packet_secret_sauce.clone())?;
        let capture = StateCaptureRecord {
            turn_index,
            token_count: assistant_text.split_whitespace().count(),
            hidden_dim,
            compressed_dim: vector_64d.len(),
            compression: "block_mean_4096_to_64_dry_run".to_string(),
            secret_sauce_version: secret_sauce_version.as_str().to_string(),
            unicode_string: secret_sauce.to_string(),
            segments: segments.clone(),
            vector_64d: vector_64d.clone(),
            assistant_preview: assistant_text.chars().take(160).collect(),
            state_packet: Some(captured_state_packet.clone()),
            motif_provenance: Some(eval.motif_provenance.clone()),
            motif_carry_forward: eval.motif_carry_forward.clone(),
        };
        std::fs::write(&state_path, serde_json::to_string_pretty(&capture)?).with_context(
            || {
                format!(
                    "Failed to write turn state capture {}",
                    state_path.display()
                )
            },
        )?;
        if let Some(codec_trace_artifact) = build_codec_trace_artifact(
            &state_path,
            args.runtime_mode.as_str(),
            turn_index,
            capture.token_count,
            hidden_dim,
            &captured_state_packet,
            vector_64d,
            &capture.unicode_string,
            &eval.hinge_summary,
            &eval.routing_summary,
            &eval.task_anchor_summary,
            &eval.hinge_window_artifact,
        ) {
            write_codec_trace_artifact(&state_path, &codec_trace_artifact)?;
        }
        write_motif_continuity_artifact(&state_path, &eval.motif_continuity_artifact)?;
        write_human_test_eval_artifact(&state_path, &eval.human_eval_artifact)?;
        write_hinge_window_artifact(&state_path, &eval.hinge_window_artifact)?;
        wrote_state = true;

        if write_kv_snapshot {
            let kv_path = turn_kv_state_path(output_dir, turn_index);
            let snapshot = KvStateRecord {
                version: "kv_state_v1".to_string(),
                runtime_mode: format!("{:?}", args.runtime_mode),
                index_pos,
                hidden_dim,
                assistant_preview: assistant_text.chars().take(160).collect(),
                kv_cache: model.export_kv_cache_snapshot()?,
                state_packet: Some(StatePacket::capture(
                    engine,
                    hidden_dim,
                    packet_secret_sauce,
                )?),
                engine_state: Some(EngineStateSnapshot::capture(engine, hidden_dim)?),
                motif_provenance: Some(eval.motif_provenance.clone()),
                motif_carry_forward: eval.motif_carry_forward.clone(),
            };
            std::fs::write(&kv_path, serde_json::to_string_pretty(&snapshot)?)
                .with_context(|| format!("Failed to write turn kv state {}", kv_path.display()))?;
            if let Some(state_packet) = snapshot.state_packet.as_ref() {
                if let Some(codec_trace_artifact) = build_codec_trace_artifact(
                    &kv_path,
                    args.runtime_mode.as_str(),
                    turn_index,
                    assistant_text.split_whitespace().count(),
                    hidden_dim,
                    state_packet,
                    vector_64d,
                    secret_sauce,
                    &eval.hinge_summary,
                    &eval.routing_summary,
                    &eval.task_anchor_summary,
                    &eval.hinge_window_artifact,
                ) {
                    write_codec_trace_artifact(&kv_path, &codec_trace_artifact)?;
                }
            }
            write_motif_continuity_artifact(&kv_path, &eval.motif_continuity_artifact)?;
            write_human_test_eval_artifact(&kv_path, &eval.human_eval_artifact)?;
            write_hinge_window_artifact(&kv_path, &eval.hinge_window_artifact)?;
            wrote_kv = true;
        }
    }

    println!(
        " [TURN_CAPTURE] turn={} state={} kv={} window={} routed_role={}",
        turn_index,
        wrote_state,
        wrote_kv,
        eval.hinge_window_artifact.version,
        eval.routing_summary
            .current_routed_motif_role
            .as_deref()
            .unwrap_or("none")
    );

    Ok(eval.motif_continuity_artifact)
}

pub(crate) fn dominant_structured_candidate_loss_reason(
    routing: &MotifRoutingSummary,
) -> Option<String> {
    routing
        .structured_candidate_loss_reason_counts
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(reason, _)| reason.clone())
}

pub(crate) fn infer_codec_window_label(
    routing: &MotifRoutingSummary,
    loss_reason: Option<&str>,
    hinge: &MotifHingeSummary,
) -> String {
    if matches!(loss_reason, Some("distance_deficit")) {
        "distance_deficit".to_string()
    } else if matches!(
        routing.current_routed_motif_role.as_deref(),
        Some("structured") | Some("structured_candidate")
    ) {
        "structured_window".to_string()
    } else if hinge.hinge_flipped {
        "hinge_window".to_string()
    } else {
        "stable_window".to_string()
    }
}

pub(crate) fn build_codec_trace_artifact(
    base_path: &Path,
    runtime_mode: &str,
    turn_index: usize,
    token_count: usize,
    hidden_dim: usize,
    state_packet: &StatePacket,
    vector_64d: &[f32],
    unicode_string: &str,
    hinge: &MotifHingeSummary,
    routing: &MotifRoutingSummary,
    task_anchor: &TaskAnchorSummary,
    hinge_window: &HingeWindowArtifact,
) -> Option<CodecTraceArtifact> {
    let input_hidden = state_packet.hidden_anchor.goal_embedding.clone()?;
    let loss_reason = dominant_structured_candidate_loss_reason(routing);
    let task_anchor_similarity = if task_anchor.similarity_hinge > 0.0 {
        task_anchor.similarity_hinge
    } else if task_anchor.similarity_24tok > 0.0 {
        task_anchor.similarity_24tok
    } else {
        task_anchor.similarity_start
    };
    Some(CodecTraceArtifact {
        version: "codec_trace_v1".to_string(),
        runtime_mode: runtime_mode.to_string(),
        source_artifact: base_path.display().to_string(),
        turn_index,
        token_count,
        hidden_dim,
        input_hidden,
        input_momentum: state_packet.hidden_anchor.momentum_buffer.clone(),
        target_latent_64d: vector_64d.to_vec(),
        unicode_string: unicode_string.to_string(),
        routed_role: routing.current_routed_motif_role.clone(),
        routed_motif_id: routing.current_routed_motif_id.clone(),
        task_anchor_similarity,
        window_label: infer_codec_window_label(routing, loss_reason.as_deref(), hinge),
        loss_reason,
        hinge_flipped: hinge.hinge_flipped,
        neutral_basin_occupancy: hinge_window.neutral_basin_occupancy,
        structured_candidate_separation: hinge_window.structured_candidate_separation,
    })
}

pub(crate) fn runtime_motif_priority(kind: &str, status: &str) -> usize {
    match (kind, status) {
        ("promoted", "recovered_promoted") => 0,
        ("promoted", "restored_compact") => 1,
        ("promoted", _) => 0,
        ("bridge", _) => 2,
        ("live", "restored_context") => 3,
        ("live", "reinforcing") => 3,
        ("live", _) => 4,
        _ => 5,
    }
}

pub(crate) fn sort_runtime_motifs_by_priority(motifs: &mut [RuntimeMotifField]) {
    motifs.sort_by(|a, b| {
        runtime_motif_priority(&a.motif_kind, &a.promotion_status)
            .cmp(&runtime_motif_priority(&b.motif_kind, &b.promotion_status))
            .then_with(|| b.member_count.cmp(&a.member_count))
            .then_with(|| {
                b.promotion_score
                    .partial_cmp(&a.promotion_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
}

impl StatePacket {
    pub(crate) fn capture(
        engine: &PrincipiaEngine,
        hidden_dim: usize,
        secret_sauce: Option<StatePacketSecretSauce>,
    ) -> Result<Self> {
        let snapshot = EngineStateSnapshot::capture(engine, hidden_dim)?;
        let mut packet = Self::from_engine_snapshot(
            snapshot,
            format!("{:?}", engine.runtime_mode),
            secret_sauce,
        );
        packet.topology_state = StatePacketTopologyState {
            live_motif_count: engine.last_live_motif_count,
            live_motif_distance: engine.last_live_motif_distance,
            live_motif_radius: engine.last_live_motif_radius,
            live_basin_pressure: engine.last_live_basin_pressure,
        };
        Ok(packet)
    }

    pub(crate) fn from_engine_snapshot(
        snapshot: EngineStateSnapshot,
        runtime_mode: String,
        secret_sauce: Option<StatePacketSecretSauce>,
    ) -> Self {
        let (live_count, promoted_count, bridge_count) =
            motif_kind_counts(&snapshot.runtime_motifs);
        Self {
            version: "state_packet_v1".to_string(),
            runtime_mode,
            hidden_anchor: StatePacketHiddenAnchor {
                goal_embedding: snapshot.goal_embedding,
                momentum_buffer: snapshot.momentum_buffer,
                secret_sauce_hidden_prior: snapshot.secret_sauce_hidden_prior,
                secret_sauce_sentence_prior: snapshot.secret_sauce_sentence_prior,
                secret_sauce_momentum_prior: snapshot.secret_sauce_momentum_prior,
                secret_sauce_version: snapshot.secret_sauce_version,
                secret_sauce_decay_steps: snapshot.secret_sauce_decay_steps,
                secret_sauce_steps_remaining: snapshot.secret_sauce_steps_remaining,
            },
            motif_state: StatePacketMotifState {
                live_count,
                promoted_count,
                bridge_count,
                runtime_motifs: snapshot.runtime_motifs,
            },
            recovery_state: StatePacketRecoveryState {
                runtime_recovery_ops: snapshot.runtime_recovery_ops,
                last_recovery_mag: snapshot.last_recovery_mag,
                last_absence_signal: snapshot.last_absence_signal,
                last_trap_score: snapshot.last_trap_score,
            },
            motion_state: StatePacketMotionState {
                physics_blend: snapshot.physics_blend,
                dynamic_gravity: snapshot.dynamic_gravity,
                dynamic_repulsion: snapshot.dynamic_repulsion,
                orbital_active: snapshot.orbital_active,
                last_motif_mag: snapshot.last_motif_mag,
                last_guardrail_active: snapshot.last_guardrail_active,
            },
            interaction_state: StatePacketInteractionState {
                stress_level: snapshot.stress_level,
                boredom_level: snapshot.boredom_level,
                empathy_spike: snapshot.empathy_spike,
                request_count: snapshot.request_count,
                insight_persistence: snapshot.insight_persistence,
                pending_insight: snapshot.pending_insight,
            },
            topology_state: StatePacketTopologyState {
                live_motif_count: 0,
                live_motif_distance: 0.0,
                live_motif_radius: 0.0,
                live_basin_pressure: 0.0,
            },
            sentence_history: snapshot.sentence_history,
            secret_sauce,
        }
    }

    pub(crate) fn into_engine_snapshot(self) -> EngineStateSnapshot {
        EngineStateSnapshot {
            sentence_history: self.sentence_history,
            runtime_motifs: self.motif_state.runtime_motifs,
            runtime_recovery_ops: self.recovery_state.runtime_recovery_ops,
            goal_embedding: self.hidden_anchor.goal_embedding,
            momentum_buffer: self.hidden_anchor.momentum_buffer,
            secret_sauce_hidden_prior: self.hidden_anchor.secret_sauce_hidden_prior,
            secret_sauce_sentence_prior: self.hidden_anchor.secret_sauce_sentence_prior,
            secret_sauce_momentum_prior: self.hidden_anchor.secret_sauce_momentum_prior,
            secret_sauce_version: self.hidden_anchor.secret_sauce_version,
            secret_sauce_decay_steps: self.hidden_anchor.secret_sauce_decay_steps,
            secret_sauce_steps_remaining: self.hidden_anchor.secret_sauce_steps_remaining,
            physics_blend: self.motion_state.physics_blend,
            dynamic_gravity: self.motion_state.dynamic_gravity,
            dynamic_repulsion: self.motion_state.dynamic_repulsion,
            stress_level: self.interaction_state.stress_level,
            boredom_level: self.interaction_state.boredom_level,
            empathy_spike: self.interaction_state.empathy_spike,
            last_motif_mag: self.motion_state.last_motif_mag,
            last_recovery_mag: self.recovery_state.last_recovery_mag,
            last_absence_signal: self.recovery_state.last_absence_signal,
            last_trap_score: self.recovery_state.last_trap_score,
            last_guardrail_active: self.motion_state.last_guardrail_active,
            orbital_active: self.motion_state.orbital_active,
            request_count: self.interaction_state.request_count,
            insight_persistence: self.interaction_state.insight_persistence,
            pending_insight: self.interaction_state.pending_insight,
        }
    }

    pub(crate) fn restore_into(
        &self,
        engine: &mut PrincipiaEngine,
        hidden_dim: usize,
        device: &Device,
    ) -> Result<()> {
        let topology_state = self.topology_state.clone();
        self.clone()
            .into_engine_snapshot()
            .restore_into(engine, hidden_dim, device)?;
        engine.last_live_motif_count = topology_state.live_motif_count;
        engine.last_live_motif_distance = topology_state.live_motif_distance;
        engine.last_live_motif_radius = topology_state.live_motif_radius;
        engine.last_live_basin_pressure = topology_state.live_basin_pressure;
        Ok(())
    }
}

impl EngineStateSnapshot {
    pub(crate) fn capture(engine: &PrincipiaEngine, hidden_dim: usize) -> Result<Self> {
        let sentence_history = engine
            .sentence_history
            .iter()
            .rev()
            .take(8)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|p| -> Result<SentenceParticleSnapshot> {
                Ok(SentenceParticleSnapshot {
                    position: tensor_to_vec_f32(&p.position)?,
                    velocity: tensor_to_vec_f32(&p.velocity)?,
                    mass: p.mass,
                    birth_step: p.birth_step,
                    token_count: p.token_count,
                    m_coh: p.m_coh,
                    m_struct: p.m_struct,
                    m_quantum: p.m_quantum,
                    m_geometric: p.m_geometric,
                    m_emo: p.m_emo,
                    fitness: p.fitness,
                    text: p.text.clone(),
                    is_attractor: p.is_attractor,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let runtime_motifs = engine
            .runtime_motifs
            .iter()
            .map(|motif| -> Result<RuntimeMotifSnapshot> {
                Ok(RuntimeMotifSnapshot {
                    motif_id: motif.motif_id.clone(),
                    source: motif.source.clone(),
                    motif_kind: motif.motif_kind.clone(),
                    promotion_status: motif.promotion_status.clone(),
                    raw_signature: motif.raw_signature.clone(),
                    vector: tensor_to_vec_f32(&motif.vector)?,
                    member_count: motif.member_count,
                    last_updated_step: motif.last_updated_step,
                    persistence_score: motif.persistence_score,
                    readiness_score: motif.readiness_score,
                    injection_strength: motif.injection_strength,
                    max_pre_energy: motif.max_pre_energy,
                    flip_rate: motif.flip_rate,
                    orbit_count: motif.orbit_count,
                    radius_mean: motif.radius_mean,
                    radius_std: motif.radius_std,
                    radius_m2: motif.radius_m2,
                    promotion_score: motif.promotion_score,
                    structured_signal: motif.structured_signal,
                    tightness_score: motif.tightness_score,
                    conflict_ratio: motif.conflict_ratio,
                    mixed_ratio: motif.mixed_ratio,
                    routing_safety_score: motif.routing_safety_score,
                    topology_density: motif.topology_density,
                    sequential_gap_rate: motif.sequential_gap_rate,
                    fragmentation: motif.fragmentation,
                    hole_pressure: motif.hole_pressure,
                    tension_anchor_strength: motif.tension_anchor_strength,
                    motif_role: motif.motif_role.clone(),
                    controller_selected_count: motif.controller_selected_count,
                    controller_rejected_count: motif.controller_rejected_count,
                    origin_run_id: motif.origin_run_id.clone(),
                    promotion_epoch: motif.promotion_epoch,
                    parent_motif_ids: motif.parent_motif_ids.clone(),
                    provenance_summary: motif.provenance_summary.clone(),
                    merge_key: motif.merge_key.clone(),
                    task_anchor_signature: motif.task_anchor_signature.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let runtime_recovery_ops = engine
            .runtime_recovery_ops
            .iter()
            .map(|operator| -> Result<RuntimeRecoverySnapshot> {
                Ok(RuntimeRecoverySnapshot {
                    specialist_id: operator.specialist_id.clone(),
                    source: operator.source.clone(),
                    motif_id: operator.motif_id.clone(),
                    role: operator.role.clone(),
                    raw_signature: operator.raw_signature.clone(),
                    vector: tensor_to_vec_f32(&operator.vector)?,
                    influence_radius: operator.influence_radius,
                    basin_variance: operator.basin_variance,
                    persistence_score: operator.persistence_score,
                    readiness_score: operator.readiness_score,
                    absence_signal: operator.absence_signal,
                    tension_point: operator.tension_point,
                    betti_0: operator.betti_0,
                    betti_1: operator.betti_1,
                    flip_rate: operator.flip_rate,
                    orbit_count: operator.orbit_count,
                    max_pre_energy: operator.max_pre_energy,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let capture_vec = |opt: &Option<Tensor>| -> Result<Option<Vec<f32>>> {
            opt.as_ref().map(tensor_to_vec_f32).transpose()
        };
        let _ = hidden_dim;
        Ok(Self {
            sentence_history,
            runtime_motifs,
            runtime_recovery_ops,
            goal_embedding: capture_vec(&engine.goal_embedding)?,
            momentum_buffer: capture_vec(&engine.momentum_buffer)?,
            secret_sauce_hidden_prior: capture_vec(&engine.secret_sauce_hidden_prior)?,
            secret_sauce_sentence_prior: capture_vec(&engine.secret_sauce_sentence_prior)?,
            secret_sauce_momentum_prior: capture_vec(&engine.secret_sauce_momentum_prior)?,
            secret_sauce_version: engine.secret_sauce_version,
            secret_sauce_decay_steps: engine.secret_sauce_decay_steps,
            secret_sauce_steps_remaining: engine.secret_sauce_steps_remaining,
            physics_blend: engine.physics_blend,
            dynamic_gravity: engine.dynamic_gravity,
            dynamic_repulsion: engine.dynamic_repulsion,
            stress_level: engine.stress_level,
            boredom_level: engine.boredom_level,
            empathy_spike: engine.empathy_spike,
            last_motif_mag: engine.last_motif_mag,
            last_recovery_mag: engine.last_recovery_mag,
            last_absence_signal: engine.last_absence_signal,
            last_trap_score: engine.last_trap_score,
            last_guardrail_active: engine.last_guardrail_active,
            orbital_active: engine.orbital_active,
            request_count: engine.request_count,
            insight_persistence: engine.insight_persistence,
            pending_insight: engine.pending_insight.clone(),
        })
    }

    pub(crate) fn restore_into(
        &self,
        engine: &mut PrincipiaEngine,
        hidden_dim: usize,
        device: &Device,
    ) -> Result<()> {
        engine.sentence_history.clear();
        for p in &self.sentence_history {
            let position = vec_to_tensor_f32(&p.position, hidden_dim, device)?.detach();
            let velocity = vec_to_tensor_f32(&p.velocity, hidden_dim, device)?.detach();
            let quantum_state = position.clone();
            engine.sentence_history.push_back(SentenceParticle {
                position,
                velocity,
                mass: p.mass,
                radius: 0.1,
                birth_step: p.birth_step,
                token_count: p.token_count,
                vad: [0.5, 0.5, 0.5],
                surprisal: 1.0,
                delta: 0.0,
                m_info: 1.0,
                m_sem: 1.0,
                m_coh: p.m_coh,
                m_struct: p.m_struct,
                m_quantum: p.m_quantum,
                m_geometric: p.m_geometric,
                m_emo: p.m_emo,
                kl_delta: 0.0,
                text: p.text.clone(),
                entangled_with: BTreeMap::new(),
                quantum_state,
                fitness: p.fitness,
                latent_thought: None,
                is_attractor: p.is_attractor,
                is_repulsor: false,
                sub_particles: Vec::new(),
                is_lpm_active: true,
            });
        }
        if !self.runtime_motifs.is_empty() {
            engine.runtime_motifs = self
                .runtime_motifs
                .iter()
                .map(|motif| -> Result<RuntimeMotifField> {
                    Ok(RuntimeMotifField {
                        motif_id: motif.motif_id.clone(),
                        source: motif.source.clone(),
                        motif_kind: motif.motif_kind.clone(),
                        promotion_status: motif.promotion_status.clone(),
                        raw_signature: motif.raw_signature.clone(),
                        vector: vec_to_tensor_f32(&motif.vector, hidden_dim, device)?.detach(),
                        member_count: motif.member_count.max(1),
                        last_updated_step: motif.last_updated_step,
                        persistence_score: motif.persistence_score,
                        readiness_score: motif.readiness_score,
                        injection_strength: motif.injection_strength,
                        max_pre_energy: motif.max_pre_energy,
                        flip_rate: motif.flip_rate,
                        orbit_count: motif.orbit_count,
                        radius_mean: motif.radius_mean,
                        radius_std: motif.radius_std,
                        radius_m2: motif.radius_m2,
                        promotion_score: motif.promotion_score,
                        structured_signal: motif.structured_signal,
                        tightness_score: motif.tightness_score,
                        conflict_ratio: motif.conflict_ratio,
                        mixed_ratio: motif.mixed_ratio,
                        routing_safety_score: motif.routing_safety_score,
                        topology_density: motif.topology_density,
                        sequential_gap_rate: motif.sequential_gap_rate,
                        fragmentation: motif.fragmentation,
                        hole_pressure: motif.hole_pressure,
                        tension_anchor_strength: motif.tension_anchor_strength,
                        motif_role: motif.motif_role.clone(),
                        controller_selected_count: motif.controller_selected_count,
                        controller_rejected_count: motif.controller_rejected_count,
                        origin_run_id: motif.origin_run_id.clone(),
                        promotion_epoch: motif.promotion_epoch,
                        parent_motif_ids: motif.parent_motif_ids.clone(),
                        provenance_summary: motif.provenance_summary.clone(),
                        merge_key: motif.merge_key.clone(),
                        task_anchor_signature: motif.task_anchor_signature.clone(),
                        live_hidden_remapped: true,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            sort_runtime_motifs_by_priority(&mut engine.runtime_motifs);
            engine.refresh_runtime_motif_metadata()?;
        }
        if !self.runtime_recovery_ops.is_empty() {
            engine.runtime_recovery_ops = self
                .runtime_recovery_ops
                .iter()
                .map(|operator| -> Result<RuntimeRecoveryOperator> {
                    Ok(RuntimeRecoveryOperator {
                        specialist_id: operator.specialist_id.clone(),
                        source: operator.source.clone(),
                        motif_id: operator.motif_id.clone(),
                        role: operator.role.clone(),
                        raw_signature: operator.raw_signature.clone(),
                        vector: vec_to_tensor_f32(&operator.vector, hidden_dim, device)?.detach(),
                        influence_radius: operator.influence_radius,
                        basin_variance: operator.basin_variance,
                        persistence_score: operator.persistence_score,
                        readiness_score: operator.readiness_score,
                        absence_signal: operator.absence_signal,
                        tension_point: operator.tension_point,
                        betti_0: operator.betti_0,
                        betti_1: operator.betti_1,
                        flip_rate: operator.flip_rate,
                        orbit_count: operator.orbit_count,
                        max_pre_energy: operator.max_pre_energy,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
        }
        let restore_vec = |opt: &Option<Vec<f32>>| -> Result<Option<Tensor>> {
            opt.as_ref()
                .map(|data| vec_to_tensor_f32(data, hidden_dim, device).map(|t| t.detach()))
                .transpose()
        };
        engine.goal_embedding = restore_vec(&self.goal_embedding)?;
        engine.momentum_buffer = restore_vec(&self.momentum_buffer)?;
        engine.secret_sauce_hidden_prior = restore_vec(&self.secret_sauce_hidden_prior)?;
        engine.secret_sauce_sentence_prior = restore_vec(&self.secret_sauce_sentence_prior)?;
        engine.secret_sauce_momentum_prior = restore_vec(&self.secret_sauce_momentum_prior)?;
        engine.secret_sauce_version = self.secret_sauce_version;
        engine.secret_sauce_decay_steps = self.secret_sauce_decay_steps;
        engine.secret_sauce_steps_remaining = self.secret_sauce_steps_remaining;
        engine.physics_blend = self.physics_blend;
        engine.dynamic_gravity = self.dynamic_gravity;
        engine.dynamic_repulsion = self.dynamic_repulsion;
        engine.stress_level = self.stress_level;
        engine.boredom_level = self.boredom_level;
        engine.empathy_spike = self.empathy_spike;
        engine.last_motif_mag = self.last_motif_mag;
        engine.last_recovery_mag = self.last_recovery_mag;
        engine.last_absence_signal = self.last_absence_signal;
        engine.last_trap_score = self.last_trap_score;
        engine.last_guardrail_active = self.last_guardrail_active;
        engine.orbital_active = self.orbital_active;
        engine.request_count = self.request_count;
        engine.insight_persistence = self.insight_persistence;
        engine.pending_insight = self.pending_insight.clone();
        engine.current_sentence_embeddings.clear();
        engine.current_sentence_tokens.clear();
        engine.request_buffer.clear();
        engine.surface_buffer.clear();
        engine.hidden_request_candidate = None;
        engine.hidden_request_streak = 0;
        engine.last_hidden_request = None;
        engine.last_hidden_request_pressure = 0.0;
        engine.hidden_request_activations = 0;
        Ok(())
    }
}

// =============================================================================
// PHASE 3: THE MIRROR (Telemetry-to-Language Translator)
// Converts raw physics state into natural language insights.
// The LLM can "read its own dashboard" and self-correct.
// =============================================================================

// TELEMETRY_TO_TEXT PURGED (Physics-Only Reasoning)

// =============================================================================
// GGUF WRAPPER
// =============================================================================
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoadedModelArch {
    Llama,
    Qwen35,
}

pub(crate) enum ModelWrapper {
    Quantized(QuantizedNakedLlama, Tokenizer),
    Qwen35(QuantizedQwen35Hybrid, Tokenizer),
    Qwen35MetadataOnly(Qwen35GgufMetadata, Tokenizer),
}

impl ModelWrapper {
    pub(crate) fn tokenizer(&self) -> &Tokenizer {
        match self {
            Self::Quantized(_, tokenizer) => tokenizer,
            Self::Qwen35(_, tokenizer) => tokenizer,
            Self::Qwen35MetadataOnly(_, tokenizer) => tokenizer,
        }
    }

    pub(crate) fn embed_tokens_forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Quantized(m, _) => m
                .embed_tokens_forward(input)
                .map_err(|e| anyhow::anyhow!(e)),
            Self::Qwen35(m, _) => m
                .embed_tokens_forward(input)
                .map_err(|e| anyhow::anyhow!(e)),
            Self::Qwen35MetadataOnly(_, _) => {
                let _ = input;
                anyhow::bail!("qwen35 forward not implemented yet")
            }
        }
    }

    pub(crate) fn hidden_dim(&self) -> usize {
        match self {
            Self::Quantized(m, _) => m.hidden_dim(),
            Self::Qwen35(m, _) => m.hidden_dim(),
            Self::Qwen35MetadataOnly(metadata, _) => metadata.hidden_size,
        }
    }

    pub(crate) fn arch(&self) -> LoadedModelArch {
        match self {
            Self::Quantized(_, _) => LoadedModelArch::Llama,
            Self::Qwen35(_, _) => LoadedModelArch::Qwen35,
            Self::Qwen35MetadataOnly(_, _) => LoadedModelArch::Qwen35,
        }
    }

    pub(crate) fn qwen35_metadata(&self) -> Option<&Qwen35GgufMetadata> {
        match self {
            Self::Qwen35(model, _) => Some(model.metadata()),
            Self::Qwen35MetadataOnly(metadata, _) => Some(metadata),
            _ => None,
        }
    }

    pub(crate) fn forward_physics(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        physics: &mut impl PhysicsEngine,
        ghost_vector: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Quantized(m, _) => m
                .forward_physics(input, index_pos, physics, ghost_vector)
                .map_err(|e| anyhow::anyhow!(e)),
            Self::Qwen35(m, _) => m
                .forward_physics(input, index_pos, physics, ghost_vector)
                .map_err(|e| anyhow::anyhow!(e)),
            Self::Qwen35MetadataOnly(_, _) => {
                let _ = (input, index_pos, physics, ghost_vector);
                anyhow::bail!("qwen35 forward not implemented yet")
            }
        }
    }

    pub(crate) fn append_token(&mut self, _token: u32) {
        match self {
            Self::Quantized(_, _) => {}
            Self::Qwen35(_, _) => {}
            Self::Qwen35MetadataOnly(_, _) => {}
        }
    }

    pub(crate) fn export_kv_cache_snapshot(&self) -> Result<ModelKvCacheSnapshot> {
        match self {
            Self::Quantized(m, _) => m.export_kv_cache_snapshot().map_err(|e| anyhow::anyhow!(e)),
            Self::Qwen35(m, _) => m.export_kv_cache_snapshot(),
            Self::Qwen35MetadataOnly(_, _) => {
                anyhow::bail!("qwen35 cache snapshots not implemented yet")
            }
        }
    }

    pub(crate) fn import_kv_cache_snapshot(
        &mut self,
        snapshot: &ModelKvCacheSnapshot,
        device: &Device,
    ) -> Result<()> {
        match self {
            Self::Quantized(m, _) => m
                .import_kv_cache_snapshot(snapshot, device)
                .map_err(|e| anyhow::anyhow!(e)),
            Self::Qwen35(m, _) => m.import_kv_cache_snapshot(snapshot, device),
            Self::Qwen35MetadataOnly(_, _) => {
                let _ = (snapshot, device);
                anyhow::bail!("qwen35 cache snapshots not implemented yet")
            }
        }
    }

    /// Drop all KV cache state so the next forward pass starts fresh. Used by
    /// `--reset-kv-cache-per-turn` for multi-prompt session-script eval runs that
    /// would otherwise overflow the model's context window across cumulative turns.
    pub(crate) fn reset_kv_cache(&mut self) -> Result<()> {
        match self {
            Self::Quantized(m, _) => {
                m.clear_kv_cache();
                Ok(())
            }
            Self::Qwen35(m, _) => m.clear_kv_cache(),
            Self::Qwen35MetadataOnly(_, _) => Ok(()),
        }
    }
}

pub(crate) fn specialist_worker_answer_window_active(assistant_text: &str) -> bool {
    let Some(marker_idx) = ["VISIBLE ANSWER:", "EXACT OUTPUT:", "WORKING ANSWER:"]
        .iter()
        .filter_map(|marker| assistant_text.find(marker).map(|idx| idx + marker.len()))
        .min()
    else {
        return false;
    };
    let tail = assistant_text[marker_idx..].trim_start();
    let lower_tail = tail.to_ascii_lowercase();
    !(tail.contains("[REQUEST:")
        || tail.contains("<|")
        || lower_tail.contains("assistant")
        || lower_tail.contains("user"))
}

pub(crate) fn specialist_worker_pre_answer_active(assistant_text: &str) -> bool {
    let lower_text = assistant_text.to_ascii_lowercase();
    if assistant_text.contains("[REQUEST:")
        || assistant_text.contains("<|")
        || lower_text.contains("assistant")
        || lower_text.contains("user")
    {
        return false;
    }

    let Some(marker_idx) = ["VISIBLE ANSWER:", "EXACT OUTPUT:", "WORKING ANSWER:"]
        .iter()
        .filter_map(|marker| assistant_text.find(marker).map(|idx| idx + marker.len()))
        .min()
    else {
        return true;
    };

    assistant_text[marker_idx..].trim().is_empty()
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CountRouteMemoryFinalizationCandidate {
    pub(crate) word: String,
    pub(crate) target_letter: char,
    pub(crate) answer: String,
    pub(crate) parser_confidence: f32,
    pub(crate) parser_version: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct CountRouteMemoryFinalizationTelemetry {
    pub(crate) candidate_enabled: bool,
    pub(crate) candidate_answer: Option<String>,
    pub(crate) candidate_word: Option<String>,
    pub(crate) candidate_target_letter: Option<String>,
    pub(crate) parser_confidence: Option<f32>,
    pub(crate) parser_version: Option<String>,
    pub(crate) state: Option<String>,
    pub(crate) answer_signature_seen: Option<String>,
    pub(crate) do_no_harm_protected: bool,
    pub(crate) would_apply: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct CountRouteMemoryFinalizationAction {
    pub(crate) action_enabled: bool,
    pub(crate) action_applied: bool,
    pub(crate) action_reason: Option<String>,
    pub(crate) replacement_answer: Option<String>,
    pub(crate) original_answer_window: Option<String>,
    pub(crate) stop_reason: Option<String>,
}

pub(crate) fn parse_count_route_memory_finalization_candidate(
    prompt: &str,
) -> Option<CountRouteMemoryFinalizationCandidate> {
    let lower_prompt = prompt.to_ascii_lowercase();
    let task_text = lower_prompt
        .rfind("task:")
        .map(|idx| &prompt[idx + "task:".len()..])
        .unwrap_or(prompt);
    let lower_task_text = task_text.to_ascii_lowercase();
    parse_count_route_memory_finalization_candidate_arrow_v1(&lower_task_text)
        .or_else(|| parse_count_route_memory_finalization_candidate_natural_v2(task_text))
}

pub(crate) fn parse_count_route_memory_finalization_candidate_arrow_v1(
    task_text: &str,
) -> Option<CountRouteMemoryFinalizationCandidate> {
    let count_line = task_text
        .lines()
        .find(|line| line.contains("->") && line.contains("count "))
        .unwrap_or(task_text);
    let parts: Vec<&str> = count_line.split("->").map(str::trim).collect();
    if parts.len() < 2 {
        return None;
    }

    let word = parts[0]
        .chars()
        .filter(|ch| ch.is_ascii_alphabetic())
        .collect::<String>();
    if word.is_empty() {
        return None;
    }

    let count_part = parts
        .iter()
        .skip(1)
        .find(|part| part.starts_with("count "))?;
    let mut count_words = count_part.split_whitespace();
    if count_words.next()? != "count" {
        return None;
    }
    let target_letter = count_words
        .next()?
        .chars()
        .find(|ch| ch.is_ascii_alphabetic())?;
    if !count_part.contains("letter") {
        return None;
    }

    let answer = word
        .chars()
        .filter(|ch| *ch == target_letter)
        .count()
        .to_string();
    Some(CountRouteMemoryFinalizationCandidate {
        word,
        target_letter,
        answer,
        parser_confidence: 1.0,
        parser_version: "arrow_v1_exact".to_string(),
    })
}

pub(crate) fn natural_count_target_letter(target_token: &str) -> Option<char> {
    let letters = target_token
        .chars()
        .filter(|ch| ch.is_ascii_alphabetic())
        .collect::<String>();
    match letters.len() {
        1 => letters.chars().next().map(|ch| ch.to_ascii_lowercase()),
        2 if letters.ends_with('s')
            && letters
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_uppercase()) =>
        {
            letters.chars().next().map(|ch| ch.to_ascii_lowercase())
        }
        _ => None,
    }
}

pub(crate) fn parse_count_route_memory_finalization_candidate_natural_v2(
    task_text: &str,
) -> Option<CountRouteMemoryFinalizationCandidate> {
    let tokens: Vec<&str> = task_text
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|part| !part.is_empty())
        .collect();
    let lower_tokens: Vec<String> = tokens
        .iter()
        .map(|part| part.to_ascii_lowercase())
        .collect();
    if tokens.len() < 5 {
        return None;
    }

    if tokens.len() >= 7 {
        for window_start in 0..=tokens.len().saturating_sub(7) {
            if lower_tokens[window_start] != "how"
                || lower_tokens[window_start + 1] != "many"
                || !matches!(
                    lower_tokens[window_start + 2].as_str(),
                    "letter" | "letters" | "character" | "characters"
                )
            {
                continue;
            }
            let target_token = tokens[window_start + 3];
            if lower_tokens[window_start + 4] != "are" || lower_tokens[window_start + 5] != "in" {
                continue;
            }
            let word = tokens
                .get(window_start + 6)?
                .chars()
                .filter(|ch| ch.is_ascii_alphabetic())
                .map(|ch| ch.to_ascii_lowercase())
                .collect::<String>();
            if word.is_empty() {
                continue;
            }
            let Some(target_letter) = natural_count_target_letter(target_token) else {
                continue;
            };
            let answer = word
                .chars()
                .filter(|ch| *ch == target_letter)
                .count()
                .to_string();
            return Some(CountRouteMemoryFinalizationCandidate {
                word,
                target_letter,
                answer,
                parser_confidence: 0.75,
                parser_version: "natural_count_v2_shadow".to_string(),
            });
        }
    }

    if tokens.len() >= 7 {
        for window_start in 0..=tokens.len().saturating_sub(7) {
            if lower_tokens[window_start] != "count"
                || lower_tokens[window_start + 1] != "the"
                || lower_tokens[window_start + 2] != "number"
                || lower_tokens[window_start + 3] != "of"
            {
                continue;
            }
            let target_token = tokens[window_start + 4];
            let target_token_lower = &lower_tokens[window_start + 4];
            if lower_tokens[window_start + 5] != "in" {
                continue;
            }
            let word = tokens
                .get(window_start + 6)?
                .chars()
                .filter(|ch| ch.is_ascii_alphabetic())
                .map(|ch| ch.to_ascii_lowercase())
                .collect::<String>();
            if word.is_empty() {
                continue;
            }
            if matches!(
                target_token_lower.as_str(),
                "letter" | "letters" | "character" | "characters"
            ) {
                continue;
            }
            let Some(target_letter) = natural_count_target_letter(target_token) else {
                continue;
            };
            let answer = word
                .chars()
                .filter(|ch| *ch == target_letter)
                .count()
                .to_string();
            return Some(CountRouteMemoryFinalizationCandidate {
                word,
                target_letter,
                answer,
                parser_confidence: 0.75,
                parser_version: "natural_count_v2_shadow".to_string(),
            });
        }
    }

    if tokens.len() >= 6 {
        for window_start in 0..=tokens.len().saturating_sub(6) {
            if lower_tokens[window_start] != "count" {
                continue;
            }
            let mut cursor = window_start + 1;
            if lower_tokens.get(cursor).map(String::as_str) == Some("the") {
                cursor += 1;
            }
            if !matches!(
                lower_tokens.get(cursor).map(String::as_str),
                Some("letter" | "letters" | "character" | "characters")
            ) {
                continue;
            }
            let Some(target_token) = tokens.get(cursor + 1) else {
                continue;
            };
            if lower_tokens.get(cursor + 2).map(String::as_str) != Some("in") {
                continue;
            }
            let word = tokens
                .get(cursor + 3)?
                .chars()
                .filter(|ch| ch.is_ascii_alphabetic())
                .map(|ch| ch.to_ascii_lowercase())
                .collect::<String>();
            if word.is_empty() {
                continue;
            }
            let Some(target_letter) = natural_count_target_letter(target_token) else {
                continue;
            };
            let answer = word
                .chars()
                .filter(|ch| *ch == target_letter)
                .count()
                .to_string();
            return Some(CountRouteMemoryFinalizationCandidate {
                word,
                target_letter,
                answer,
                parser_confidence: 0.75,
                parser_version: "natural_count_v2_shadow".to_string(),
            });
        }
    }

    for window_start in 0..tokens.len() {
        if lower_tokens[window_start] != "count" {
            continue;
        }
        let Some(relative_letters_idx) = lower_tokens[window_start + 1..]
            .iter()
            .position(|part| matches!(part.as_str(), "letter" | "letters"))
        else {
            continue;
        };
        let letters_idx = window_start + 1 + relative_letters_idx;
        if letters_idx <= window_start + 1 {
            continue;
        }
        if lower_tokens.get(letters_idx + 1).map(String::as_str) != Some("in") {
            continue;
        }
        let target_terms = &tokens[window_start + 1..letters_idx];
        let target_terms_lower = &lower_tokens[window_start + 1..letters_idx];
        if target_terms
            .iter()
            .zip(target_terms_lower.iter())
            .any(|(_, term)| {
                matches!(
                    term.as_str(),
                    "the" | "number" | "of" | "letter" | "letters"
                )
            })
        {
            continue;
        }
        let target_text = target_terms.join("");
        let Some(target_letter) = natural_count_target_letter(&target_text) else {
            continue;
        };
        let word = tokens
            .get(letters_idx + 2)?
            .chars()
            .filter(|ch| ch.is_ascii_alphabetic())
            .map(|ch| ch.to_ascii_lowercase())
            .collect::<String>();
        if word.is_empty() {
            continue;
        }
        let answer = word
            .chars()
            .filter(|ch| *ch == target_letter)
            .count()
            .to_string();
        return Some(CountRouteMemoryFinalizationCandidate {
            word,
            target_letter,
            answer,
            parser_confidence: 0.75,
            parser_version: "natural_count_v2_shadow".to_string(),
        });
    }

    None
}

pub(crate) fn count_route_memory_answer_window_bounds(
    assistant_text: &str,
) -> Option<(usize, usize)> {
    let marker_idx = ["VISIBLE ANSWER:", "EXACT OUTPUT:", "WORKING ANSWER:"]
        .iter()
        .filter_map(|marker| assistant_text.find(marker).map(|idx| idx + marker.len()))
        .min()?;
    let tail = assistant_text[marker_idx..].trim_start();
    let start = marker_idx
        + assistant_text[marker_idx..]
            .len()
            .saturating_sub(tail.len());
    let lower_tail = tail.to_ascii_lowercase();
    let mut end = tail.len();
    for marker in [
        "\n[REQUEST:",
        "\n[INTERNAL",
        "\n[SYSTEM",
        "\nVISIBLE ",
        "\nWORKING ",
        "\nREQUEST ",
        "\nMETRIC_SUMMARY",
        "\n===",
    ] {
        if let Some(idx) = tail.find(marker) {
            end = end.min(idx);
        }
    }
    for marker in ["<|start_header_id|>", "assistant", "user"] {
        if let Some(idx) = lower_tail.find(marker) {
            end = end.min(idx);
        }
    }
    let raw_window = &tail[..end];
    let leading_trim = raw_window
        .len()
        .saturating_sub(raw_window.trim_start().len());
    let trailing_trim = raw_window.len().saturating_sub(raw_window.trim_end().len());
    let start = start + leading_trim;
    let end = start
        + raw_window
            .len()
            .saturating_sub(leading_trim + trailing_trim);
    if start >= end {
        None
    } else {
        Some((start, end))
    }
}

pub(crate) fn count_route_memory_answer_window(assistant_text: &str) -> Option<String> {
    let (start, end) = count_route_memory_answer_window_bounds(assistant_text)?;
    let window = assistant_text[start..end].trim();
    if window.is_empty() {
        None
    } else {
        Some(window.to_string())
    }
}

pub(crate) fn replace_count_route_memory_answer_window(
    assistant_text: &str,
    replacement_answer: &str,
) -> Option<(String, String)> {
    let (start, end) = count_route_memory_answer_window_bounds(assistant_text)?;
    let original = assistant_text[start..end].trim().to_string();
    if original.is_empty() {
        return None;
    }
    let mut replaced = String::with_capacity(
        assistant_text.len() + replacement_answer.len().saturating_sub(end - start),
    );
    replaced.push_str(&assistant_text[..start]);
    replaced.push_str(replacement_answer);
    replaced.push_str(&assistant_text[end..]);
    Some((replaced, original))
}

pub(crate) fn count_route_memory_answer_signature(text: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    let mut digits = String::new();
    for ch in lower.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
        } else if !digits.is_empty() {
            return Some(digits);
        }
    }
    if !digits.is_empty() {
        return Some(digits);
    }

    for (word, digit) in [
        ("zero", "0"),
        ("one", "1"),
        ("two", "2"),
        ("three", "3"),
        ("four", "4"),
        ("five", "5"),
        ("six", "6"),
        ("seven", "7"),
        ("eight", "8"),
        ("nine", "9"),
        ("ten", "10"),
        ("eleven", "11"),
        ("twelve", "12"),
    ] {
        if lower
            .split(|ch: char| !ch.is_ascii_alphabetic())
            .any(|part| part == word)
        {
            return Some(digit.to_string());
        }
    }
    None
}

pub(crate) fn count_route_memory_last_visible_answer_surface(
    assistant_text: &str,
) -> Option<String> {
    assistant_text
        .lines()
        .filter(|line| {
            let upper = line.to_ascii_uppercase();
            upper.contains("VISIBLE ANSWER")
                || upper.contains("WORKING ANSWER")
                || upper.contains("EXACT OUTPUT")
                || upper.contains("[REQUEST: LOCK]")
        })
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .last()
}

pub(crate) fn count_route_memory_protected_lock_surface_needed(
    assistant_text: &str,
    expected_answer: &str,
) -> bool {
    let Some(last_surface) = count_route_memory_last_visible_answer_surface(assistant_text) else {
        return false;
    };
    if !lock_line_complete_for_stream_stop(assistant_text) {
        return false;
    }
    if !last_surface
        .to_ascii_uppercase()
        .contains("[REQUEST: LOCK]")
    {
        return false;
    }
    count_route_memory_answer_signature(&last_surface).as_deref() != Some(expected_answer)
}

pub(crate) fn count_route_memory_enumerated_word_count(
    assistant_text: &str,
    candidate: &CountRouteMemoryFinalizationCandidate,
) -> Option<usize> {
    let target_word: Vec<char> = candidate.word.chars().collect();
    if target_word.len() < 2 {
        return None;
    }

    for line in assistant_text.lines() {
        let chars: Vec<char> = line.chars().collect();
        for start in 0..chars.len() {
            if !chars[start].is_ascii_alphabetic() {
                continue;
            }
            let mut idx = start;
            let mut matched = 0usize;
            let mut separators_between_letters = 0usize;
            let mut target_count = 0usize;
            while matched < target_word.len() {
                while idx < chars.len() && !chars[idx].is_ascii_alphabetic() {
                    idx += 1;
                }
                if idx >= chars.len() {
                    break;
                }
                if chars[idx].to_ascii_lowercase() != target_word[matched] {
                    break;
                }
                if chars[idx].to_ascii_lowercase() == candidate.target_letter {
                    target_count += 1;
                }
                matched += 1;
                idx += 1;
                if matched < target_word.len() {
                    let mut saw_separator = false;
                    while idx < chars.len() && !chars[idx].is_ascii_alphabetic() {
                        saw_separator = true;
                        idx += 1;
                    }
                    if saw_separator {
                        separators_between_letters += 1;
                    }
                }
            }
            if matched == target_word.len()
                && separators_between_letters >= target_word.len().saturating_sub(1)
            {
                let trailing_alpha = idx < chars.len() && chars[idx].is_ascii_alphabetic();
                let leading_alpha = start > 0 && chars[start - 1].is_ascii_alphabetic();
                if !leading_alpha && !trailing_alpha {
                    return Some(target_count);
                }
            }
        }
    }
    None
}

pub(crate) fn count_route_memory_numbered_prefix_count(
    assistant_text: &str,
    candidate: &CountRouteMemoryFinalizationCandidate,
) -> Option<usize> {
    let target_word: Vec<char> = candidate.word.chars().collect();
    if target_word.len() < 2 {
        return None;
    }
    let last_target_idx = target_word
        .iter()
        .rposition(|ch| *ch == candidate.target_letter)?;

    let mut prefix_len = 0usize;
    let mut target_evidence_count = 0usize;
    let mut started = false;
    for line in assistant_text.lines() {
        let trimmed = line.trim_start();
        let mut chars = trimmed.char_indices().peekable();
        let mut row_digits = String::new();
        while let Some((_, ch)) = chars.peek().copied() {
            if ch.is_ascii_digit() {
                row_digits.push(ch);
                chars.next();
            } else {
                break;
            }
        }
        if row_digits.is_empty() {
            continue;
        }
        let Ok(row_index) = row_digits.parse::<usize>() else {
            continue;
        };
        if row_index != prefix_len + 1 {
            if started && row_index > prefix_len + 1 {
                break;
            }
            continue;
        }

        let Some((letter_pos, row_letter)) = chars
            .by_ref()
            .find(|(_, ch)| ch.is_ascii_alphabetic())
            .map(|(idx, ch)| (idx, ch.to_ascii_lowercase()))
        else {
            continue;
        };
        if target_word
            .get(prefix_len)
            .map_or(true, |expected| *expected != row_letter)
        {
            if started {
                break;
            }
            continue;
        }

        let rest = &trimmed[letter_pos + row_letter.len_utf8()..];
        let mut numbers_after_letter = Vec::new();
        let mut digits = String::new();
        for ch in rest.chars() {
            if ch.is_ascii_digit() {
                digits.push(ch);
            } else if !digits.is_empty() {
                if let Ok(value) = digits.parse::<usize>() {
                    numbers_after_letter.push(value);
                }
                digits.clear();
            }
        }
        if !digits.is_empty() {
            if let Ok(value) = digits.parse::<usize>() {
                numbers_after_letter.push(value);
            }
        }
        let Some(row_count) = numbers_after_letter.first().copied() else {
            continue;
        };
        let expected_row_count = usize::from(row_letter == candidate.target_letter);
        if row_count != expected_row_count {
            if started {
                break;
            }
            continue;
        }

        started = true;
        prefix_len += 1;
        target_evidence_count += expected_row_count;
    }

    let covered_all_target_positions = prefix_len > last_target_idx;
    let saw_non_target_after_last_target =
        prefix_len > last_target_idx + 1 || prefix_len == target_word.len();
    if covered_all_target_positions
        && saw_non_target_after_last_target
        && target_evidence_count.to_string() == candidate.answer
    {
        Some(target_evidence_count)
    } else {
        None
    }
}

pub(crate) fn count_route_memory_enumeration_aggregation_candidate(
    enabled: bool,
    candidate: Option<&CountRouteMemoryFinalizationCandidate>,
    assistant_text: &str,
) -> Option<CountRouteMemoryFinalizationCandidate> {
    let candidate = candidate?.clone();
    if !enabled {
        return Some(candidate);
    }
    let Some(enumerated_count) =
        count_route_memory_enumerated_word_count(assistant_text, &candidate)
            .or_else(|| count_route_memory_numbered_prefix_count(assistant_text, &candidate))
    else {
        return Some(candidate);
    };
    Some(CountRouteMemoryFinalizationCandidate {
        answer: enumerated_count.to_string(),
        parser_confidence: 0.7,
        parser_version: "enumeration_aggregate_v1".to_string(),
        ..candidate
    })
}

pub(crate) fn count_route_memory_finalization_telemetry(
    enabled: bool,
    candidate: Option<&CountRouteMemoryFinalizationCandidate>,
    assistant_text: &str,
) -> CountRouteMemoryFinalizationTelemetry {
    if !enabled {
        return CountRouteMemoryFinalizationTelemetry::default();
    }

    let Some(candidate) = candidate else {
        return CountRouteMemoryFinalizationTelemetry {
            candidate_enabled: true,
            state: Some("parser_unmatched".to_string()),
            ..Default::default()
        };
    };

    let answer_signature_seen = count_route_memory_answer_window(assistant_text)
        .as_deref()
        .and_then(count_route_memory_answer_signature);
    let state = match answer_signature_seen.as_deref() {
        None => "pending",
        Some(seen) if seen == candidate.answer => "protected_correct",
        Some(_) => "eligible_same_run_failure",
    }
    .to_string();
    let do_no_harm_protected = state == "protected_correct";
    let would_apply = state == "eligible_same_run_failure";

    CountRouteMemoryFinalizationTelemetry {
        candidate_enabled: true,
        candidate_answer: Some(candidate.answer.clone()),
        candidate_word: Some(candidate.word.clone()),
        candidate_target_letter: Some(candidate.target_letter.to_string()),
        parser_confidence: Some(candidate.parser_confidence),
        parser_version: Some(candidate.parser_version.clone()),
        state: Some(state),
        answer_signature_seen,
        do_no_harm_protected,
        would_apply,
    }
}

pub(crate) fn count_route_memory_finalization_action(
    exact_action_enabled: bool,
    natural_v2_action_enabled: bool,
    enumeration_aggregation_action_enabled: bool,
    enumeration_preserve_stop_enabled: bool,
    protected_lock_surface_enabled: bool,
    candidate: Option<&CountRouteMemoryFinalizationCandidate>,
    telemetry: &CountRouteMemoryFinalizationTelemetry,
    assistant_text: &str,
) -> (CountRouteMemoryFinalizationAction, Option<String>) {
    if !exact_action_enabled
        && !natural_v2_action_enabled
        && !enumeration_aggregation_action_enabled
        && !enumeration_preserve_stop_enabled
        && !protected_lock_surface_enabled
    {
        return (CountRouteMemoryFinalizationAction::default(), None);
    }

    let mut action = CountRouteMemoryFinalizationAction {
        action_enabled: true,
        ..Default::default()
    };
    let Some(candidate) = candidate else {
        action.action_reason = Some("parser_unmatched".to_string());
        return (action, None);
    };
    let enumeration_preserve_allowed = candidate.parser_version == "enumeration_aggregate_v1"
        && enumeration_preserve_stop_enabled
        && (candidate.parser_confidence - 0.7).abs() <= 1e-6;
    if enumeration_preserve_allowed {
        if telemetry.state.as_deref() == Some("pending")
            && telemetry.answer_signature_seen.is_none()
        {
            let mut preserved = assistant_text.trim_end().to_string();
            if !preserved.is_empty() {
                preserved.push('\n');
            }
            preserved.push_str("VISIBLE ANSWER: ");
            preserved.push_str(&candidate.answer);
            action.action_applied = true;
            action.action_reason =
                Some("enumeration_evidence_preserved_before_answer_window".to_string());
            action.replacement_answer = Some(candidate.answer.clone());
            action.stop_reason =
                Some("count_route_memory_finalization_enumeration_preserve_stop".to_string());
            return (action, Some(preserved));
        }
        if !exact_action_enabled
            && !natural_v2_action_enabled
            && !enumeration_aggregation_action_enabled
        {
            action.action_reason = match telemetry.state.as_deref() {
                Some("protected_correct") => Some("protected_correct".to_string()),
                Some("eligible_same_run_failure") => {
                    Some("enumeration_preserve_not_pending".to_string())
                }
                Some(_) => Some("enumeration_preserve_not_ready".to_string()),
                None => Some("enumeration_preserve_not_ready".to_string()),
            };
            return (action, None);
        }
    }
    let parser_allowed = if candidate.parser_version == "arrow_v1_exact" {
        (exact_action_enabled || protected_lock_surface_enabled)
            && (candidate.parser_confidence - 1.0).abs() <= 1e-6
    } else if candidate.parser_version == "natural_count_v2_shadow" {
        (natural_v2_action_enabled || protected_lock_surface_enabled)
            && (candidate.parser_confidence - 0.75).abs() <= 1e-6
    } else if candidate.parser_version == "enumeration_aggregate_v1" {
        (enumeration_aggregation_action_enabled || protected_lock_surface_enabled)
            && (candidate.parser_confidence - 0.7).abs() <= 1e-6
    } else {
        false
    };
    if !parser_allowed {
        action.action_reason = if candidate.parser_version == "natural_count_v2_shadow"
            && !natural_v2_action_enabled
            && !protected_lock_surface_enabled
        {
            if exact_action_enabled {
                Some("parser_confidence_below_exact".to_string())
            } else {
                Some("natural_v2_action_not_enabled".to_string())
            }
        } else if candidate.parser_version == "arrow_v1_exact" && !exact_action_enabled {
            Some("exact_action_not_enabled".to_string())
        } else if candidate.parser_version == "arrow_v1_exact" {
            Some("parser_confidence_below_exact".to_string())
        } else if candidate.parser_version == "natural_count_v2_shadow" {
            Some("parser_confidence_below_natural_v2".to_string())
        } else if candidate.parser_version == "enumeration_aggregate_v1"
            && !enumeration_aggregation_action_enabled
        {
            Some("enumeration_aggregation_action_not_enabled".to_string())
        } else if candidate.parser_version == "enumeration_aggregate_v1" {
            Some("parser_confidence_below_enumeration_aggregate".to_string())
        } else {
            Some("parser_version_unsupported".to_string())
        };
        return (action, None);
    }
    if telemetry.do_no_harm_protected {
        if protected_lock_surface_enabled
            && count_route_memory_protected_lock_surface_needed(assistant_text, &candidate.answer)
        {
            let mut surfaced = assistant_text.trim_end().to_string();
            if !surfaced.is_empty() {
                surfaced.push('\n');
            }
            surfaced.push_str("VISIBLE ANSWER: ");
            surfaced.push_str(&candidate.answer);
            action.action_applied = true;
            action.action_reason = Some("protected_lock_surface_answer_exposed".to_string());
            action.replacement_answer = Some(candidate.answer.clone());
            action.stop_reason =
                Some("count_route_memory_finalization_protected_lock_surface".to_string());
            return (action, Some(surfaced));
        }
        action.action_reason = Some("protected_correct".to_string());
        return (action, None);
    }
    if protected_lock_surface_enabled
        && !exact_action_enabled
        && !natural_v2_action_enabled
        && !enumeration_aggregation_action_enabled
    {
        action.action_reason = Some("protected_lock_surface_not_protected".to_string());
        return (action, None);
    }
    if !telemetry.would_apply
        || telemetry.state.as_deref() != Some("eligible_same_run_failure")
        || telemetry.answer_signature_seen.is_none()
    {
        action.action_reason = Some("not_eligible_same_run_failure".to_string());
        return (action, None);
    }

    let Some((replacement_text, original_window)) =
        replace_count_route_memory_answer_window(assistant_text, &candidate.answer)
    else {
        action.action_reason = Some("answer_window_unavailable".to_string());
        return (action, None);
    };

    action.action_applied = true;
    action.action_reason = Some("eligible_same_run_failure_replaced".to_string());
    action.replacement_answer = Some(candidate.answer.clone());
    action.original_answer_window = Some(original_window);
    action.stop_reason = Some("count_route_memory_finalization_replacement".to_string());
    (action, Some(replacement_text))
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct CompactResumeAnchor {
    pub(crate) kind: String,
    pub(crate) name: String,
    pub(crate) value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct CompactResumeActiveContextShadowSteeringReadiness {
    pub(crate) surface_id: String,
    pub(crate) adapter_id: String,
    pub(crate) source: String,
    pub(crate) shadow_steering_ready: bool,
    pub(crate) selected_packet_ref_count: usize,
    pub(crate) route_steer_shadow_decision_count: usize,
    pub(crate) recommended_steer_count: usize,
    pub(crate) safety_gate_count: usize,
    pub(crate) failed_gate_count: usize,
    pub(crate) read_only: bool,
    pub(crate) prompt_text_injected: bool,
    pub(crate) final_answer_injected: bool,
    pub(crate) answer_scoring: bool,
    pub(crate) runtime_steering_applied: bool,
    pub(crate) reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct CompactResumeState {
    pub(crate) version: String,
    #[serde(default)]
    pub(crate) turn_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) task_frame: Option<String>,
    #[serde(default)]
    pub(crate) decision_critical_anchors: Vec<CompactResumeAnchor>,
    #[serde(default)]
    pub(crate) names: Vec<String>,
    #[serde(default)]
    pub(crate) constraints: Vec<String>,
    #[serde(default)]
    pub(crate) deadlines: Vec<String>,
    #[serde(default)]
    pub(crate) preference_flags: Vec<String>,
    #[serde(default)]
    pub(crate) unresolved_questions: Vec<String>,
    #[serde(default)]
    pub(crate) requested_output_shape: Vec<String>,
    #[serde(default)]
    pub(crate) prior_results: Vec<String>,
    #[serde(default)]
    pub(crate) corrections: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) active_context_shadow_steering_readiness:
        Option<CompactResumeActiveContextShadowSteeringReadiness>,
    /// DEEP_DIVE_ROADMAP P1-C task anchor persisted across compact resumes.
    /// Empty vec when the saving turn had no task anchor signature
    /// (i.e. the prompt didn't trip `structured_reasoning_signal >= 0.42`
    /// AND no prior turn injected one). On load, if non-empty, the engine's
    /// `current_task_anchor_signature` is overridden with this vector — that
    /// way hinge similarity, drift, and routing scores compare against the
    /// persisted anchor rather than re-hashing the new turn's prompt text
    /// (which "shears" the task payload across hinge transitions in fresh
    /// processes per the deep dive — `2026-05-02_138-inertial-task-anchoring`).
    /// Skip serializing if empty so old files without this field still round-trip.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) task_anchor_vector: Vec<f32>,
}

impl CompactResumeState {
    pub(crate) fn new() -> Self {
        Self {
            version: "compact_resume_state_v1".to_string(),
            ..Self::default()
        }
    }

    pub(crate) fn has_anchors(&self) -> bool {
        self.task_frame.is_some()
            || !self.decision_critical_anchors.is_empty()
            || !self.names.is_empty()
            || !self.constraints.is_empty()
            || !self.deadlines.is_empty()
            || !self.preference_flags.is_empty()
            || !self.unresolved_questions.is_empty()
            || !self.requested_output_shape.is_empty()
            || !self.prior_results.is_empty()
            || !self.corrections.is_empty()
    }

    pub(crate) fn anchor_count(&self) -> usize {
        self.task_frame.iter().count()
            + self.decision_critical_anchors.len()
            + self.names.len()
            + self.constraints.len()
            + self.deadlines.len()
            + self.preference_flags.len()
            + self.unresolved_questions.len()
            + self.requested_output_shape.len()
            + self.prior_results.len()
            + self.corrections.len()
    }
}

pub(crate) fn compact_resume_active_context_shadow_steering_readiness(
    readiness: &ActiveContextShadowSteeringReadiness,
) -> CompactResumeActiveContextShadowSteeringReadiness {
    CompactResumeActiveContextShadowSteeringReadiness {
        surface_id: readiness.surface_id.to_string(),
        adapter_id: readiness.adapter_id.to_string(),
        source: readiness.source.clone(),
        shadow_steering_ready: readiness.shadow_steering_ready,
        selected_packet_ref_count: readiness.selected_packet_ref_count,
        route_steer_shadow_decision_count: readiness.route_steer_shadow_decision_count,
        recommended_steer_count: readiness.recommended_steer_count,
        safety_gate_count: readiness.safety_gate_count,
        failed_gate_count: readiness.failed_gate_count,
        read_only: readiness.read_only,
        prompt_text_injected: readiness.prompt_text_injected,
        final_answer_injected: readiness.final_answer_injected,
        answer_scoring: readiness.answer_scoring,
        runtime_steering_applied: readiness.runtime_steering_applied,
        reason_codes: readiness.reason_codes.clone(),
    }
}

pub(crate) fn compact_resume_active_context_shadow_steering_readiness_safe(
    readiness: &CompactResumeActiveContextShadowSteeringReadiness,
) -> bool {
    let expected_surface = readiness.surface_id == "active_context_shadow_steering_readiness_v1";
    let expected_adapter = readiness.adapter_id == ACTIVE_CONTEXT_ADAPTER_ID;
    let expected_source = readiness.source == "live_turn_start_metadata";
    let metadata_only = readiness.read_only
        && !readiness.prompt_text_injected
        && !readiness.final_answer_injected
        && !readiness.answer_scoring
        && !readiness.runtime_steering_applied;
    let count_shape = readiness.safety_gate_count > 0
        && readiness.failed_gate_count <= readiness.safety_gate_count
        && readiness.route_steer_shadow_decision_count >= readiness.recommended_steer_count;
    let readiness_consistent =
        readiness.shadow_steering_ready == (readiness.failed_gate_count == 0);
    let reason_shape = readiness
        .reason_codes
        .iter()
        .any(|code| code == "observe_only")
        && readiness
            .reason_codes
            .iter()
            .any(|code| code == "shadow_steering_readiness_metadata_only")
        && readiness
            .reason_codes
            .iter()
            .any(|code| code == "no_prompt_or_answer_payload")
        && readiness
            .reason_codes
            .iter()
            .any(|code| code == "no_runtime_steering_applied")
        && readiness
            .reason_codes
            .iter()
            .any(|code| code.starts_with("gate_pass:") || code.starts_with("gate_fail:"));

    expected_surface
        && expected_adapter
        && expected_source
        && metadata_only
        && count_shape
        && readiness_consistent
        && reason_shape
}

pub(crate) fn sanitize_compact_resume_active_context_shadow_steering_readiness(
    readiness: Option<CompactResumeActiveContextShadowSteeringReadiness>,
) -> Option<CompactResumeActiveContextShadowSteeringReadiness> {
    readiness.filter(compact_resume_active_context_shadow_steering_readiness_safe)
}

#[derive(Copy, Clone)]
pub(crate) struct ContinuityModeTuning {
    pub(crate) restore_steps_scale: f32,
    pub(crate) restore_strength_scale: f32,
    pub(crate) regression_steps_scale: f32,
    pub(crate) regression_strength_scale: f32,
    pub(crate) prune_window_scale: f32,
    pub(crate) prune_threshold_scale: f32,
    pub(crate) stable_release_scale: f32,
}

pub(crate) fn continuity_mode_tuning(mode: RuntimeMode) -> ContinuityModeTuning {
    match mode {
        RuntimeMode::Research => ContinuityModeTuning {
            restore_steps_scale: 1.20,
            restore_strength_scale: 1.10,
            regression_steps_scale: 1.20,
            regression_strength_scale: 1.10,
            prune_window_scale: 1.25,
            prune_threshold_scale: 1.15,
            stable_release_scale: 0.80,
        },
        RuntimeMode::Agency => ContinuityModeTuning {
            restore_steps_scale: 1.00,
            restore_strength_scale: 1.00,
            regression_steps_scale: 1.00,
            regression_strength_scale: 1.00,
            prune_window_scale: 1.00,
            prune_threshold_scale: 1.00,
            stable_release_scale: 1.00,
        },
        RuntimeMode::Clean => ContinuityModeTuning {
            restore_steps_scale: 0.80,
            restore_strength_scale: 0.85,
            regression_steps_scale: 0.85,
            regression_strength_scale: 0.90,
            prune_window_scale: 0.80,
            prune_threshold_scale: 0.90,
            stable_release_scale: 1.25,
        },
    }
}

#[derive(Copy, Clone)]
pub(crate) struct ContinuityScaleTuning {
    pub(crate) support_scale: f32,
    pub(crate) release_scale: f32,
}

pub(crate) fn continuity_scale_tuning(params_billions: f32) -> ContinuityScaleTuning {
    let safe_params = params_billions.max(1.0);
    let support_scale = clamp_f32((13.0 / safe_params).sqrt(), 0.70, 1.35);
    let release_scale = clamp_f32((safe_params / 13.0).sqrt(), 0.80, 1.30);
    ContinuityScaleTuning {
        support_scale,
        release_scale,
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum ModelArchetype {
    Auto,
    Standard,
    Thinking,
    Coding,
    Instruct,
    Chat,
}

impl ModelArchetype {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Standard => "standard",
            Self::Thinking => "thinking",
            Self::Coding => "coding",
            Self::Instruct => "instruct",
            Self::Chat => "chat",
        }
    }

    pub(crate) fn resolve(self, model_path: &str, model_size: &str) -> Self {
        if self != Self::Auto {
            return self;
        }
        let hay = format!(
            "{} {}",
            model_path.to_ascii_lowercase(),
            model_size.to_ascii_lowercase()
        );
        if hay.contains("coder") || hay.contains("code") {
            Self::Coding
        } else if hay.contains("think") || hay.contains("reason") || hay.contains("o1") {
            Self::Thinking
        } else if hay.contains("instruct") {
            Self::Instruct
        } else if hay.contains("chat") {
            Self::Chat
        } else {
            Self::Standard
        }
    }

    pub(crate) fn multiplier(self) -> f32 {
        match self {
            Self::Auto | Self::Standard => 1.0,
            Self::Thinking => 0.88,
            Self::Coding => 0.82,
            Self::Instruct => 1.0,
            Self::Chat => 1.04,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ModelScalingProfile {
    pub(crate) params_billions: f32,
    pub(crate) archetype: ModelArchetype,
    pub(crate) scale: f32,
    pub(crate) sigma: f32,
    pub(crate) theta: f32,
    pub(crate) beta: f32,
    pub(crate) loop_repulsion: f32,
    pub(crate) temperature: f32,
    pub(crate) motif_force_scale: f32,
    pub(crate) recovery_force_scale: f32,
    pub(crate) guardrail_bias_scale: f32,
    pub(crate) focus_lock_ticks: usize,
}

pub(crate) fn build_scaling_profile_value(
    args: &Args,
    scaling_profile: &Option<ModelScalingProfile>,
) -> serde_json::Value {
    let resolved_archetype = scaling_profile
        .as_ref()
        .map(|profile| profile.archetype)
        .unwrap_or_else(|| {
            args.model_archetype
                .resolve(&args.model_path, &args.model_size)
        });

    let profile_json = scaling_profile.as_ref().map(|profile| {
        serde_json::json!({
            "params_billions": profile.params_billions,
            "archetype": profile.archetype.as_str(),
            "scale": profile.scale,
            "sigma": profile.sigma,
            "theta": profile.theta,
            "beta": profile.beta,
            "loop_repulsion": profile.loop_repulsion,
            "temperature": profile.temperature,
            "motif_force_scale": profile.motif_force_scale,
            "recovery_force_scale": profile.recovery_force_scale,
            "guardrail_bias_scale": profile.guardrail_bias_scale,
            "focus_lock_ticks": profile.focus_lock_ticks,
        })
    });

    serde_json::json!({
        "model_path": args.model_path,
        "model_size": args.model_size,
        "model_auto_scale": args.model_auto_scale,
        "resolved_archetype": resolved_archetype.as_str(),
        "profile": profile_json,
        "applied": {
            "sigma": args.sigma,
            "physics_blend": args.physics_blend,
            "repulsion_strength": args.repulsion_strength,
            "temperature": args.temperature,
            "visible_request_gate": args.visible_request_gate,
        },
    })
}

pub(crate) fn calculate_model_scaling_profile(
    params_billions: f32,
    archetype: ModelArchetype,
    base_temperature: f32,
) -> ModelScalingProfile {
    // Golden Ratio profile: 8B instruct is the anchor and should land on the
    // successful "bend, don't break" regime by default.
    const GOLDEN_PARAMS: f32 = 8.0;
    const GOLDEN_SIGMA: f32 = 0.15;
    const GOLDEN_THETA: f32 = 0.55;
    const GOLDEN_BETA: f32 = 100.0;
    const GOLDEN_REPULSION: f32 = 0.60;
    const GOLDEN_TEMPERATURE: f32 = 0.7;
    const GOLDEN_MOTIF_FORCE: f32 = 0.35;
    const GOLDEN_RECOVERY_FORCE: f32 = 0.45;
    const GOLDEN_GUARDRAIL_BIAS: f32 = 1.5;
    const GOLDEN_FOCUS_LOCK_TICKS: usize = 30;

    let scale = (params_billions / GOLDEN_PARAMS).sqrt();
    let type_multiplier = archetype.multiplier();
    let force_scale = scale * type_multiplier;
    let sigma = clamp_f32(GOLDEN_SIGMA * force_scale, 0.04, 0.42);
    let theta = clamp_f32(GOLDEN_THETA * force_scale, 0.45, 1.8);
    let beta = clamp_f32(GOLDEN_BETA * scale, 70.0, 220.0);
    let loop_repulsion = clamp_f32(GOLDEN_REPULSION * force_scale, 0.35, 1.9);
    let base_temp = if base_temperature > 0.0 {
        base_temperature
    } else {
        GOLDEN_TEMPERATURE
    };
    let temperature = clamp_f32(base_temp * (GOLDEN_BETA / beta), 0.25, 1.2);
    let motif_force_scale = clamp_f32(GOLDEN_MOTIF_FORCE * force_scale, 0.18, 1.1);
    let recovery_force_scale = clamp_f32(GOLDEN_RECOVERY_FORCE * force_scale, 0.22, 1.25);
    let guardrail_bias_scale = clamp_f32(GOLDEN_GUARDRAIL_BIAS * force_scale.sqrt(), 1.0, 2.6);
    // Focus lock scales with model size: larger models get longer lock windows.
    let focus_lock_ticks = clamp_usize(
        (GOLDEN_FOCUS_LOCK_TICKS as f32 * force_scale) as usize,
        12,
        72,
    );

    ModelScalingProfile {
        params_billions,
        archetype,
        scale,
        sigma,
        theta,
        beta,
        loop_repulsion,
        temperature,
        motif_force_scale,
        recovery_force_scale,
        guardrail_bias_scale,
        focus_lock_ticks,
    }
}

pub(crate) fn apply_model_auto_scaling(args: &mut Args) -> Option<ModelScalingProfile> {
    if !args.model_auto_scale {
        return None;
    }
    let params_billions = parse_params_billions(&args.model_size)?;
    let archetype = args
        .model_archetype
        .resolve(&args.model_path, &args.model_size);
    let profile = calculate_model_scaling_profile(params_billions, archetype, args.temperature);

    // Allow overrides to take precedence when they differ from defaults
    if args.sigma_override != 0.15 {
        args.sigma = args.sigma_override;
    } else {
        args.sigma = profile.sigma as f64;
    }

    if args.theta_override != 2.0 {
        args.physics_blend = args.theta_override;
    } else {
        args.physics_blend = profile.theta;
    }

    args.repulsion_strength = -(profile.loop_repulsion as f64);
    args.temperature = profile.temperature;

    Some(profile)
}

#[derive(Debug, Clone)]
pub(crate) struct Attractor {
    #[allow(dead_code)]
    pub(crate) pos: Tensor,
    #[allow(dead_code)]
    pub(crate) strength: f32,
    #[allow(dead_code)]
    pub(crate) radius: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RuntimeBridgeManifest {
    pub(crate) bridge_version: String,
    pub(crate) canonical_runtime: RuntimeBridgeCanonicalRuntime,
    pub(crate) architecture_spec: RuntimeBridgeArchitectureSpec,
    pub(crate) runtime_hooks: RuntimeBridgeHooks,
    pub(crate) motifs: RuntimeBridgeMotifs,
    pub(crate) specialists: RuntimeBridgeSpecialists,
    pub(crate) warnings: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RuntimeBridgeCanonicalRuntime {
    pub(crate) project: String,
    pub(crate) runtime_root: String,
    pub(crate) status: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RuntimeBridgeArchitectureSpec {
    pub(crate) design_target: String,
    pub(crate) source_path: String,
    pub(crate) status: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RuntimeBridgeHooks {
    pub(crate) state_dynamics: String,
    pub(crate) pressure_signal: String,
    pub(crate) governor_head: String,
    pub(crate) power_gate: String,
    pub(crate) motif_bank_source: String,
    pub(crate) local_recovery_operator_source: String,
    pub(crate) minted_code_path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RuntimeBridgeMotifs {
    pub(crate) entry_count: usize,
    pub(crate) entries: Vec<RuntimeBridgeMotifEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RuntimeBridgeMotifEntry {
    pub(crate) motif_id: String,
    pub(crate) source: String,
    pub(crate) phase: String,
    #[serde(default)]
    pub(crate) injection_strength: f32,
    #[serde(default = "default_live_motif_member_count")]
    pub(crate) member_count: usize,
    pub(crate) persistence_score: f32,
    pub(crate) readiness_score: f32,
    #[serde(default)]
    pub(crate) promotion_score: f32,
    #[serde(default)]
    pub(crate) structured_signal: f32,
    #[serde(default)]
    pub(crate) tightness_score: f32,
    #[serde(default)]
    pub(crate) routing_safety_score: f32,
    #[serde(default = "default_motif_role_neutral")]
    pub(crate) motif_role: String,
    #[serde(default)]
    pub(crate) origin_run_id: String,
    #[serde(default)]
    pub(crate) parent_motif_ids: Vec<String>,
    #[serde(default)]
    pub(crate) provenance_summary: String,
    #[serde(default)]
    pub(crate) merge_key: String,
    #[serde(default)]
    pub(crate) task_anchor_signature: Option<Vec<f32>>,
    pub(crate) status: String,
    #[serde(default)]
    pub(crate) reflex_basin: RuntimeBridgeReflexBasin,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RuntimeBridgeSpecialists {
    pub(crate) count: usize,
    pub(crate) unresolved_count: usize,
    pub(crate) unresolved_sources: Vec<String>,
    #[serde(default)]
    pub(crate) entries: Vec<RuntimeBridgeSpecialistEntry>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub(crate) struct RuntimeBridgeReflexBasin {
    #[serde(default)]
    pub(crate) core_centroid: Vec<f32>,
    #[serde(default)]
    pub(crate) max_pre_energy: f32,
    #[serde(default)]
    pub(crate) flip_rate: f32,
    #[serde(default)]
    pub(crate) orbit_count: f32,
    #[serde(default)]
    pub(crate) radius_mean: f32,
    #[serde(default)]
    pub(crate) radius_std: f32,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub(crate) struct RuntimeBridgeSpecialistEntry {
    #[serde(default)]
    pub(crate) specialist_id: String,
    #[serde(default)]
    pub(crate) source: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) mode: String,
    #[serde(default)]
    pub(crate) motif_id: String,
    #[serde(default)]
    pub(crate) reflex_policy_role: String,
    #[serde(default)]
    pub(crate) centroid_coordinate: Vec<f32>,
    #[serde(default)]
    pub(crate) influence_radius: f32,
    #[serde(default)]
    pub(crate) basin_variance: f32,
    #[serde(default)]
    pub(crate) persistence_score: f32,
    #[serde(default)]
    pub(crate) runtime_readiness_score: f32,
    #[serde(default)]
    pub(crate) absence_signal: Option<f32>,
    #[serde(default)]
    pub(crate) tension_point: Option<f32>,
    #[serde(default)]
    pub(crate) betti_0: Option<f32>,
    #[serde(default)]
    pub(crate) betti_1: Option<f32>,
    #[serde(default)]
    pub(crate) ghost_diagnostics: RuntimeBridgeGhostDiagnostics,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub(crate) struct RuntimeBridgeGhostDiagnostics {
    #[serde(default)]
    pub(crate) flip_rate: f32,
    #[serde(default)]
    pub(crate) orbit_count: f32,
    #[serde(default)]
    pub(crate) max_pre_energy: f32,
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) radius_mean: f32,
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) radius_std: f32,
}

#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct RuntimeMotifField {
    pub(crate) motif_id: String,
    pub(crate) source: String,
    pub(crate) motif_kind: String,
    pub(crate) promotion_status: String,
    pub(crate) raw_signature: Vec<f32>,
    pub(crate) vector: Tensor,
    pub(crate) member_count: usize,
    pub(crate) last_updated_step: usize,
    pub(crate) persistence_score: f32,
    pub(crate) readiness_score: f32,
    pub(crate) injection_strength: f32,
    pub(crate) max_pre_energy: f32,
    pub(crate) flip_rate: f32,
    pub(crate) orbit_count: f32,
    pub(crate) radius_mean: f32,
    pub(crate) radius_std: f32,
    pub(crate) radius_m2: f32,
    pub(crate) promotion_score: f32,
    pub(crate) structured_signal: f32,
    pub(crate) tightness_score: f32,
    pub(crate) conflict_ratio: f32,
    pub(crate) mixed_ratio: f32,
    pub(crate) routing_safety_score: f32,
    pub(crate) topology_density: f32,
    pub(crate) sequential_gap_rate: f32,
    pub(crate) fragmentation: f32,
    pub(crate) hole_pressure: f32,
    pub(crate) tension_anchor_strength: f32,
    pub(crate) motif_role: String,
    pub(crate) controller_selected_count: usize,
    pub(crate) controller_rejected_count: usize,
    pub(crate) origin_run_id: String,
    pub(crate) promotion_epoch: usize,
    pub(crate) parent_motif_ids: Vec<String>,
    pub(crate) provenance_summary: String,
    pub(crate) merge_key: String,
    pub(crate) task_anchor_signature: Option<Vec<f32>>,
    pub(crate) live_hidden_remapped: bool,
}
