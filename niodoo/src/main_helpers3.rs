//! Final main-level helpers (parse correction packet source-target,
//! parse_params_billions, build_run_id, emit_ui_event_value, NoopPhysicsEngine,
//! RuntimeRecoveryOperator/SpecialistMemoryWorkerPacket/RuntimeSpecialistMemoryWorker/LiveMotifProbeStats,
//! resolve_runtime_bridge_path).
//! Extracted from main.rs as part of the comprehensive refactor.

#![allow(unused_imports)]

use anyhow::{Context, Result};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::cli::*;
use crate::physics::naked_llama::PhysicsEngine;
use crate::*;

pub(crate) fn parse_correction_packet_prompt_source_target_map(s: &str) -> Vec<(String, String)> {
    s.split(',')
        .filter_map(|pair| {
            let pair = pair.trim();
            if pair.is_empty() {
                return None;
            }
            let (sub, target) = pair.rsplit_once(':')?;
            let sub = sub.trim().to_lowercase();
            let target = target.trim().to_lowercase();
            if sub.is_empty() || target.is_empty() {
                return None;
            }
            Some((sub, target))
        })
        .collect()
}

pub(crate) fn resolve_correction_packet_prompt_source_target_override(
    prompt: &str,
    map: &[(String, String)],
) -> Option<String> {
    if map.is_empty() {
        return None;
    }
    let prompt_lc = prompt.to_lowercase();
    map.iter()
        .find(|(sub, _)| prompt_lc.contains(sub.as_str()))
        .map(|(_, target)| target.clone())
}

pub(crate) fn parse_params_billions(model_size: &str) -> Option<f32> {
    let lower = model_size.trim().to_ascii_lowercase();
    let mut buf = String::new();
    let mut best = None;
    for ch in lower.chars() {
        if ch.is_ascii_digit() || ch == '.' {
            buf.push(ch);
        } else {
            if !buf.is_empty() {
                if let Ok(v) = buf.parse::<f32>() {
                    best = Some(v);
                }
                buf.clear();
            }
            if ch == 'b' {
                break;
            }
        }
    }
    if best.is_none() && !buf.is_empty() {
        best = buf.parse::<f32>().ok();
    }
    best.filter(|v| *v > 0.0)
}

pub(crate) fn build_run_id(args: &Args) -> String {
    let model_stem = Path::new(&args.model_path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("model");
    let sanitized = model_stem
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>();
    format!(
        "{}::{}::seed{}::n{}",
        args.runtime_mode.as_str(),
        sanitized,
        args.seed,
        args.n
    )
}

// 2. THE PRESSURE GATE
// Geometry, not token count, decides when the steering loop should push.
pub const NIODOO_PRESSURE_GATE_START: f32 = 0.5;
pub const NIODOO_PRESSURE_GATE_FULL: f32 = 10.0;
pub const NIODOO_WOBBLE_PRESSURE_THRESHOLD: f32 = 14.0;
pub const NIODOO_VISIBLE_REQUEST_GATE_FLOOR: f32 = 0.35;
pub const NIODOO_VISIBLE_REQUEST_GATE_RAMP_START: usize = 3;
pub const NIODOO_VISIBLE_REQUEST_GATE_RAMP_FULL: usize = 10;
pub const LIVE_MOTIF_MERGE_DISTANCE_THRESHOLD: f32 = 0.18;
pub const LIVE_MOTIF_SEED_FITNESS_FLOOR: f32 = 0.30;
pub const COMPACT_PROMOTED_MOTIF_TOP_K: usize = 3;
pub const LIVE_MOTIF_STALE_STEP_WINDOW: usize = 96;
pub const LIVE_MOTIF_PRUNE_MEMBER_MAX: usize = 1;
pub const LIVE_MOTIF_PRUNE_PROMOTION_MAX: f32 = 0.38;
pub const LIVE_MOTIF_PROMOTION_ATTEMPT_THRESHOLD: f32 = 0.52;
pub const LIVE_MOTIF_CRYSTAL_PROMOTION_THRESHOLD: f32 = 0.66;
pub const LIVE_MOTIF_STRUCTURED_SIGNAL_THRESHOLD: f32 = 0.50;
pub const STRUCTURED_REENTRY_PROMPT_THRESHOLD: f32 = 0.42;
pub const STRUCTURED_STREAK_SIGNAL_THRESHOLD: f32 = 0.42;
pub const STRUCTURED_RATCHET_MIN_STREAK: usize = 2;
pub const STRUCTURED_PROMOTION_OVERRIDE_STREAK: usize = 3;
pub const REENTRY_CLAMP_MIN_STEPS: usize = 12;
pub const REENTRY_CLAMP_MAX_STEPS: usize = 48;
pub const MOTIF_CARRY_FORWARD_ASSIST_RATIO_THRESHOLD: f32 = 0.60;
pub const MOTIF_CARRY_FORWARD_ASSIST_SIM_THRESHOLD: f32 = 0.86;
pub const MOTIF_RESTORE_BIAS_MIN_STEPS: usize = 32;
pub const MOTIF_RESTORE_BIAS_MAX_STEPS: usize = 128;
pub const RESTORED_PROMOTED_DECAY_STEP_WINDOW: usize = 72;
pub const RESTORED_PROMOTED_RECOVERY_MEMBER_FLOOR: usize = 3;
pub const MOTIF_REGRESSION_ASSIST_MIN_STEPS: usize = 48;
pub const MOTIF_REGRESSION_ASSIST_MAX_STEPS: usize = 160;
pub const ROUTING_CONTROLLER_INTERVAL: usize = 4;
pub const ROUTING_CONTROLLER_TOP_K: usize = 3;
pub const ROUTING_CACHE_LIFETIME: usize = 4;
pub const ROUTING_TIE_BREAK_MARGIN: f32 = 0.03;
pub const ROUTING_STRUCTURED_WIDE_PENALTY: f32 = 0.05;
pub const ROUTING_GAP_PENALTY_SCALE: f32 = 0.05;
pub const ROUTING_TIGHTNESS_BONUS_SCALE: f32 = 0.06;
pub const ROUTING_TASK_UTILITY_BONUS_SCALE: f32 = 0.08;
pub const ROUTING_NEUTRAL_BASIN_PENALTY_SCALE: f32 = 0.06;
pub const ROUTING_STRUCTURED_CANDIDATE_BONUS_SCALE: f32 = 0.06;
pub const ROUTING_STRUCTURED_CANDIDATE_ESCALATION_TIGHTNESS: f32 = 0.64;
pub const ROUTING_STRUCTURED_CANDIDATE_ESCALATION_SIGNAL: f32 = 0.42;
pub const ROUTING_STRUCTURED_CANDIDATE_ESCALATION_TOPOLOGY: f32 = 0.26;
pub const ROUTING_STRUCTURED_CANDIDATE_ESCALATION_TASK: f32 = 0.28;
pub const STRUCTURED_FRAGMENTATION_DISCOUNT: f32 = 0.40;
pub const RESTORED_TOPOLOGY_FLOOR_SIGNAL: f32 = 0.55;
pub const RESTORED_TOPOLOGY_FLOOR_TIGHTNESS: f32 = 0.35;
pub const ROUTING_STICKINESS_BONUS: f32 = 0.08;
pub const ROUTING_STICKINESS_TICKS: usize = 3;
pub const MOTIF_ROLE_STRUCTURED_TIGHTNESS: f32 = 0.72;
pub const MOTIF_ROLE_STRUCTURED_SIGNAL: f32 = 0.55;
pub const MOTIF_ROLE_STRUCTURED_CANDIDATE_TIGHTNESS: f32 = 0.60;
pub const MOTIF_ROLE_STRUCTURED_CANDIDATE_SIGNAL: f32 = 0.42;
pub const MOTIF_ROLE_CONVERSATIONAL_TIGHTNESS: f32 = 0.45;
pub const MOTIF_ROLE_STRUCTURED_CANDIDATE_TASK_SIM: f32 = 0.12;
pub const TASK_ANCHOR_DIM: usize = 64;
pub const TASK_ANCHOR_BIND_TOKENS: usize = 24;
pub const HINGE_WINDOW_MAX_RECORDS: usize = 24;
pub const HINGE_WINDOW_POST_HINGE_TOKENS: usize = 16;
pub const STRUCTURED_RESUME_LOCK_WINDOW: usize = 24;

// 3. THE BLACK HOLES
// Repel these specifically to prevent loops and zombie modes.
pub const BLACK_HOLE_TOKENS: &[&str] =
    &["swift", "very", "really", "basically", "assistant", "User"];

// =============================================================================
// PHASE 1: TELEMETRY (Self-Awareness Layer)
// Tracks per-token geometry, gains, and applied forces for introspection.
// =============================================================================

const UI_EVENT_PREFIX: &str = "[UI_EVENT]";

pub(crate) fn emit_ui_event_value(enabled: bool, event: &str, payload: serde_json::Value) {
    if !enabled {
        return;
    }
    let record = serde_json::json!({
        "event": event,
        "payload": payload,
    });
    if let Ok(serialized) = serde_json::to_string(&record) {
        println!("{UI_EVENT_PREFIX} {}", serialized);
    }
}

#[derive(Serialize)]
pub struct CognitiveTrace {
    pub prompt: String,
    pub tokens: Vec<TokenPhysics>,
    pub config: String,
}

pub(crate) struct NoopPhysicsEngine;

impl crate::physics::naked_llama::PhysicsEngine for NoopPhysicsEngine {
    fn apply_forces(
        &mut self,
        attn: &Tensor,
        _layer_idx: usize,
        _ghost_vector: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        Ok(attn.zeros_like()?)
    }
}

// =============================================================================
// PHASE 4: AUTONOMIC OVERRIDE (Cybernetic Steering Loop)
// The model can REQUEST physics changes by outputting special tags.
// In research/agency these surfaces are part of the live steering loop.
// Clean mode is the later surface-suppression / polished-output path.
// =============================================================================

#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct RuntimeRecoveryOperator {
    pub(crate) specialist_id: String,
    pub(crate) source: String,
    pub(crate) motif_id: String,
    pub(crate) role: String,
    pub(crate) raw_signature: Vec<f32>,
    pub(crate) vector: Tensor,
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

#[derive(Debug, Clone, Deserialize, Default)]
pub(crate) struct SpecialistMemoryWorkerPacket {
    #[serde(default)]
    pub(crate) worker_id: String,
    #[serde(default)]
    pub(crate) packet_id: String,
    #[serde(default)]
    pub(crate) prompt_id: String,
    #[serde(default)]
    pub(crate) unicode_escape: String,
    #[serde(default)]
    pub(crate) original_route_id: String,
    #[serde(default)]
    pub(crate) decoded_route_id: String,
    #[serde(default)]
    pub(crate) route_preserved: bool,
    #[serde(default)]
    pub(crate) topk_hit: bool,
    #[serde(default)]
    pub(crate) worker_score: f32,
    #[serde(default)]
    pub(crate) decoded_64d: Vec<f32>,
    #[serde(default)]
    pub(crate) hidden_64d: Vec<f32>,
}

#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct RuntimeSpecialistMemoryWorker {
    pub(crate) worker_id: String,
    pub(crate) packet_id: String,
    pub(crate) source_prompt_id: String,
    pub(crate) unicode_escape: String,
    pub(crate) original_route_id: String,
    pub(crate) decoded_route_id: String,
    pub(crate) route_preserved: bool,
    pub(crate) topk_hit: bool,
    pub(crate) worker_score: f32,
    pub(crate) raw_signature: Vec<f32>,
    pub(crate) vector: Tensor,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct LiveMotifProbeStats {
    pub(crate) live_motif_count: usize,
    pub(crate) nearest_distance: f32,
    pub(crate) nearest_radius: f32,
    pub(crate) trap_pressure: f32,
    pub(crate) fragmentation: f32,
}

pub(crate) fn resolve_runtime_bridge_path(input: &str) -> Option<PathBuf> {
    let normalized = input.trim().to_ascii_lowercase();
    if matches!(normalized.as_str(), "none" | "off" | "disabled") {
        return None;
    }

    let provided = PathBuf::from(input);
    if provided.exists() {
        return Some(provided);
    }

    let candidates = [
        PathBuf::from(input),
        PathBuf::from("..").join(input),
        PathBuf::from("../..").join(input),
        PathBuf::from("niodoo/memory/runtime_bridge/niodoo_runtime_bridge.json"),
        PathBuf::from("memory/runtime_bridge/niodoo_runtime_bridge.json"),
        PathBuf::from("../../niodoo/memory/runtime_bridge/niodoo_runtime_bridge.json"),
        PathBuf::from("../../memory/runtime_bridge/niodoo_runtime_bridge.json"),
    ];

    candidates.into_iter().find(|path| path.exists())
}

// =============================================================================
