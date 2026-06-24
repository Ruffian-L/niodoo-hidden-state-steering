//! Telemetry/state struct definitions (Gate34, motif/hinge summaries, routing
//! decision cache, controller candidate records, hinge window/correlation
//! artifacts).
//! Extracted from main.rs as part of the comprehensive refactor
//! (pre-refactor-main-split-20260508 backup).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Gate34Phase {
    Inactive,
    Warmup,
    Latched,
    Released,
}

impl Gate34Phase {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Inactive => "inactive",
            Self::Warmup => "warmup",
            Self::Latched => "latched",
            Self::Released => "released",
        }
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub(crate) struct Gate34CandidateRecord {
    pub(crate) candidate_ghost_id: String,
    pub(crate) gate34_target_source: String,
    pub(crate) gate34_target_kind: String,
    pub(crate) count: u32,
    pub(crate) count_ratio: f32,
    pub(crate) mean_margin: f32,
    pub(crate) best_margin: f32,
    pub(crate) distance_to_probe_at_acquire: f32,
    pub(crate) prompt_ghost_cosine: f32,
    pub(crate) prompt_ghost_cosine_norm: f32,
    pub(crate) inverse_distance: f32,
    pub(crate) inverse_distance_norm: f32,
    pub(crate) mean_margin_norm: f32,
    pub(crate) best_margin_norm: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) motif_role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) routing_safety_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) injection_strength: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) persistence_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) readiness_score: Option<f32>,
    pub(crate) window_bias: f32,
    pub(crate) acquisition_score: f32,
    pub(crate) selected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct MotifHingeSummary {
    pub(crate) organic_promoted_observed: bool,
    pub(crate) recovered_promoted_observed: bool,
    pub(crate) hinge_flipped: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) organic_promoted_timing: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) recovered_promoted_timing: Option<String>,
    #[serde(default)]
    pub(crate) promotion_attempt_count: usize,
    #[serde(default)]
    pub(crate) promotion_failure_count: usize,
    #[serde(default)]
    pub(crate) structured_streak_peak: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct MotifRoutingSummary {
    #[serde(default)]
    pub(crate) controller_tick_count: usize,
    #[serde(default)]
    pub(crate) controller_selected_structured_count: usize,
    #[serde(default)]
    pub(crate) controller_selected_structured_candidate_count: usize,
    #[serde(default)]
    pub(crate) controller_selected_conversational_count: usize,
    #[serde(default)]
    pub(crate) conflict_tie_break_count: usize,
    #[serde(default)]
    pub(crate) structured_basin_lock_count: usize,
    #[serde(default)]
    pub(crate) neutral_basin_penalty_applied: usize,
    #[serde(default)]
    pub(crate) task_utility_bonus_applied: usize,
    #[serde(default)]
    pub(crate) structured_candidate_escalation_attempts: usize,
    #[serde(default)]
    pub(crate) structured_candidate_escalation_wins: usize,
    #[serde(default)]
    pub(crate) wrong_basin_lock_suspected: bool,
    #[serde(default)]
    pub(crate) routing_improved_vs_previous: bool,
    #[serde(default)]
    pub(crate) structured_candidate_loss_reason_counts: BTreeMap<String, usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) current_routed_motif_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) current_routed_motif_role: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub(crate) struct RoutingDecisionCache {
    pub(crate) motif_id: String,
    pub(crate) motif_role: String,
    pub(crate) routing_score: f32,
    pub(crate) expires_at_step: usize,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ControllerCandidateRecord {
    pub(crate) motif_id: String,
    pub(crate) motif_role: String,
    pub(crate) promotion_status: String,
    pub(crate) distance: f32,
    pub(crate) routing_score: f32,
    pub(crate) task_anchor_similarity: f32,
    pub(crate) topology_density: f32,
    pub(crate) sequential_gap_rate: f32,
    pub(crate) tension_anchor_strength: f32,
    pub(crate) tightness_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct TaskAnchorSummary {
    #[serde(default)]
    pub(crate) present: bool,
    #[serde(default)]
    pub(crate) similarity_start: f32,
    #[serde(default)]
    pub(crate) similarity_hinge: f32,
    #[serde(default)]
    pub(crate) similarity_24tok: f32,
    #[serde(default)]
    pub(crate) drift: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct HingeWindowCandidateSummary {
    pub(crate) motif_id: String,
    pub(crate) motif_role: String,
    pub(crate) promotion_status: String,
    pub(crate) distance: f32,
    pub(crate) routing_score: f32,
    pub(crate) task_anchor_similarity: f32,
    pub(crate) topology_density: f32,
    pub(crate) sequential_gap_rate: f32,
    pub(crate) tension_anchor_strength: f32,
    pub(crate) tightness_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct HingeWindowTickRecord {
    pub(crate) step: usize,
    pub(crate) event: String,
    pub(crate) structured_streak: usize,
    pub(crate) clamp_active: bool,
    pub(crate) clamp_strength: f32,
    pub(crate) task_anchor_similarity: f32,
    pub(crate) basin_width: f32,
    pub(crate) curvature_tension: f32,
    pub(crate) neutral_basin_occupancy: f32,
    pub(crate) structured_candidate_separation: f32,
    pub(crate) task_vector_drift: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) routed_motif_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) routed_motif_role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) structured_candidate_loss_reason: Option<String>,
    #[serde(default)]
    pub(crate) candidates: Vec<HingeWindowCandidateSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct HingeWindowArtifact {
    pub(crate) version: String,
    pub(crate) runtime_mode: String,
    pub(crate) restored_run: bool,
    #[serde(default)]
    pub(crate) hinge_flipped: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) first_promotion_attempt_step: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) first_hinge_step: Option<usize>,
    #[serde(default)]
    pub(crate) task_anchor: TaskAnchorSummary,
    #[serde(default)]
    pub(crate) neutral_basin_occupancy: f32,
    #[serde(default)]
    pub(crate) structured_candidate_separation: f32,
    #[serde(default)]
    pub(crate) records: Vec<HingeWindowTickRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct HingeCorrelationSummary {
    #[serde(default)]
    pub(crate) task_detected: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) expected_answer: Option<String>,
    #[serde(default)]
    pub(crate) task_success: bool,
    #[serde(default)]
    pub(crate) task_near_miss: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) observed_answer_hint: Option<String>,
    #[serde(default)]
    pub(crate) hinge_task_success: bool,
    #[serde(default)]
    pub(crate) recovered_promoted_and_success: bool,
    #[serde(default)]
    pub(crate) organic_promoted_and_success: bool,
}
