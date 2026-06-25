use clap::ValueEnum;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TelemetryProfile {
    Full,
    Score,
    Minimal,
}

impl TelemetryProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Score => "score",
            Self::Minimal => "minimal",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ForceEngineStatus {
    Idle,
    Coasting,
    Active,
}

impl Default for ForceEngineStatus {
    fn default() -> Self {
        Self::Idle
    }
}

impl ForceEngineStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Coasting => "coasting",
            Self::Active => "active",
        }
    }
}

#[derive(Serialize, Clone, Default)]
pub struct TokenPhysics {
    pub token: String,
    pub step: usize,
    pub engine_status: ForceEngineStatus,
    pub forces_applied: bool,
    pub gravity_force: f32,
    pub ghost_pre_norm: f32,
    pub ghost_gain: f32,
    pub applied_ghost_force: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applied_ghost_vector: Option<Vec<f32>>,
    pub goal_force: f32,
    pub repulsion_force: f32,
    pub motif_force: f32,
    pub recovery_force: f32,
    pub total_force: f32,
    pub activation_gate: f32,
    pub empathy_spike: f32,
    pub live_motif_count: usize,
    pub bridge_motif_count: usize,
    pub organic_promoted_count: usize,
    pub recovered_promoted_count: usize,
    pub restored_compact_count: usize,
    pub nearest_live_motif_distance: f32,
    pub nearest_live_motif_radius: f32,
    pub bridge_force_selection: String,
    pub bridge_force_selected_count: usize,
    pub bridge_force_selected_ids: Vec<String>,
    pub bridge_force_selection_source: String,
    pub bridge_force_selected_score_max: Option<f32>,
    pub bridge_force_selected_role: Option<String>,
    pub bridge_force_second_score: Option<f32>,
    pub bridge_force_selected_margin: Option<f32>,
    pub bridge_force_role_filter: String,
    pub bridge_force_min_margin: f32,
    pub routed_motif_id: Option<String>,
    pub routed_motif_role: Option<String>,
    pub routed_motif_score: Option<f32>,
    pub route_surface_id: Option<String>,
    pub route_surface_source: Option<String>,
    pub route_surface_role: Option<String>,
    pub controller_candidate_count: Option<u32>,
    pub live_basin_pressure: f32,
    pub surface_heuristic_flag: bool,
    pub lock_detected: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lock_detected_step: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lock_text: Option<String>,
    pub lock_stop_policy: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lock_taper_remaining: Option<usize>,
    pub lock_stop_triggered: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lock_stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_after_lock: Option<usize>,
    pub mistake_memory_matched: bool,
    pub mistake_memory_match_count: usize,
    pub mistake_memory_event_ids: Vec<String>,
    pub mistake_rejected_answer_seen: bool,
    pub mistake_accepted_answer_seen: bool,
    pub mistake_accepted_boundary_seen: bool,
    pub mistake_guard_blocked_lock: bool,
    pub mistake_guard_blocked_count: usize,
    pub mistake_reflex_matched: bool,
    pub mistake_reflex_match_count: usize,
    pub mistake_reflex_event_ids: Vec<String>,
    pub mistake_reflex_domains: Vec<String>,
    pub mistake_reflex_action_level: u8,
    pub mistake_reflex_resolution_level: u8,
    pub mistake_reflex_vector_slice_available: bool,
    pub mistake_reflex_unicode_packet_ids: Vec<String>,
    pub mistake_reflex_route_preserved: Option<bool>,
    pub mistake_reflex_unfold_reason: Option<String>,
    pub mistake_reflex_decay_reason: Option<String>,
    pub mistake_reflex_evidence_seen: bool,
    pub mistake_reflex_accepted_answer_candidate_seen: bool,
    pub mistake_reflex_old_mistake_seen: bool,
    pub mistake_reflex_old_path_after_earned: bool,
    pub mistake_reflex_earned_answer_seen: bool,
    pub mistake_reflex_earned_answer_text: Option<String>,
    pub mistake_reflex_earned_boundary_step: Option<usize>,
    pub mistake_reflex_earned_boundary_byte_len: Option<usize>,
    pub mistake_reflex_lock_blocked: bool,
    pub mistake_reflex_blocked_count: usize,
    pub mistake_reflex_retry_triggered: bool,
    pub mistake_reflex_retry_count: usize,
    pub mistake_reflex_retry_reason: Option<String>,
    pub mistake_reflex_retry_tokens_remaining: usize,
    pub mistake_reflex_prompt_applied: bool,
    pub mistake_reflex_prompt_injection_timing: Option<String>,
    pub mistake_reflex_prompt_injection_repeated: bool,
    pub mistake_reflex_prompt_hint_text: Option<String>,
    pub bridge_enabled: bool,
    pub req_id: String,
    pub prompt_hash: String,
    pub ghost_basins_loaded: usize,
    pub nearest_ghost_id: Option<String>,
    pub nearest_ghost_distance: f32,
    pub second_nearest_ghost_distance: f32,
    pub route_margin: f32,
    pub projection_strategy: String,
    pub ghost_pull_delta_norm: f32,
    pub intervention_applied: bool,
    pub gate34_target_source: Option<String>,
    pub gate34_target_kind: Option<String>,
    pub gate34_phase: Option<String>,
    pub gate34_target_ghost_id: Option<String>,
    pub gate34_target_specialist_id: Option<String>,
    pub gate34_target_motif_id: Option<String>,
    pub gate34_target_acquired_step: Option<i64>,
    pub gate34_target_margin_at_acquire: Option<f32>,
    pub gate34_target_distance_at_acquire: Option<f32>,
    pub gate34_current_target_distance: Option<f32>,
    pub gate34_warmup_distance_min: Option<f32>,
    pub gate34_warmup_distance_mean: Option<f32>,
    pub gate34_warmup_distance_max: Option<f32>,
    pub gate34_warmup_distance_std: Option<f32>,
    pub gate34_distance_drift_score: Option<f32>,
    pub gate34_distance_limit_ratio: Option<f32>,
    pub gate34_distance_limit_warmup: Option<f32>,
    pub gate34_distance_gate_mode: Option<String>,
    pub gate34_target_hold_remaining: Option<u32>,
    pub gate34_release_reason: Option<String>,
    pub gate34_latched_steps: Option<u32>,
    pub gate34_intervention_count: Option<u32>,
    pub active_recovery_specialist_id: Option<String>,
    pub active_recovery_weight: Option<f32>,
    pub specialist_run_length: Option<u32>,
    pub specialist_worker_enabled: bool,
    pub specialist_worker_mode: String,
    pub specialist_worker_selected_id: Option<String>,
    pub specialist_worker_packet_id: Option<String>,
    pub specialist_worker_unicode_escape: Option<String>,
    pub specialist_worker_original_route_id: Option<String>,
    pub specialist_worker_decoded_route_id: Option<String>,
    pub specialist_worker_route_preserved: Option<bool>,
    pub specialist_worker_topk_hit: Option<bool>,
    pub specialist_worker_score: Option<f32>,
    pub specialist_worker_source_prompt_id: Option<String>,
    pub specialist_worker_direction_source: Option<String>,
    pub specialist_worker_delta_norm_64d: Option<f32>,
    pub specialist_worker_hidden_delta_norm: Option<f32>,
    pub specialist_worker_influence_clamp: Option<f32>,
    pub specialist_worker_influence_scale: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub specialist_worker_probe_signature_64d: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub specialist_worker_target_signature_64d: Option<Vec<f32>>,
    /// Shadow telemetry for the count route-memory finalization lane. This
    /// records an explicit parser-derived answer candidate and same-run
    /// do-no-harm state. Replacement-action fields stay false/empty unless
    /// the default-off count finalization action is explicitly enabled.
    pub count_route_memory_finalization_candidate_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count_route_memory_finalization_candidate_answer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count_route_memory_finalization_candidate_word: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count_route_memory_finalization_candidate_target_letter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count_route_memory_finalization_candidate_parser_confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count_route_memory_finalization_candidate_parser_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count_route_memory_finalization_candidate_state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count_route_memory_finalization_answer_signature_seen: Option<String>,
    pub count_route_memory_finalization_do_no_harm_protected: bool,
    pub count_route_memory_finalization_would_apply: bool,
    pub count_route_memory_finalization_action_enabled: bool,
    pub count_route_memory_finalization_action_applied: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count_route_memory_finalization_action_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count_route_memory_finalization_replacement_answer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count_route_memory_finalization_original_answer_window: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count_route_memory_finalization_stop_reason: Option<String>,
    pub prompt_embedding_source: Option<String>,
    pub prompt_vec_norm: Option<f32>,
    pub prompt_similarity_unavailable: Option<bool>,
    /// Runtime TDA shadow monitor. Disabled by default; when enabled, the
    /// monitor computes H0/H1 persistence over a rolling telemetry window and
    /// records the current shadow gate decision without changing behavior
    /// unless --tda-shadow-breath-apply is also set.
    pub tda_shadow_enabled: bool,
    pub tda_shadow_breath_apply_enabled: bool,
    pub tda_shadow_window_size: usize,
    pub tda_shadow_stride: usize,
    pub tda_shadow_window_ready: bool,
    pub tda_shadow_decision_fresh: bool,
    pub tda_shadow_action: String,
    pub tda_shadow_reason: String,
    pub tda_shadow_breath_requested: bool,
    pub tda_shadow_loop_pressure: f32,
    pub tda_shadow_route_fragmentation: f32,
    pub tda_shadow_margin_collapse: f32,
    pub tda_shadow_force_overfire: f32,
    pub tda_shadow_route_churn: f32,
    pub tda_shadow_tag_density: f32,
    pub tda_shadow_repetition_pressure: f32,
    pub tda_shadow_breath_score: f32,
    pub tda_shadow_h0_bars: usize,
    pub tda_shadow_h0_finite_bars: usize,
    pub tda_shadow_h0_infinite_bars: usize,
    pub tda_shadow_h0_total_persistence: f32,
    pub tda_shadow_h0_max_persistence: f32,
    pub tda_shadow_h1_bars: usize,
    pub tda_shadow_h1_finite_bars: usize,
    pub tda_shadow_h1_infinite_bars: usize,
    pub tda_shadow_h1_total_persistence: f32,
    pub tda_shadow_h1_max_persistence: f32,
    pub tda_shadow_involution_residual_max: f32,
    pub tda_shadow_involution_residual_mean: f32,
    pub tda_shadow_involution_valid: bool,
    // VQ codec + phase2 specialist fields (niodv4_bridge integration)
    pub vq_code_assigned: Option<u8>,
    pub vq_encode_error: f32,
    pub correction_delta_norm: f32,
    pub specialist_activated: bool,
    /// True when --specialist-correction-apply is on AND the codec-mediated specialist
    /// force was actually added to probe_force this step. Distinct from `specialist_activated`,
    /// which only signals that the rule fired in 2D-coord space (observational).
    pub specialist_force_applied: bool,
    /// L2 norm in 4096D hidden space of the codec-mediated specialist force actually added
    /// to probe_force. Zero when --specialist-correction-apply is off or the force was clamped
    /// to nothing. NOT in 2D coord space — this is the real force magnitude.
    pub specialist_force_norm: f32,
    /// VQ bucket code used for correction-packet lookup on this step (packet path).
    /// Distinct from `vq_code_assigned` which is the rule-based specialist-side assignment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correction_packet_vq_code: Option<u8>,
    /// Number of vq-keyed correction packets that fired this step.
    pub correction_packet_fire_count: usize,
    /// RC5: of the fires this step, how many were packets minted THIS session and
    /// inserted into the live store (vs loaded from disk at startup). A nonzero value
    /// on a later turn is the proof that the mint->insert->fire loop closed in-process.
    pub correction_packet_live_minted_fired_count: usize,
    /// RC1: mean per-packet effectiveness EMA over the packets that fired this step
    /// (measured target-distance reduction; >0 = corrections are helping). 0.0 when
    /// no fired packet has been measured yet or outcome feedback is off.
    pub correction_packet_effectiveness_avg: f32,
    /// RC6: the correction-packet decay rate actually used this step after
    /// per-trajectory class routing (equals the global rate when routing is off).
    pub correction_packet_effective_decay_rate: f32,
    /// RC3: sliding-window mean fire count used for mid-turn re-classification.
    pub trajectory_window_mean: f32,
    /// RC3: cumulative mid-turn trajectory-label flips this turn (0 in legacy one-shot).
    pub trajectory_reclassify_count: u32,
    /// RC2: number of fires this step whose force used a within-block residual shape
    /// (high-resolution projection) instead of the flat 64-block smear.
    pub correction_packet_residual_applied: usize,
    /// Sum of L2 norms (4096D) of all correction-packet forces added this step.
    pub correction_packet_force_norm: f32,
    /// Packet IDs that fired this step. Empty when no packets fired.
    pub correction_packet_ids: Vec<String>,
    /// Runtime packet authority score for the strongest packet candidate this
    /// step. Shadow mode records this without changing legacy packet force.
    pub packet_authority_score: f32,
    /// True when the strongest packet candidate has enough source/route/vector
    /// evidence to be trusted by the authority gate.
    pub packet_authority_allowed: bool,
    /// Positive evidence that contributed to the authority score.
    pub packet_authority_reason: String,
    /// Block reason when authority is weak or unknown; "none" when allowed.
    pub packet_authority_blocked_reason: String,
    /// Packet steering arbitration result for this step. `disabled` when the
    /// explicit arbitration flag is off; otherwise one of no_packet,
    /// packet_shadow, or packet_force.
    pub correction_packet_arbitration_mode: String,
    /// Short reason for the arbitration result.
    pub correction_packet_arbitration_reason: String,
    /// Candidate packet count considered before arbitration. In packet_shadow,
    /// this can be nonzero while `correction_packet_fire_count` remains zero.
    pub correction_packet_arbitration_candidate_count: usize,
    /// Nearest target distance among candidate packets before arbitration.
    pub correction_packet_arbitration_min_target_distance: f32,
    /// Estimated candidate force norm before arbitration.
    pub correction_packet_arbitration_force_norm_estimate: f32,
    /// §10ck prompt-level packet top-K override for the current turn. `None`
    /// means no prompt-map entry matched and the configured
    /// `--correction-packet-fire-top-k` is in use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correction_packet_prompt_top_k_override: Option<usize>,
    /// §10cm bullet-10 observability: the literal substring from the §10ck
    /// top-K map that matched the prompt. Mirrors
    /// `correction_packet_prompt_top_k_override` — `None` when no rule fired.
    /// Lets a human verify *which* routing rule actually triggered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correction_packet_prompt_top_k_match_substring: Option<String>,
    /// §10cn task-neutrality gate: true when a non-empty prompt top-K map is
    /// configured, no rule matched the current prompt, and
    /// `--correction-packet-suppress-when-no-prompt-match` suppresses packet
    /// firing for this turn.
    pub correction_packet_suppress_for_current_prompt: bool,
    /// Prompt-level source target filter for the current turn. `None` means
    /// no prompt-map entry matched and packet firing is not source-target
    /// restricted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correction_packet_prompt_source_target_override: Option<String>,
    /// §10ck effective packet firing top-K after applying the prompt map.
    /// `0` retains the legacy "do not truncate by K" behavior.
    pub correction_packet_effective_fire_top_k: usize,
    /// Average post-decay (and post-unfold) effective pull strength across all packets
    /// that fired this step. Drops as packets accumulate fire_count when
    /// `--correction-packet-decay-rate < 1.0`. Zero when no packets fired or decay is
    /// disabled (constant pull == raw pull_strength).
    pub correction_packet_effective_pull_avg: f32,
    /// True on a step when the relapse trigger fired and decayed packets were
    /// re-strengthened by `--correction-packet-unfold-factor`. When false, raw decayed
    /// pull is used.
    pub correction_packet_unfold_active: bool,
    /// Quantization error of the current step's probe against its assigned codebook
    /// centroid. The relapse trigger compares this to
    /// `--correction-packet-unfold-encode-error-threshold`.
    pub correction_packet_vq_encode_error: f32,
    /// Multiplicative factor applied to firing packets' effective pull this step.
    /// `1.0` when unfold inactive; `--correction-packet-unfold-factor` when active.
    pub correction_packet_unfold_factor_applied: f32,
    /// Multiplicative factor applied to firing packets' effective pull from the
    /// competence-aware suppression mechanism (§10aw). `1.0` when previous-step
    /// entropy was at-or-above threshold (model uncertain) or when threshold is
    /// disabled. `< 1.0` (down to `--correction-packet-competence-suppress-factor`)
    /// when previous-step entropy was below threshold (model confident → suppress
    /// pull to preserve earned trajectory).
    pub correction_packet_competence_factor: f32,
    /// Previous step's normalized sampling entropy in [0, 1] used by the
    /// competence-aware suppression decision this step. 0.0 = perfectly confident,
    /// 1.0 = uniform distribution. Below `correction-packet-competence-entropy-
    /// threshold`, suppression engages.
    pub correction_packet_competence_entropy: f32,
    /// Minimum 64D distance from probe to any matched packet's target_z this
    /// step. Used by the distance-gated competence trigger (§10ay). When below
    /// `correction-packet-competence-distance-threshold`, suppression engages.
    /// `f32::INFINITY` (serialized large) means no firings or distance gate
    /// disabled. Lower values indicate probe is closer to known-correct geometry.
    pub correction_packet_min_target_distance: f32,
    /// §10bd v11.1 trajectory routing classification for the current turn.
    /// `None` = routing disabled OR pre-classify_step. `Some("competent")` =
    /// classifier ran and overrode mode to AND. `Some("drifting")` = classifier
    /// ran and kept user-configured mode. North Star bullet 10 (human inspect):
    /// makes the runtime's per-turn routing decision visible in telemetry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trajectory_classified: Option<String>,
    /// §10bf prompt-level codec activation flag for the current turn. `None` =
    /// gate not configured (legacy / always active). `Some(true)` = gate
    /// configured and prompt matched. `Some(false)` = gate configured and
    /// prompt did not match (codec/bridge force suppressed for this turn).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codec_active_for_current_prompt: Option<bool>,
    /// §10bg v11.5 internal classifier signal. The running mean of
    /// per-step total firing density that the v11.5 classifier accumulated
    /// up to this telemetry record. NaN before any sample. Codex AGI loop
    /// (see §10be.1) flagged that the emitted `trajectory_classified`
    /// labels did not match the observable first-10 fire_count distribution
    /// — exposing the internal mean here closes the feedback loop so the
    /// classifier can be tuned to the signal it actually sees, not the
    /// per-token telemetry signal that's a different aggregation. Default
    /// 0.0 = classifier disabled or no samples yet.
    pub trajectory_fire_count_running_mean: f32,
    /// §10bg v11.5 number of completed-step samples the classifier has
    /// accumulated within the current turn. Lets the consumer distinguish
    /// "classifier hasn't seen enough samples" from "samples were low."
    pub trajectory_fire_count_samples: u32,
}

impl TokenPhysics {
    pub fn to_profile_value(&self, profile: TelemetryProfile) -> serde_json::Value {
        match profile {
            TelemetryProfile::Full => serde_json::to_value(self).unwrap_or_else(|_| {
                serde_json::json!({
                    "record_type": "token",
                    "step": self.step,
                    "token": &self.token,
                    "serialization_error": true,
                })
            }),
            TelemetryProfile::Score => self.score_profile_value(),
            TelemetryProfile::Minimal => serde_json::json!({
                "record_type": "token",
                "step": self.step,
                "token": &self.token,
                "lock_detected": self.lock_detected,
                "lock_stop_triggered": self.lock_stop_triggered,
                "lock_stop_reason": &self.lock_stop_reason,
                "mistake_memory_matched": self.mistake_memory_matched,
                "mistake_memory_match_count": self.mistake_memory_match_count,
                "mistake_rejected_answer_seen": self.mistake_rejected_answer_seen,
                "mistake_accepted_answer_seen": self.mistake_accepted_answer_seen,
                "mistake_accepted_boundary_seen": self.mistake_accepted_boundary_seen,
                "mistake_guard_blocked_lock": self.mistake_guard_blocked_lock,
                "mistake_reflex_matched": self.mistake_reflex_matched,
                "mistake_reflex_evidence_seen": self.mistake_reflex_evidence_seen,
                "mistake_reflex_accepted_answer_candidate_seen": self.mistake_reflex_accepted_answer_candidate_seen,
                "mistake_reflex_old_path_after_earned": self.mistake_reflex_old_path_after_earned,
                "mistake_reflex_earned_answer_seen": self.mistake_reflex_earned_answer_seen,
                "mistake_reflex_earned_boundary_step": self.mistake_reflex_earned_boundary_step,
                "mistake_reflex_lock_blocked": self.mistake_reflex_lock_blocked,
                "mistake_reflex_retry_triggered": self.mistake_reflex_retry_triggered,
                "mistake_reflex_retry_count": self.mistake_reflex_retry_count,
                "mistake_reflex_retry_reason": &self.mistake_reflex_retry_reason,
                "mistake_reflex_retry_tokens_remaining": self.mistake_reflex_retry_tokens_remaining,
                "mistake_reflex_prompt_applied": self.mistake_reflex_prompt_applied,
                "mistake_reflex_prompt_injection_timing": &self.mistake_reflex_prompt_injection_timing,
                "mistake_reflex_prompt_injection_repeated": self.mistake_reflex_prompt_injection_repeated,
                "mistake_reflex_prompt_hint_text": &self.mistake_reflex_prompt_hint_text,
            }),
        }
    }

    fn score_profile_value(&self) -> serde_json::Value {
        let probe_signature_len = self
            .specialist_worker_probe_signature_64d
            .as_ref()
            .map(|values| values.len());
        let target_signature_len = self
            .specialist_worker_target_signature_64d
            .as_ref()
            .map(|values| values.len());
        let probe_signature_hash = self
            .specialist_worker_probe_signature_64d
            .as_ref()
            .map(|values| signature_hash(values));
        let target_signature_hash = self
            .specialist_worker_target_signature_64d
            .as_ref()
            .map(|values| signature_hash(values));
        let mut value = serde_json::json!({
            "record_type": "token",
            "token": &self.token,
            "step": self.step,
            "engine_status": self.engine_status.as_str(),
            "forces_applied": self.forces_applied,
            "motif_force": self.motif_force,
            "recovery_force": self.recovery_force,
            "total_force": self.total_force,
            "activation_gate": self.activation_gate,
            "live_motif_count": self.live_motif_count,
            "bridge_motif_count": self.bridge_motif_count,
            "organic_promoted_count": self.organic_promoted_count,
            "recovered_promoted_count": self.recovered_promoted_count,
            "restored_compact_count": self.restored_compact_count,
            "nearest_live_motif_distance": self.nearest_live_motif_distance,
            "nearest_live_motif_radius": self.nearest_live_motif_radius,
            "bridge_force_selection": &self.bridge_force_selection,
            "bridge_force_selected_count": self.bridge_force_selected_count,
            "bridge_force_selected_ids": &self.bridge_force_selected_ids,
            "bridge_force_selection_source": &self.bridge_force_selection_source,
            "bridge_force_selected_score_max": self.bridge_force_selected_score_max,
            "bridge_force_selected_role": &self.bridge_force_selected_role,
            "bridge_force_second_score": self.bridge_force_second_score,
            "bridge_force_selected_margin": self.bridge_force_selected_margin,
            "bridge_force_role_filter": &self.bridge_force_role_filter,
            "bridge_force_min_margin": self.bridge_force_min_margin,
            "routed_motif_id": &self.routed_motif_id,
            "routed_motif_role": &self.routed_motif_role,
            "routed_motif_score": self.routed_motif_score,
            "route_surface_id": &self.route_surface_id,
            "route_surface_source": &self.route_surface_source,
            "route_surface_role": &self.route_surface_role,
            "controller_candidate_count": self.controller_candidate_count,
            "lock_detected": self.lock_detected,
            "lock_detected_step": self.lock_detected_step,
            "lock_text": &self.lock_text,
            "lock_stop_policy": &self.lock_stop_policy,
            "lock_taper_remaining": self.lock_taper_remaining,
            "lock_stop_triggered": self.lock_stop_triggered,
            "lock_stop_reason": &self.lock_stop_reason,
            "tokens_after_lock": self.tokens_after_lock,
            "mistake_memory_matched": self.mistake_memory_matched,
            "mistake_memory_match_count": self.mistake_memory_match_count,
            "mistake_memory_event_ids": &self.mistake_memory_event_ids,
            "mistake_rejected_answer_seen": self.mistake_rejected_answer_seen,
            "mistake_accepted_answer_seen": self.mistake_accepted_answer_seen,
            "mistake_accepted_boundary_seen": self.mistake_accepted_boundary_seen,
            "mistake_guard_blocked_lock": self.mistake_guard_blocked_lock,
            "mistake_guard_blocked_count": self.mistake_guard_blocked_count,
            "mistake_reflex_matched": self.mistake_reflex_matched,
            "mistake_reflex_match_count": self.mistake_reflex_match_count,
            "mistake_reflex_event_ids": &self.mistake_reflex_event_ids,
            "mistake_reflex_domains": &self.mistake_reflex_domains,
            "mistake_reflex_action_level": self.mistake_reflex_action_level,
            "mistake_reflex_resolution_level": self.mistake_reflex_resolution_level,
            "mistake_reflex_vector_slice_available": self.mistake_reflex_vector_slice_available,
            "mistake_reflex_unicode_packet_ids": &self.mistake_reflex_unicode_packet_ids,
            "mistake_reflex_route_preserved": self.mistake_reflex_route_preserved,
            "mistake_reflex_unfold_reason": &self.mistake_reflex_unfold_reason,
            "mistake_reflex_decay_reason": &self.mistake_reflex_decay_reason,
            "mistake_reflex_evidence_seen": self.mistake_reflex_evidence_seen,
            "mistake_reflex_accepted_answer_candidate_seen": self.mistake_reflex_accepted_answer_candidate_seen,
            "mistake_reflex_old_mistake_seen": self.mistake_reflex_old_mistake_seen,
            "mistake_reflex_old_path_after_earned": self.mistake_reflex_old_path_after_earned,
            "mistake_reflex_earned_answer_seen": self.mistake_reflex_earned_answer_seen,
            "mistake_reflex_earned_answer_text": &self.mistake_reflex_earned_answer_text,
            "mistake_reflex_earned_boundary_step": self.mistake_reflex_earned_boundary_step,
            "mistake_reflex_earned_boundary_byte_len": self.mistake_reflex_earned_boundary_byte_len,
            "mistake_reflex_lock_blocked": self.mistake_reflex_lock_blocked,
            "mistake_reflex_blocked_count": self.mistake_reflex_blocked_count,
            "mistake_reflex_retry_triggered": self.mistake_reflex_retry_triggered,
            "mistake_reflex_retry_count": self.mistake_reflex_retry_count,
            "mistake_reflex_retry_reason": &self.mistake_reflex_retry_reason,
            "mistake_reflex_retry_tokens_remaining": self.mistake_reflex_retry_tokens_remaining,
            "mistake_reflex_prompt_applied": self.mistake_reflex_prompt_applied,
            "mistake_reflex_prompt_injection_timing": &self.mistake_reflex_prompt_injection_timing,
            "mistake_reflex_prompt_injection_repeated": self.mistake_reflex_prompt_injection_repeated,
            "mistake_reflex_prompt_hint_text": &self.mistake_reflex_prompt_hint_text,
            "bridge_enabled": self.bridge_enabled,
            "req_id": &self.req_id,
            "prompt_hash": &self.prompt_hash,
            // VQ/packet routing observability in Score telemetry (keeps smokes compact while
            // still supporting per-token bucket histograms and packet-fire attribution).
            "vq_code_assigned": self.vq_code_assigned,
            "vq_encode_error": self.vq_encode_error,
            "correction_packet_vq_code": self.correction_packet_vq_code,
            "correction_packet_fire_count": self.correction_packet_fire_count,
            "correction_packet_live_minted_fired_count": self.correction_packet_live_minted_fired_count,
            "correction_packet_effectiveness_avg": self.correction_packet_effectiveness_avg,
            "correction_packet_effective_decay_rate": self.correction_packet_effective_decay_rate,
            "trajectory_window_mean": self.trajectory_window_mean,
            "trajectory_reclassify_count": self.trajectory_reclassify_count,
            "correction_packet_residual_applied": self.correction_packet_residual_applied,
            "correction_packet_force_norm": self.correction_packet_force_norm,
            "correction_packet_ids": &self.correction_packet_ids,
        });

        let Some(obj) = value.as_object_mut() else {
            return value;
        };
        if let Some(route) = serde_json::json!({
            "ghost_basins_loaded": self.ghost_basins_loaded,
            "nearest_ghost_id": &self.nearest_ghost_id,
            "nearest_ghost_distance": self.nearest_ghost_distance,
            "second_nearest_ghost_distance": self.second_nearest_ghost_distance,
            "route_margin": self.route_margin,
            "projection_strategy": &self.projection_strategy,
            "ghost_pull_delta_norm": self.ghost_pull_delta_norm,
            "intervention_applied": self.intervention_applied,
            "gate34_target_source": &self.gate34_target_source,
            "gate34_target_kind": &self.gate34_target_kind,
            "gate34_phase": &self.gate34_phase,
            "gate34_target_ghost_id": &self.gate34_target_ghost_id,
            "gate34_target_specialist_id": &self.gate34_target_specialist_id,
            "gate34_target_motif_id": &self.gate34_target_motif_id,
            "gate34_target_acquired_step": self.gate34_target_acquired_step,
            "gate34_target_margin_at_acquire": self.gate34_target_margin_at_acquire,
            "gate34_target_distance_at_acquire": self.gate34_target_distance_at_acquire,
            "gate34_current_target_distance": self.gate34_current_target_distance,
            "gate34_release_reason": &self.gate34_release_reason,
            "gate34_intervention_count": self.gate34_intervention_count,
        })
        .as_object()
        {
            obj.extend(route.clone());
        }

        if self.tda_shadow_enabled {
            if let Some(tda) = serde_json::json!({
                "tda_shadow_enabled": self.tda_shadow_enabled,
                "tda_shadow_breath_apply_enabled": self.tda_shadow_breath_apply_enabled,
                "tda_shadow_window_size": self.tda_shadow_window_size,
                "tda_shadow_stride": self.tda_shadow_stride,
                "tda_shadow_window_ready": self.tda_shadow_window_ready,
                "tda_shadow_decision_fresh": self.tda_shadow_decision_fresh,
                "tda_shadow_action": &self.tda_shadow_action,
                "tda_shadow_reason": &self.tda_shadow_reason,
                "tda_shadow_breath_requested": self.tda_shadow_breath_requested,
                "tda_shadow_loop_pressure": self.tda_shadow_loop_pressure,
                "tda_shadow_route_fragmentation": self.tda_shadow_route_fragmentation,
                "tda_shadow_margin_collapse": self.tda_shadow_margin_collapse,
                "tda_shadow_force_overfire": self.tda_shadow_force_overfire,
                "tda_shadow_route_churn": self.tda_shadow_route_churn,
                "tda_shadow_tag_density": self.tda_shadow_tag_density,
                "tda_shadow_repetition_pressure": self.tda_shadow_repetition_pressure,
                "tda_shadow_breath_score": self.tda_shadow_breath_score,
                "tda_shadow_h0_bars": self.tda_shadow_h0_bars,
                "tda_shadow_h0_finite_bars": self.tda_shadow_h0_finite_bars,
                "tda_shadow_h0_infinite_bars": self.tda_shadow_h0_infinite_bars,
                "tda_shadow_h0_total_persistence": self.tda_shadow_h0_total_persistence,
                "tda_shadow_h0_max_persistence": self.tda_shadow_h0_max_persistence,
                "tda_shadow_h1_bars": self.tda_shadow_h1_bars,
                "tda_shadow_h1_finite_bars": self.tda_shadow_h1_finite_bars,
                "tda_shadow_h1_infinite_bars": self.tda_shadow_h1_infinite_bars,
                "tda_shadow_h1_total_persistence": self.tda_shadow_h1_total_persistence,
                "tda_shadow_h1_max_persistence": self.tda_shadow_h1_max_persistence,
                "tda_shadow_involution_residual_max": self.tda_shadow_involution_residual_max,
                "tda_shadow_involution_residual_mean": self.tda_shadow_involution_residual_mean,
                "tda_shadow_involution_valid": self.tda_shadow_involution_valid,
            })
            .as_object()
            {
                obj.extend(tda.clone());
            }
        }

        if let Some(recovery) = serde_json::json!({
            "active_recovery_specialist_id": &self.active_recovery_specialist_id,
            "active_recovery_weight": self.active_recovery_weight,
            "specialist_run_length": self.specialist_run_length,
            "specialist_worker_enabled": self.specialist_worker_enabled,
            "specialist_worker_mode": &self.specialist_worker_mode,
            "specialist_worker_selected_id": &self.specialist_worker_selected_id,
            "specialist_worker_packet_id": &self.specialist_worker_packet_id,
            "specialist_worker_unicode_escape": &self.specialist_worker_unicode_escape,
            "specialist_worker_original_route_id": &self.specialist_worker_original_route_id,
            "specialist_worker_decoded_route_id": &self.specialist_worker_decoded_route_id,
            "specialist_worker_route_preserved": self.specialist_worker_route_preserved,
            "specialist_worker_topk_hit": self.specialist_worker_topk_hit,
            "specialist_worker_score": self.specialist_worker_score,
            "specialist_worker_source_prompt_id": &self.specialist_worker_source_prompt_id,
            "specialist_worker_direction_source": &self.specialist_worker_direction_source,
            "specialist_worker_delta_norm_64d": self.specialist_worker_delta_norm_64d,
            "specialist_worker_hidden_delta_norm": self.specialist_worker_hidden_delta_norm,
            "specialist_worker_influence_clamp": self.specialist_worker_influence_clamp,
            "specialist_worker_influence_scale": self.specialist_worker_influence_scale,
            "specialist_worker_direction_auditable": self.specialist_worker_probe_signature_64d.is_some()
                && self.specialist_worker_target_signature_64d.is_some(),
            "specialist_worker_probe_signature_len_64d": probe_signature_len,
            "specialist_worker_target_signature_len_64d": target_signature_len,
            "specialist_worker_probe_signature_hash": probe_signature_hash,
            "specialist_worker_target_signature_hash": target_signature_hash,
            "count_route_memory_finalization_candidate_enabled": self.count_route_memory_finalization_candidate_enabled,
            "count_route_memory_finalization_candidate_answer": &self.count_route_memory_finalization_candidate_answer,
            "count_route_memory_finalization_candidate_word": &self.count_route_memory_finalization_candidate_word,
            "count_route_memory_finalization_candidate_target_letter": &self.count_route_memory_finalization_candidate_target_letter,
            "count_route_memory_finalization_candidate_parser_confidence": self.count_route_memory_finalization_candidate_parser_confidence,
            "count_route_memory_finalization_candidate_parser_version": &self.count_route_memory_finalization_candidate_parser_version,
            "count_route_memory_finalization_candidate_state": &self.count_route_memory_finalization_candidate_state,
            "count_route_memory_finalization_answer_signature_seen": &self.count_route_memory_finalization_answer_signature_seen,
            "count_route_memory_finalization_do_no_harm_protected": self.count_route_memory_finalization_do_no_harm_protected,
            "count_route_memory_finalization_would_apply": self.count_route_memory_finalization_would_apply,
            "count_route_memory_finalization_action_enabled": self.count_route_memory_finalization_action_enabled,
            "count_route_memory_finalization_action_applied": self.count_route_memory_finalization_action_applied,
            "count_route_memory_finalization_action_reason": &self.count_route_memory_finalization_action_reason,
            "count_route_memory_finalization_replacement_answer": &self.count_route_memory_finalization_replacement_answer,
            "count_route_memory_finalization_original_answer_window": &self.count_route_memory_finalization_original_answer_window,
            "count_route_memory_finalization_stop_reason": &self.count_route_memory_finalization_stop_reason,
            "prompt_embedding_source": &self.prompt_embedding_source,
            "prompt_vec_norm": self.prompt_vec_norm,
            "prompt_similarity_unavailable": self.prompt_similarity_unavailable,
        })
        .as_object()
        {
            obj.extend(recovery.clone());
        }

        // Compact digest for applied_ghost_vector: keeps force-diff and regression triage
        // possible without dumping ~4096 floats per token.
        if let Some(vec) = self.applied_ghost_vector.as_ref() {
            let sha256 = sha256_f32_le_hex(vec);
            let l2 = l2_norm_f32(vec);
            let head8: Vec<f32> = vec.iter().take(8).cloned().collect();
            // 16 block sums over the vector for quick "shape" comparisons.
            let block_sums_16 = block_sums_f32(vec, 16);
            obj.insert(
                "applied_ghost_vector_len".to_string(),
                serde_json::json!(vec.len()),
            );
            obj.insert(
                "applied_ghost_vector_sha256".to_string(),
                serde_json::json!(sha256),
            );
            obj.insert("applied_ghost_vector_l2".to_string(), serde_json::json!(l2));
            obj.insert(
                "applied_ghost_vector_head8".to_string(),
                serde_json::json!(head8),
            );
            obj.insert(
                "applied_ghost_vector_block_sums_16".to_string(),
                serde_json::json!(block_sums_16),
            );
        }

        value
    }
}

fn signature_hash(values: &[f32]) -> String {
    let mut hasher = DefaultHasher::new();
    for value in values {
        value.to_bits().hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut s, "{:02x}", b);
    }
    s
}

fn sha256_f32_le_hex(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for v in values {
        hasher.update(v.to_le_bytes());
    }
    let digest = hasher.finalize();
    bytes_to_hex(&digest)
}

fn l2_norm_f32(values: &[f32]) -> f32 {
    let sum = values
        .iter()
        .map(|v| {
            let v = *v as f64;
            v * v
        })
        .sum::<f64>();
    (sum.sqrt()) as f32
}

fn block_sums_f32(values: &[f32], blocks: usize) -> Vec<f32> {
    if blocks == 0 || values.is_empty() {
        return vec![];
    }
    let block_len = (values.len() + blocks - 1) / blocks;
    let mut out = Vec::with_capacity(blocks);
    for b in 0..blocks {
        let start = b * block_len;
        if start >= values.len() {
            break;
        }
        let end = ((b + 1) * block_len).min(values.len());
        let sum = values[start..end].iter().copied().sum::<f32>();
        out.push(sum);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_profile_omits_large_force_vector() {
        let mut record = TokenPhysics::default();
        record.token = "x".to_string();
        record.step = 7;
        record.applied_ghost_vector = Some(vec![1.0, 2.0, 3.0]);
        record.lock_detected = true;
        record.lock_stop_policy = "taper".to_string();

        let value = record.to_profile_value(TelemetryProfile::Score);
        assert_eq!(value["record_type"], "token");
        assert_eq!(value["step"], 7);
        assert!(value.get("applied_ghost_vector").is_none());
        assert_eq!(value["applied_ghost_vector_len"], 3);
        assert!(value.get("applied_ghost_vector_sha256").is_some());
        assert!(value.get("applied_ghost_vector_head8").is_some());
        assert!(value.get("applied_ghost_vector_block_sums_16").is_some());
        assert_eq!(value["lock_detected"], true);
    }

    #[test]
    fn score_profile_keeps_motif_bridge_fields() {
        let mut record = TokenPhysics::default();
        record.motif_force = 0.25;
        record.recovery_force = 0.5;
        record.total_force = 0.75;
        record.activation_gate = 0.125;
        record.live_motif_count = 2;
        record.bridge_motif_count = 640;
        record.bridge_force_selection = "routed".to_string();
        record.bridge_force_selected_count = 1;
        record.bridge_force_selected_ids = vec!["scalar_codec::bits07::record0001".to_string()];
        record.bridge_force_selection_source = "routing_cache".to_string();
        record.bridge_force_selected_score_max = Some(0.75);
        record.bridge_force_selected_role = Some("structured".to_string());
        record.bridge_force_second_score = Some(0.25);
        record.bridge_force_selected_margin = Some(0.5);
        record.bridge_force_role_filter = "structured".to_string();
        record.bridge_force_min_margin = 0.1;
        record.routed_motif_id = Some("scalar_codec::bits07::record0001".to_string());
        record.routed_motif_role = Some("structured".to_string());
        record.routed_motif_score = Some(0.125);
        record.route_surface_id = Some("scalar_codec::bits07::record0001".to_string());
        record.route_surface_source = Some("bridge_force_selection".to_string());
        record.route_surface_role = Some("structured".to_string());
        record.controller_candidate_count = Some(4);

        let value = record.to_profile_value(TelemetryProfile::Score);
        assert_eq!(value["motif_force"], 0.25);
        assert_eq!(value["recovery_force"], 0.5);
        assert_eq!(value["total_force"], 0.75);
        assert_eq!(value["activation_gate"], 0.125);
        assert_eq!(value["live_motif_count"], 2);
        assert_eq!(value["bridge_motif_count"], 640);
        assert_eq!(value["bridge_force_selection"], "routed");
        assert_eq!(value["bridge_force_selected_count"], 1);
        assert_eq!(
            value["bridge_force_selected_ids"][0],
            "scalar_codec::bits07::record0001"
        );
        assert_eq!(value["bridge_force_selection_source"], "routing_cache");
        assert_eq!(value["bridge_force_selected_score_max"], 0.75);
        assert_eq!(value["bridge_force_selected_role"], "structured");
        assert_eq!(value["bridge_force_second_score"], 0.25);
        assert_eq!(value["bridge_force_selected_margin"], 0.5);
        assert_eq!(value["bridge_force_role_filter"], "structured");
        let min_margin = value["bridge_force_min_margin"]
            .as_f64()
            .unwrap_or_default();
        assert!((min_margin - 0.1).abs() < 1e-6);
        assert_eq!(value["routed_motif_id"], "scalar_codec::bits07::record0001");
        assert_eq!(value["routed_motif_role"], "structured");
        assert_eq!(value["routed_motif_score"], 0.125);
        assert_eq!(
            value["route_surface_id"],
            "scalar_codec::bits07::record0001"
        );
        assert_eq!(value["route_surface_source"], "bridge_force_selection");
        assert_eq!(value["route_surface_role"], "structured");
        assert_eq!(value["controller_candidate_count"], 4);
    }

    #[test]
    fn minimal_profile_keeps_token_and_lock_status() {
        let mut record = TokenPhysics::default();
        record.token = "x".to_string();
        record.lock_stop_triggered = true;
        record.lock_stop_reason = Some("lock_taper_exhausted".to_string());

        let value = record.to_profile_value(TelemetryProfile::Minimal);
        assert_eq!(value["token"], "x");
        assert_eq!(value["lock_stop_triggered"], true);
        assert!(value.get("ghost_pull_delta_norm").is_none());
    }

    #[test]
    fn score_profile_keeps_tda_shadow_fields_when_enabled() {
        let mut record = TokenPhysics::default();
        record.tda_shadow_enabled = true;
        record.tda_shadow_window_ready = true;
        record.tda_shadow_decision_fresh = true;
        record.tda_shadow_action = "would_focus".to_string();
        record.tda_shadow_reason = "route_fragmentation_or_surface_churn_high".to_string();
        record.tda_shadow_breath_requested = true;
        record.tda_shadow_h1_total_persistence = 1.25;
        record.tda_shadow_involution_valid = true;

        let value = record.to_profile_value(TelemetryProfile::Score);
        assert_eq!(value["tda_shadow_enabled"], true);
        assert_eq!(value["tda_shadow_window_ready"], true);
        assert_eq!(value["tda_shadow_decision_fresh"], true);
        assert_eq!(value["tda_shadow_action"], "would_focus");
        assert_eq!(
            value["tda_shadow_reason"],
            "route_fragmentation_or_surface_churn_high"
        );
        assert_eq!(value["tda_shadow_breath_requested"], true);
        assert_eq!(value["tda_shadow_h1_total_persistence"], 1.25);
        assert_eq!(value["tda_shadow_involution_valid"], true);
    }

    #[test]
    fn score_profile_keeps_mistake_reflex_prompt_fields() {
        let mut record = TokenPhysics::default();
        record.mistake_reflex_prompt_applied = true;
        record.mistake_reflex_prompt_injection_timing = Some("pre_decode".to_string());
        record.mistake_reflex_prompt_injection_repeated = true;
        record.mistake_reflex_prompt_hint_text = Some("show the unit conversion work".to_string());

        let value = record.to_profile_value(TelemetryProfile::Score);
        assert_eq!(value["mistake_reflex_prompt_applied"], true);
        assert_eq!(
            value["mistake_reflex_prompt_injection_timing"],
            "pre_decode"
        );
        assert_eq!(value["mistake_reflex_prompt_injection_repeated"], true);
        assert_eq!(
            value["mistake_reflex_prompt_hint_text"],
            "show the unit conversion work"
        );
    }

    #[test]
    fn score_profile_keeps_specialist_worker_shadow_fields() {
        let mut record = TokenPhysics::default();
        record.specialist_worker_enabled = true;
        record.specialist_worker_mode = "shadow".to_string();
        record.specialist_worker_selected_id = Some("worker::001".to_string());
        record.specialist_worker_packet_id = Some("packet::001".to_string());
        record.specialist_worker_unicode_escape = Some("\\ue000\\ue001".to_string());
        record.specialist_worker_original_route_id = Some("motif::a".to_string());
        record.specialist_worker_decoded_route_id = Some("motif::a".to_string());
        record.specialist_worker_route_preserved = Some(true);
        record.specialist_worker_topk_hit = Some(true);
        record.specialist_worker_score = Some(0.875);
        record.specialist_worker_source_prompt_id = Some("restore_owner_project_jason".to_string());
        record.specialist_worker_direction_source = Some("decoded_64d_normalized".to_string());
        record.specialist_worker_delta_norm_64d = Some(0.25);
        record.specialist_worker_hidden_delta_norm = Some(2.0);
        record.specialist_worker_influence_clamp = Some(0.03);
        record.specialist_worker_influence_scale = Some(0.015);
        record.specialist_worker_probe_signature_64d = Some(vec![0.1, 0.2]);
        record.specialist_worker_target_signature_64d = Some(vec![0.3, 0.4]);

        let value = record.to_profile_value(TelemetryProfile::Score);
        assert_eq!(value["specialist_worker_enabled"], true);
        assert_eq!(value["specialist_worker_mode"], "shadow");
        assert_eq!(value["specialist_worker_selected_id"], "worker::001");
        assert_eq!(value["specialist_worker_packet_id"], "packet::001");
        assert_eq!(value["specialist_worker_unicode_escape"], "\\ue000\\ue001");
        assert_eq!(value["specialist_worker_route_preserved"], true);
        assert_eq!(value["specialist_worker_topk_hit"], true);
        assert_eq!(value["specialist_worker_score"], 0.875);
        assert_eq!(
            value["specialist_worker_direction_source"],
            "decoded_64d_normalized"
        );
        assert_eq!(value["specialist_worker_delta_norm_64d"], 0.25);
        assert_eq!(value["specialist_worker_hidden_delta_norm"], 2.0);
        let influence_clamp = value["specialist_worker_influence_clamp"]
            .as_f64()
            .unwrap_or_default();
        let influence_scale = value["specialist_worker_influence_scale"]
            .as_f64()
            .unwrap_or_default();
        assert!((influence_clamp - 0.03).abs() < 1e-6);
        assert!((influence_scale - 0.015).abs() < 1e-6);
        assert_eq!(value["specialist_worker_direction_auditable"], true);
        assert_eq!(value["specialist_worker_probe_signature_len_64d"], 2);
        assert_eq!(value["specialist_worker_target_signature_len_64d"], 2);
        assert_eq!(
            value["specialist_worker_probe_signature_hash"],
            signature_hash(&[0.1, 0.2])
        );
        assert_eq!(
            value["specialist_worker_target_signature_hash"],
            signature_hash(&[0.3, 0.4])
        );
        assert!(value.get("specialist_worker_probe_signature_64d").is_none());
        assert!(value
            .get("specialist_worker_target_signature_64d")
            .is_none());
        assert_eq!(
            value["specialist_worker_source_prompt_id"],
            "restore_owner_project_jason"
        );
    }
}
