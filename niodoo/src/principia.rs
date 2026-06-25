//! PrincipiaEngine: physics-driven decode-loop core (struct + 2 impl + PhysicsEngine trait impl).
//! Extracted from main.rs as part of the comprehensive refactor
//! (pre-refactor-main-split-20260508 backup).
//!
//! NOTE: this file pulls a wildcard `use crate::*;` because the engine touches
//! the majority of crate-root types/fns (Args, model loader, telemetry,
//! correction packets, agency hands, compact resume, etc.). Subsequent
//! refactors should narrow the import surface.

#![allow(unused_imports)]

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use crate::cli::*;
use crate::physics::naked_llama::{ModelKvCacheSnapshot, PhysicsEngine, QuantizedNakedLlama};
use crate::physics::optimizer::PhysicsParams;
use crate::physics::qwen35_hybrid::{
    summarize_qwen35_metadata, QuantizedQwen35Hybrid, Qwen35GgufMetadata,
};
use crate::physics::sensors::Sensor;
use crate::runtime::activation::*;
use crate::runtime::finalization::{
    AnswerBoundaryFinalizer, FinalizationController, LockStopPolicy,
};
use crate::runtime::secret_sauce_codec::*;
use crate::runtime::state_types::*;
use crate::runtime::tda_monitor::TdaShadowMonitor;
use crate::runtime::telemetry::{ForceEngineStatus, TelemetryProfile, TokenPhysics};
use crate::runtime::timing::*;
use crate::*;

fn hidden_trajectory_step(motif_id: &str) -> Option<usize> {
    motif_id
        .strip_prefix("hidden_trajectory::")
        .and_then(|_| motif_id.rsplit_once("::step_").map(|(_, step)| step))
        .and_then(|step| step.parse::<usize>().ok())
}

pub(crate) struct PrincipiaEngine {
    #[allow(dead_code)]
    pub(crate) mass_tensor: Tensor,
    pub(crate) charge_tensor: Tensor,
    pub(crate) kv_prefix_charge_tensor: Tensor,
    pub(crate) particle_words: Vec<String>,
    #[allow(dead_code)]
    pub(crate) sensors: Vec<Box<dyn Sensor>>,
    #[allow(dead_code)]
    pub(crate) vae: Option<ManifoldVAE>,
    #[allow(dead_code)]
    pub(crate) sigma: Option<Tensor>,
    #[allow(dead_code)]
    pub(crate) attractors: Vec<Attractor>,

    // NIODOO STATE
    pub(crate) vad_head: Option<VADHead>,
    pub(crate) sentence_history: VecDeque<SentenceParticle>,

    // Dynamics
    #[allow(dead_code)]
    pub(crate) start_logits: Option<Tensor>,
    #[allow(dead_code)]
    pub(crate) graviton_proj: Option<Tensor>,
    pub(crate) layer_norms: std::collections::HashMap<usize, f32>,
    pub(crate) last_deltas: std::collections::HashMap<usize, Tensor>,
    pub(crate) params: PhysicsParams,
    pub(crate) evo_population: BinaryHeap<EvoEntry>,
    pub(crate) symbolic_solver: Option<SymbolicModule>,
    pub(crate) pinn_loss: Option<Tensor>,
    pub(crate) lpm_collaborator: Option<LPMInterface>,
    pub(crate) black_hole_embeddings: Vec<Tensor>,

    // V3 Stubs
    pub(crate) geometric_dl: Option<GraphConv>,
    pub(crate) deepmd_kit: Option<DeePMDKit>,
    pub(crate) nvidia_physicsnemo: Option<PhysicsNeMo>,

    pub(crate) current_step: usize,
    pub(crate) current_sentence_embeddings: Vec<Tensor>,
    #[allow(dead_code)]
    pub(crate) current_surprisals: Vec<f32>,
    pub(crate) current_sentence_tokens: Vec<u32>,
    pub(crate) goal_embedding: Option<Tensor>,
    pub momentum_buffer: Option<Tensor>,
    pub(crate) secret_sauce_hidden_prior: Option<Tensor>,
    pub(crate) secret_sauce_sentence_prior: Option<Tensor>,
    pub(crate) secret_sauce_momentum_prior: Option<Tensor>,
    pub(crate) secret_sauce_version: Option<SecretSauceVersion>,
    pub(crate) secret_sauce_decay_steps: usize,
    pub(crate) secret_sauce_steps_remaining: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) emb_dim: usize,
    pub(crate) proj_matrix: Option<Tensor>,
    pub(crate) physics_blend: f32,
    pub(crate) physics_start_layer: usize,
    pub(crate) physics_end_layer: usize,
    pub(crate) multiplicative_blend: bool,
    pub(crate) runtime_mode: RuntimeMode,
    pub(crate) hidden_request_inference: bool,
    pub(crate) ui_events_json: bool,
    pub(crate) runtime_speed_profile: RuntimeSpeedProfile,
    pub(crate) stdout_profile: StdoutProfile,
    pub(crate) bridge_force_layer_policy: BridgeForceLayerPolicy,
    pub(crate) bridge_force_layer: usize,
    pub(crate) bridge_force_selection: BridgeForceSelection,
    pub(crate) bridge_force_trajectory_schedule: BridgeForceTrajectorySchedule,
    pub(crate) bridge_force_role_filter: BridgeForceRoleFilter,
    pub(crate) bridge_force_min_margin: f32,
    pub(crate) secret_sauce_capture_policy: SecretSauceCapturePolicy,
    pub(crate) specialist_memory_workers: Vec<RuntimeSpecialistMemoryWorker>,
    pub(crate) specialist_memory_workers_mode: SpecialistMemoryWorkerMode,
    pub(crate) specialist_memory_worker_top_k: usize,
    pub(crate) specialist_memory_worker_influence_clamp: f32,
    pub(crate) specialist_memory_worker_influence_sign: f32,
    pub(crate) specialist_memory_worker_influence_scope: SpecialistMemoryWorkerInfluenceScope,
    pub(crate) specialist_memory_worker_influence_direction:
        SpecialistMemoryWorkerInfluenceDirection,
    pub(crate) specialist_memory_worker_influence_layers: Option<(usize, usize)>,
    pub(crate) specialist_memory_worker_answer_window_active: bool,
    pub(crate) specialist_memory_worker_pre_answer_active: bool,
    pub(crate) specialist_memory_worker_pre_earned_active: bool,
    pub(crate) specialist_memory_worker_was_pre_answer_active: bool,
    pub(crate) specialist_memory_worker_at_boundary_active: bool,
    pub(crate) specialist_memory_worker_fixed_packet_id: Option<String>,
    pub(crate) runtime_motifs: Vec<RuntimeMotifField>,
    pub(crate) runtime_recovery_ops: Vec<RuntimeRecoveryOperator>,
    pub(crate) control_token_ids: HashSet<u32>,
    pub(crate) hidden_request_profiles: Vec<RequestSurfaceProfile>,
    pub(crate) motif_force_scale: f32,
    pub(crate) bridge_motif_gate_floor: f32,
    pub(crate) recovery_force_scale: f32,
    pub(crate) guardrail_bias_scale: f32,

    // Phase 1 Telemetry
    #[allow(dead_code)]
    pub last_force_trace: Option<TokenPhysics>,
    pub last_gravity_mag: f32,
    pub last_ghost_pre_norm: f32,
    pub last_ghost_gain: f32,
    pub last_applied_ghost_mag: f32,
    pub last_applied_ghost_vector: Option<Vec<f32>>,
    pub last_goal_mag: f32,
    pub last_repulsion_mag: f32,
    pub last_activation_gate: f32,
    pub last_motif_mag: f32,
    pub last_bridge_force_selection: String,
    pub last_bridge_force_selected_count: usize,
    pub last_bridge_force_selected_ids: Vec<String>,
    pub last_bridge_force_selection_source: String,
    pub last_bridge_force_selected_score_max: Option<f32>,
    pub last_bridge_force_selected_role: Option<String>,
    pub last_bridge_force_second_score: Option<f32>,
    pub last_bridge_force_selected_margin: Option<f32>,
    pub last_bridge_force_role_filter: String,
    pub last_bridge_force_min_margin: f32,
    pub last_recovery_mag: f32,
    pub last_absence_signal: f32,
    pub last_trap_score: f32,
    pub last_live_motif_count: usize,
    pub last_live_motif_distance: f32,
    pub last_live_motif_radius: f32,
    pub last_live_basin_pressure: f32,
    pub last_guardrail_active: bool,
    pub last_forces_applied: bool,
    pub last_engine_status: ForceEngineStatus,
    pub last_wobble_pressure_crossing: bool,
    pub last_task_anchor_clamp: Option<(f32, f32)>,

    // Phase 2 State
    pub orbital_active: bool,
    pub momentum: Vec<f32>,
    pub braking: bool,
    pub dynamic_gravity: f32,
    pub dynamic_repulsion: f32,

    // Phase 4: Heartbeat (Autonomic Regulation) + Defibrillator
    pub stress_buffer: VecDeque<f32>,
    #[allow(dead_code)]
    pub heartbeat_blend: f32,
    pub heartbeat_gravity: f32,
    #[allow(dead_code)]
    pub heartbeat_repulsion: f32,
    pub stress_level: f32,
    pub boredom_level: f32,
    pub defibrillator_active: bool, // 1-step transient spike
    pub defib_cooldown: usize,      // Tokens to wait before next defib
    pub adrenaline: f32,            // Decaying energy boost (5->4->3->2->1->0)

    // Phase 3: The Mirror (Context Injection)
    pub pending_insight: Option<String>, // Insight to inject at next sentence boundary
    #[allow(dead_code)]
    pub last_insight_step: usize, // Prevent spam (min 10 tokens between insights)
    pub insight_persistence: usize,      // How many tokens in bad state (for escalation)

    // Phase 4: Autonomic Override (Model-Requested Physics)
    pub request_count: usize,      // Requests this generation (max 5)
    pub last_request_token: usize, // Token of last request (cooldown 15)
    pub visible_request_gate: bool,
    pub request_buffer: String, // Buffer for multi-token request detection
    pub surface_buffer: String, // Rolling output window for clean-mode surface filtering
    pub hidden_request_candidate: Option<RequestType>,
    pub hidden_request_streak: usize,
    pub last_hidden_request: Option<RequestType>,
    pub last_hidden_request_pressure: f32,
    pub hidden_request_activations: usize,
    pub empathy_spike: f32,
    pub current_turn_structure_bias: f32,
    pub current_task_anchor_signature: Option<Vec<f32>>,
    pub task_anchor_similarity_start: f32,
    pub task_anchor_similarity_hinge: f32,
    pub task_anchor_similarity_24tok: f32,
    pub task_anchor_drift: f32,
    pub task_anchor_window_tokens_seen: usize,
    pub first_promotion_attempt_step: Option<usize>,
    pub structured_streak: usize,
    pub max_structured_streak: usize,
    pub promotion_attempt_count: usize,
    pub promotion_failure_count: usize,
    pub first_organic_promoted_step: Option<usize>,
    pub first_recovered_promoted_step: Option<usize>,
    pub motif_restore_bias_steps_remaining: usize,
    pub motif_restore_bias_strength: f32,
    pub reentry_clamp_steps_remaining: usize,
    pub reentry_clamp_strength: f32,
    pub reentry_temp_scale: f32,
    pub motif_regression_assist_steps_remaining: usize,
    pub motif_regression_assist_strength: f32,
    pub(crate) restored_run_active: bool,
    pub(crate) current_run_id: String,
    pub(crate) routing_cache: Option<RoutingDecisionCache>,
    pub(crate) controller_tick_count: usize,
    pub(crate) controller_selected_structured_count: usize,
    pub(crate) controller_selected_structured_candidate_count: usize,
    pub(crate) controller_selected_conversational_count: usize,
    pub(crate) conflict_tie_break_count: usize,
    pub(crate) structured_basin_lock_count: usize,
    pub(crate) neutral_basin_penalty_applied: usize,
    pub(crate) task_utility_bonus_applied: usize,
    pub(crate) structured_candidate_escalation_attempts: usize,
    pub(crate) structured_candidate_escalation_wins: usize,
    pub(crate) structured_candidate_loss_reason_counts: BTreeMap<String, usize>,
    pub(crate) structured_resume_window_remaining: usize,
    pub(crate) structured_resume_conversational_hits: usize,
    pub(crate) last_routed_motif_id: Option<String>,
    pub(crate) last_routed_motif_role: Option<String>,
    pub(crate) last_routed_motif_score: f32,
    pub(crate) last_controller_candidates: Vec<ControllerCandidateRecord>,
    pub(crate) hinge_window_records: Vec<HingeWindowTickRecord>,
    pub(crate) continuity_support_scale: f32,
    pub(crate) continuity_release_scale: f32,
    pub(crate) ablate_periodic_controller: bool,
    pub(crate) ablate_live_motifs: bool,
    pub(crate) ablate_conflict_routing: bool,
    pub(crate) ablate_reentry_clamp: bool,
    pub(crate) ablate_crystal_ratchet: bool,
    pub(crate) ablate_promotion_override: bool,
    // Distance bridge: routing stickiness/persistence for structured candidates.
    pub(crate) routing_stickiness_motif_id: Option<String>,
    pub(crate) routing_stickiness_remaining_ticks: usize,
    // Focus lock: when FOCUS is accepted, hold low-blend/low-repulsion for N ticks.
    pub(crate) focus_lock_remaining_ticks: usize,
    pub(crate) focus_lock_max_ticks: usize,
    pub(crate) tda_shadow_monitor_enabled: bool,
    pub(crate) tda_shadow_breath_apply: bool,
    pub(crate) tda_shadow_monitor: TdaShadowMonitor,

    // Dev runtime overrides for routing/scorer constants (0.0 = use default constant).
    pub(crate) dev_structured_candidate_task_sim: f32,
    pub(crate) dev_structured_candidate_bonus_scale: f32,
    pub(crate) dev_neutral_basin_penalty_scale: f32,
    pub(crate) dev_task_utility_bonus_scale: f32,
    pub(crate) dev_fragmentation_discount: f32,
    pub(crate) dev_restored_topology_floor_signal: f32,
    pub(crate) dev_restored_topology_floor_tightness: f32,
    pub(crate) dev_structured_candidate_escalation_topology: f32,
    pub(crate) dev_structured_candidate_escalation_task: f32,
    pub(crate) dev_routing_stickiness_bonus: f32,
    pub(crate) dev_routing_stickiness_ticks: f32,

    // Bridge Telemetry (niodv4_bridge)
    pub bridge_enabled: bool,
    pub bridge_influence_smoke: bool,
    pub bridge_influence_smoke_clamp: f32,
    pub bridge_influence_selective: bool,
    pub bridge_gate34_latch: bool,
    pub gate34_warmup_steps: u32,
    pub gate34_hold_steps: u32,
    pub gate34_release_margin_floor: f32,
    pub gate34_release_patience: u32,
    pub gate34_release_distance_mult: f32,
    pub gate34_acquire_top_k: usize,
    pub bridge_prompt_weight: f32,
    pub gate34_target_source: String,
    pub gate34_motif_routing_safety_floor: f32,
    pub prompt_vec: Option<Vec<f32>>,
    pub prompt_vec_norm: f32,
    pub prompt_embedding_source: String,
    pub prompt_similarity_unavailable: bool,
    pub gate34_acquisition_candidates: Vec<Gate34CandidateRecord>,
    pub gate34_phase: Gate34Phase,
    pub gate34_target_ghost_id: Option<String>,
    pub gate34_target_specialist_id: Option<String>,
    pub gate34_target_motif_id: Option<String>,
    pub gate34_target_vector: Option<Tensor>,
    pub gate34_target_acquired_step: i64,
    pub gate34_target_margin_at_acquire: f32,
    pub gate34_target_distance_at_acquire: f32,
    pub gate34_current_target_distance: f32,
    pub gate34_warmup_step_count: u32,
    pub gate34_held_step_count: u32,
    pub gate34_last_step: i64,
    pub gate34_bad_margin_count: u32,
    pub gate34_bad_distance_count: u32,
    pub gate34_release_reason: Option<String>,
    pub gate34_intervention_count: u32,
    pub gate34_target_switch_count: u32,
    pub gate34_candidate_counts: HashMap<String, u32>,
    pub gate34_candidate_margin_sum: HashMap<String, f32>,
    pub gate34_candidate_best_margin: HashMap<String, f32>,
    pub gate34_candidate_distance_sum: HashMap<String, f32>,
    pub gate34_candidate_distance_sq_sum: HashMap<String, f32>,
    pub gate34_candidate_distance_min: HashMap<String, f32>,
    pub gate34_candidate_distance_max: HashMap<String, f32>,
    pub gate34_target_warmup_distance_min: f32,
    pub gate34_target_warmup_distance_mean: f32,
    pub gate34_target_warmup_distance_max: f32,
    pub gate34_target_warmup_distance_std: f32,
    pub gate34_last_distance_drift_score: f32,
    pub gate34_last_distance_limit_ratio: f32,
    pub gate34_last_distance_limit_warmup: f32,
    pub gate34_last_distance_gate_mode: String,
    pub bridge_margin_threshold: f32,
    pub bridge_stability_k: u32,
    pub bridge_cooldown_after_switch: u32,
    pub bridge_scale_by_margin: bool,
    pub last_ghost_id_run_length: u32,
    pub last_ghost_switch_cooldown_remaining: u32,
    pub last_bridge_counter_step: i64,
    pub last_bridge_cooldown_step: i64,
    pub ghost_basins_loaded: usize,
    pub last_prompt_hash: String,
    pub last_nearest_ghost_id: Option<String>,
    pub last_nearest_ghost_distance: f32,
    pub last_second_nearest_ghost_distance: f32,
    pub last_route_margin: f32,
    pub last_bridge_route_probe_64d: Vec<f32>,
    pub last_projection_strategy: String,
    pub last_ghost_pull_delta_norm: f32,
    pub last_intervention_applied: bool,
    pub active_recovery_specialist_id: Option<String>,
    pub active_recovery_weight: f32,
    pub specialist_run_length: u32,
    pub last_recovery_specialist_id: Option<String>,
    pub last_recovery_counter_step: i64,
    pub last_specialist_worker_enabled: bool,
    pub last_specialist_worker_selected_id: Option<String>,
    pub last_specialist_worker_packet_id: Option<String>,
    pub last_specialist_worker_unicode_escape: Option<String>,
    pub last_specialist_worker_original_route_id: Option<String>,
    pub last_specialist_worker_decoded_route_id: Option<String>,
    pub last_specialist_worker_route_preserved: Option<bool>,
    pub last_specialist_worker_topk_hit: Option<bool>,
    pub last_specialist_worker_score: Option<f32>,
    pub last_specialist_worker_source_prompt_id: Option<String>,
    pub last_specialist_worker_direction_source: Option<String>,
    pub last_specialist_worker_delta_norm_64d: Option<f32>,
    pub last_specialist_worker_hidden_delta_norm: Option<f32>,
    pub last_specialist_worker_influence_clamp: Option<f32>,
    pub last_specialist_worker_influence_scale: Option<f32>,
    pub last_specialist_worker_probe_signature_64d: Option<Vec<f32>>,
    pub last_specialist_worker_target_signature_64d: Option<Vec<f32>>,

    // Ghost Registry for niodv4_bridge
    #[cfg(feature = "niodv4_bridge")]
    pub ghost_registry: Option<crate::bridge::registry::GhostRegistry>,

    // VQ codec + phase2 specialist (niodv4_bridge integration)
    #[cfg(feature = "niodv4_bridge")]
    pub vq_codebook: Option<crate::bridge::CodebookVQ>,
    #[cfg(feature = "niodv4_bridge")]
    pub vq_specialist: Option<crate::bridge::RuleBasedSpecialist>,
    // Per-step VQ telemetry (unconditional so TokenPhysics fields are always populated)
    pub last_vq_code_assigned: Option<u8>,
    pub last_vq_encode_error: f32,
    pub last_correction_delta_norm: f32,
    pub last_specialist_activated: bool,
    /// Apply the codec-mediated specialist correction force this run (--specialist-correction-apply).
    pub specialist_correction_apply: bool,
    /// L2 clamp for the codec-mediated specialist force in 4096D hidden space.
    pub specialist_correction_clamp: f32,
    /// Set to true on a step when the codec-mediated specialist force was actually added to probe_force.
    pub last_specialist_force_applied: bool,
    /// L2 norm in 4096D space of the codec-mediated specialist force added this step.
    pub last_specialist_force_norm: f32,
    /// VQ code (bucket) used for correction-packet lookup this step. Distinct from
    /// `last_vq_code_assigned`, which is the rule-based specialist-side assignment.
    pub last_correction_packet_vq_code: Option<u8>,
    /// VQ-keyed correction-packet store (--correction-packets-path). Packets are looked up
    /// every step by the codec-encoded probe's vq_code. Each firing packet contributes a
    /// codec-mediated pull-toward-target force, summed into probe_force.
    #[cfg(feature = "niodv4_bridge")]
    pub correction_packets: Option<crate::bridge::CorrectionPacketStore>,
    /// Per-fire L2 clamp for codec-mediated correction-packet forces in 4096D space.
    pub correction_packet_clamp: f32,
    /// Optional blend of packet `payload_z_64d` into the 64D force direction.
    /// `0.0` preserves legacy target-only pull.
    pub correction_packet_payload_blend: f32,
    /// Number of correction packets that fired this step.
    pub last_correction_packet_fire_count: usize,
    /// Sum of L2 norms (4096D) of all correction-packet forces added this step.
    pub last_correction_packet_force_norm: f32,
    /// IDs of the packets that fired this step (small Vec for telemetry).
    pub last_correction_packet_ids: Vec<String>,
    /// Latest bucket-mean compressed probe (64D). Updated each apply_forces call so the
    /// end-of-run packet writer has a final probe to mint from.
    pub last_probe_bucket_mean_64: Option<[f32; 64]>,

    // === REMEMBER vault tether (64D probe -> Qdrant semantic flashback for self-curated memory) ===
    /// Direct client for the vault (prior chats + self memories). Created when --qdrant-url (or equivalent)
    /// is provided. Uses the live 64D probe captured at the exact step the model emits [REQUEST: REMEMBER].
    pub vault_client: Option<crate::runtime::vault_retrieval::VaultClient>,
    /// Collection name for the vault (historical "770MB" Grok/Claude + niodoo's own creations).
    /// Default "niodoo-4096-vault" matches the scripts that prep the prior_chats.
    pub vault_collection: String,
    /// Optional output path for end-of-run packet writer. When Some(path), the runtime
    /// appends one CorrectionPacket JSONL record at end-of-run capturing
    /// `last_probe_bucket_mean_64` as the target. None = writer disabled.
    pub correction_packets_out: Option<PathBuf>,
    /// When true, minted packets omit numeric `target_z_64d` and store Unicode-only
    /// `target_z_unicode_v3` instead (see `--correction-packet-out-unicode-v3`).
    pub correction_packet_out_unicode_v3: bool,
    /// RC5: when true, every packet minted this session (LOCK/REMEMBER/end-of-run
    /// capture) is ALSO inserted into the live in-memory firing store immediately
    /// after the file write, so it can fire on a later step of the SAME process
    /// instead of only after a restart that reloads the out-file. Default false so
    /// existing runs and replay determinism stay byte-identical until enabled via
    /// `--correction-packet-live-mint`. No effect unless a writer
    /// (`correction_packets_out`) is configured.
    pub correction_packet_live_mint: bool,
    /// RC5 telemetry: number of fires at the current step whose packet was
    /// live-minted into the store this session (proof the closed loop fired).
    pub last_correction_packet_live_minted_fired_count: usize,
    /// RC1: master enable for correction-packet outcome feedback (fire->measure->
    /// adjust). When true, each fired packet's effect is measured the NEXT step (did
    /// the probe move toward target?) and folded into a per-packet effectiveness EMA
    /// that scales its applied force — replacing blind fire_count decay as the
    /// adaptation signal. Default false => byte-identical legacy behavior.
    pub correction_packet_outcome_feedback: bool,
    /// RC1: gain on the EMA when scaling force: `factor = (1 + ema*gain).clamp(floor, ceil)`.
    pub correction_packet_outcome_gain: f32,
    /// RC1: EMA smoothing factor in [0, 1].
    pub correction_packet_outcome_ema_alpha: f32,
    /// RC1: lower clamp on the EMA force factor (a consistently-unhelpful packet is
    /// damped to at most this fraction of its base force, never fully zeroed unless 0).
    pub correction_packet_outcome_floor: f32,
    /// RC1: upper clamp on the EMA force factor (a consistently-helpful packet can be
    /// boosted up to this multiple of its base force).
    pub correction_packet_outcome_ceil: f32,
    /// RC1 telemetry: mean effectiveness EMA over packets that fired this step
    /// (0.0 when none measured).
    pub last_correction_packet_effectiveness_avg: f32,
    /// RC2: gain on the within-block residual shape when projecting a packet's force
    /// to 4096D. 0.0 = strict legacy flat 64-block smear (no-op). >0 rotates each
    /// block's force toward the captured residual direction so steering is no longer
    /// piecewise-constant. Magnitude stays bounded by the existing clamps.
    pub correction_packet_residual_gain: f32,
    /// RC2: most recent mint's within-block residual shape (unit 4096D vector). Captured
    /// alongside `last_probe_bucket_mean_64`; attached to live-minted packets.
    pub last_probe_residual_shape_4096: Option<Vec<f32>>,
    /// RC2 telemetry: number of fires this step whose force used a residual shape.
    pub last_correction_packet_residual_applied: usize,
    /// `pull_strength` written on minted packets.
    pub correction_packet_out_pull_strength: f32,
    /// `distance_threshold` written on minted packets.
    pub correction_packet_out_distance_threshold: f32,
    /// `pull_strength` written on LOCK-derived "earned answer" packets (see
    /// `--correction-packet-lock-pull-strength`).
    pub correction_packet_lock_pull_strength: f32,
    /// Multiplier applied to lock pull strength when the agency-hands learning-event
    /// signal fires this turn (user contradicted a prior LOCK). See
    /// `--correction-packet-lock-contradiction-multiplier`.
    pub correction_packet_lock_contradiction_multiplier: f32,
    /// Whether to explicitly invalidate prior correction packets matching the
    /// contradicted LOCK payload's hash (see
    /// `--correction-packet-invalidate-on-contradiction`).
    pub correction_packet_invalidate_on_contradiction: bool,
    /// Whether to revalidate previously invalidated packets matching a LOCK
    /// payload's hash on every LOCK emission (see
    /// `--correction-packet-revalidate-on-affirmation`).
    pub correction_packet_revalidate_on_affirmation: bool,
    /// Cap on the per-payload-key contradiction count for adaptive multiplier
    /// scaling (see `--correction-packet-adaptive-contradiction-cap`).
    pub correction_packet_adaptive_contradiction_cap: u64,
    /// Per-payload-key contradiction counts. Incremented each time the user
    /// contradicts a LOCK with the given key. Read by
    /// `record_contradiction_for_key` to compute the effective multiplier.
    /// Persisted via `--correction-contradiction-counts-path` when set, so the
    /// escalation accumulates across sessions (§10ad).
    pub contradiction_counts: std::collections::HashMap<String, u64>,
    /// Optional JSONL path for persisting contradiction counts. When set, the
    /// runtime loads at startup and atomically rewrites at end-of-run.
    pub correction_contradiction_counts_path: Option<PathBuf>,
    /// Threshold on retry-count for the secondary relapse trigger
    /// (`--correction-packet-unfold-on-retry-count`). 0 disables it.
    pub correction_packet_unfold_on_retry_count: usize,
    /// Mistake-reflex retry count carried into this turn's generation. Set by
    /// the chat REPL before each forward pass. apply_forces reads this to fire
    /// the secondary relapse trigger when it reaches the threshold.
    pub last_mistake_reflex_retry_count: usize,
    /// Per-source unfold factor for retry-relapse (§10af).
    /// 0.0 = inherit `correction_packet_unfold_factor` (legacy single-factor).
    /// > 1.0 = override; applied factor on a step is max of source factors.
    pub correction_packet_unfold_retry_factor: f32,
    /// Eviction floor for long-decayed packets, as ratio of effective_pull to
    /// pull_strength (§10ai). 0 disables.
    pub correction_packet_eviction_floor: f32,
    /// Per-fire decay rate for correction packets (see `--correction-packet-decay-rate`).
    /// `effective_pull = pull_strength * decay_rate.powi(fire_count)`. Values outside
    /// `(0.0, 1.0)` mean "no decay" and the raw `pull_strength` is used.
    pub correction_packet_decay_rate: f32,
    /// Relapse trigger threshold on `vq_encode_error` (see
    /// `--correction-packet-unfold-encode-error-threshold`). When >0 and the current
    /// step's quantization error exceeds it, firing packets get unfold-boosted.
    pub correction_packet_unfold_encode_error_threshold: f32,
    /// Multiplicative boost applied to firing packets' effective pull when the relapse
    /// trigger activates (see `--correction-packet-unfold-factor`).
    pub correction_packet_unfold_factor: f32,
    /// Competence-aware suppression factor. When `last_sampling_entropy_norm` is below
    /// `correction_packet_competence_entropy_threshold` (confident next-token), each
    /// firing packet's effective pull is multiplied by this factor. Default 1.0 = no
    /// suppression. < 1.0 = reduce pull on competent steps. North Star bullet 7
    /// (preserve earned answers before drift). See §10aw.
    pub correction_packet_competence_suppress_factor: f32,
    /// Entropy threshold below which the previous step's sampling is "competent" and
    /// the suppress factor engages. Default 0.0 = disabled.
    pub correction_packet_competence_entropy_threshold: f32,
    /// Density threshold for the alternative competence trigger. When the number
    /// of firing packets at the current step is ≥ this value, trajectory is in
    /// a high-density correct-geometry region and the suppress factor engages.
    /// Default 0 = disabled.
    pub correction_packet_competence_density_threshold: usize,
    /// Distance threshold for the per-trajectory competence trigger. When the
    /// minimum 64D distance from probe to any matched packet's target_z is
    /// below this value, the suppress factor engages. Finer-grained than
    /// density. Default 0.0 = disabled.
    pub correction_packet_competence_distance_threshold: f32,
    /// Combine mode for competence triggers: "or" (any trigger engages) or
    /// "and" (all enabled triggers must agree). Defaults to "or" for legacy.
    pub correction_packet_competence_combine_mode: String,
    /// §10bd Track 2 v11: per-trajectory adaptive gate routing.
    pub correction_packet_trajectory_routing: bool,
    pub correction_packet_trajectory_classify_step: usize,
    pub correction_packet_trajectory_fire_count_threshold: f32,
    /// §10cq per-class top_k overrides for the trajectory router.
    /// 0 = no override (preserve §10cp legacy: router only flips
    /// combine_mode, not K).
    pub correction_packet_trajectory_top_k_competent: usize,
    pub correction_packet_trajectory_top_k_drifting: usize,
    /// RC6: per-trajectory decay rate. When trajectory routing is on and the turn is
    /// classified, the firing decay rate is routed by class (mirrors the top-k
    /// router). 0.0 = no override => fall back to the global `correction_packet_decay_rate`.
    /// Lets decay be non-monotonic within a run (e.g. competent=0.9 preserves earned
    /// answers, drifting=0.5 is the population optimum). Per-packet LOCK overrides
    /// (decay_rate=Some(1.0)) still take precedence, so earned answers don't regress.
    pub correction_packet_trajectory_decay_competent: f32,
    pub correction_packet_trajectory_decay_drifting: f32,
    /// RC6 telemetry: the decay rate actually used this step after class routing.
    pub last_correction_packet_effective_decay_rate: f32,
    /// Per-turn rolling sum of `last_correction_packet_fire_count`. Reset
    /// on turn_start by `reset_trajectory_routing_state`. Accumulated
    /// each gate call until classification. Tuned to fire_count rather
    /// than entropy: at temp=0 entropy is degenerate, but fire_count
    /// varies with bucket-match density (the §10az signal).
    pub trajectory_fire_count_sum: f32,
    pub trajectory_fire_count_samples: usize,
    /// Per-turn one-shot classification result. None until classify_step
    /// is reached, then "competent" or "drifting". Reset on turn_start.
    pub trajectory_classified: Option<String>,
    /// Per-turn step counter. Reset on turn_start. Distinct from the
    /// session-monotonic `current_step`.
    pub trajectory_turn_step: usize,
    /// §10bd v11.4: last `current_step` value at which the v11 accumulator
    /// fired. The gate function is called per-LAYER (multiple calls per
    /// decode step) but we want to count once per token. usize::MAX is
    /// the sentinel for "haven't seen any step yet."
    pub trajectory_last_classified_step: usize,
    /// §10bd v11.5: per-step accumulator summing firings.len() across
    /// all layer calls for the current decode step. On step boundary
    /// the sum is captured as one classifier sample, then reset.
    pub trajectory_pending_step_fires: usize,
    /// RC3: sliding window of recent per-step fire counts for mid-turn
    /// re-classification (cap = effective window length). Reset each turn.
    pub trajectory_window: std::collections::VecDeque<f32>,
    /// RC3: turn-step of the last (re)classification, gates the reclassify interval.
    pub trajectory_last_reclassify_step: usize,
    /// RC3: number of mid-turn label flips this turn (telemetry/proof the latch broke).
    pub trajectory_reclassify_count: u32,
    /// RC3 telemetry: most recent window mean used for re-classification.
    pub last_trajectory_window_mean: f32,
    /// RC3 config: re-classify every N turn-steps once classified. 0 = one-shot
    /// (legacy byte-identical). Requires `correction_packet_trajectory_routing`.
    pub correction_packet_trajectory_reclassify_interval: usize,
    /// RC3 config: sliding-window length for the window mean. 0 falls back to
    /// `correction_packet_trajectory_classify_step`.
    pub correction_packet_trajectory_window_len: usize,
    /// RC3 config: hysteresis band around the fire-count threshold to avoid thrash.
    pub correction_packet_trajectory_hysteresis: f32,
    /// §10bs threshold above which packets are suppressed when the
    /// previous step's applied_ghost_mag exceeds this value.
    /// `0.0` disables the gate (legacy).
    pub correction_packet_suppress_when_bridge_force_above: f32,
    /// §10bt cached previous-step max ghost_mag. NOT zeroed by
    /// reset_force_telemetry so the §10bs packet gate can see ghost
    /// activity from the prior step before the current step's bridge
    /// fires (which happens after the gate runs). Updated at the end
    /// of bridge force application; read by the §10bs gate.
    pub prev_step_max_ghost_mag: f32,
    /// §10bx post-bridge re-encoding mode (CLI flag mirror).
    pub correction_packet_post_bridge_mode: bool,
    /// DEEP_DIVE_ROADMAP P1-B mint-readiness lock threshold (CLI mirror).
    /// `0.0` disables; `0.55` is the deep-dive recommendation. Read inside
    /// `try_apply_correction_packet_force` against the active routing cache's
    /// motif `readiness_score`.
    pub correction_packet_readiness_lock_threshold: f32,
    /// DEEP_DIVE_ROADMAP P2-A — per-layer physics blend mask CLI mirrors.
    /// When `layer_idx >= physics_blend_deep_layer_from`, motif_force and
    /// recovery_force are scaled by `physics_blend_deep_layer_multiplier`
    /// before being added to probe_force. `from = 0` disables. Read at the
    /// bridge force application site near main.rs:15689.
    pub physics_blend_deep_layer_from: usize,
    pub physics_blend_deep_layer_multiplier: f32,
    /// Counter incremented each time the deep-layer mask actually scales a
    /// force. Surfaces in telemetry for audit.
    pub physics_blend_deep_layer_mask_count: usize,
    /// DEEP_DIVE_ROADMAP P2-B consensus-weight motif routing toggle.
    /// When true, `routing_score_for_motif` uses the softmax-shape
    /// `score -= exp(persistence - 0.08*conflict_ratio - 0.03*mixed_ratio)`
    /// instead of the legacy additive penalty.
    pub motif_routing_consensus_weight: bool,
    /// DEEP_DIVE_ROADMAP P2-C autonomic physics adaptation knobs (CLI mirrors).
    pub autonomic_physics_force_threshold: f32,
    pub autonomic_physics_window_size: usize,
    /// Rolling window of recent bridge-force magnitudes
    /// (last_motif_mag + last_recovery_mag) for P2-C feedback loop.
    pub autonomic_physics_force_window: VecDeque<f32>,
    /// Original motif_force_scale captured at engine init; used as ceiling
    /// for adaptive restore.
    pub autonomic_physics_motif_scale_origin: f32,
    pub autonomic_physics_recovery_scale_origin: f32,
    /// Counter for adaptive scale-down events.
    pub autonomic_physics_scale_down_count: usize,
    pub autonomic_physics_scale_up_count: usize,
    /// Counter incremented every time the readiness lock fires (i.e. the
    /// active motif's readiness exceeded the threshold and packet firing
    /// was suppressed). Surfaces in per-step telemetry alongside
    /// `conflict_tie_break_count`.
    pub readiness_lock_skip_count: usize,
    /// iter-58: which selection source last evaluated for the readiness
    /// lock — "routing_cache_motif", "active_recovery_specialist", or "" if
    /// neither path produced a candidate. Empty string when the lock is
    /// disabled (`threshold == 0.0`).
    pub last_readiness_lock_source: String,
    /// iter-58: highest readiness_score observed across the active selection
    /// sources at the most recent gate evaluation. 0.0 when the lock is
    /// disabled or neither selection source produced a candidate.
    pub last_readiness_lock_score: f32,
    /// Packet steering arbitration mode. Disabled preserves legacy force behavior.
    pub correction_packet_arbitration_mode: CorrectionPacketArbitrationMode,
    /// Auto arbitration threshold for standing down when competence suppression
    /// already indicates a healthy unsteered route.
    pub correction_packet_arbitration_healthy_factor_threshold: f32,
    /// Auto arbitration threshold for standing down when the nearest packet target
    /// is too far from the live probe and likely points toward a stale/wrong basin.
    pub correction_packet_arbitration_stale_distance_threshold: f32,
    /// Last selected arbitration mode for telemetry.
    pub last_correction_packet_arbitration_mode: String,
    /// Human-readable reason for the last arbitration choice.
    pub last_correction_packet_arbitration_reason: String,
    /// Candidate packet count considered before arbitration.
    pub last_correction_packet_arbitration_candidate_count: usize,
    /// Nearest target distance among candidate packets before arbitration.
    pub last_correction_packet_arbitration_min_target_distance: f32,
    /// Estimated force norm budget for candidate packets before arbitration.
    pub last_correction_packet_arbitration_force_norm_estimate: f32,
    /// §10bf prompt-level codec activation gate. Non-empty → gate is
    /// active. Set by `set_codec_active_prompt_substrings_csv` at
    /// engine init.
    pub codec_active_prompt_substrings: Vec<String>,
    /// §10ck per-prompt-substring → top-K override map. Each (substring,
    /// K) pair is matched case-insensitive against the user prompt; first
    /// match wins. Empty = no override.
    pub correction_packet_prompt_top_k_map: Vec<(String, usize)>,
    /// §10cm per-turn matched substring (bullet-10 observability).
    /// Holds the literal substring that triggered the §10ck top-K
    /// override, or None when no substring matched. Exposed via
    /// telemetry so a human can verify the routing rule actually fired.
    pub current_prompt_top_k_match_substring: Option<String>,
    /// §10cn task-neutrality gate flag (CLI mirror).
    pub correction_packet_suppress_when_no_prompt_match: bool,
    /// §10cn per-turn flag set by `apply_correction_packet_prompt_top_k_gate`:
    /// true when the prompt-K map is non-empty, no substring matched,
    /// and the suppress flag is set — meaning packets should NOT fire
    /// this turn (out-of-distribution for the trained reflex).
    pub correction_packet_suppress_for_current_prompt: bool,
    /// §10ck per-turn override resolved at turn start by
    /// `apply_correction_packet_prompt_top_k_gate`. None = no override
    /// active for this turn (use `correction_packet_fire_top_k`).
    pub current_prompt_top_k_override: Option<usize>,
    /// Prompt-substring -> source target id map. When matched for the current
    /// prompt, correction-packet firing is restricted to packets whose
    /// `source_label` carries `target_id=<id>`.
    pub correction_packet_prompt_source_target_map: Vec<(String, String)>,
    /// Per-turn source target override resolved at turn start. None = no
    /// source-target filter active for this turn.
    pub current_prompt_source_target_override: Option<String>,
    /// §10ck telemetry mirror of the top-K actually used by packet firing
    /// this turn. Equals the prompt override when present, otherwise the
    /// configured `correction_packet_fire_top_k`.
    pub last_correction_packet_effective_fire_top_k: usize,
    /// §10bf per-turn flag set at turn start by
    /// `apply_codec_prompt_gate`. When the gate is configured and the
    /// current prompt does NOT contain any active substring, this is
    /// false and codec/bridge force application skips. When the gate is
    /// not configured (codec_active_prompt_substrings is empty), this
    /// is always true (legacy behavior).
    pub codec_active_for_current_prompt: bool,
    /// Total per-step force budget across all firing packets (§10bb). When
    /// 0.0, no total clamp (legacy behavior — total force can scale linearly
    /// with store size). When > 0, the cumulative new_probe_force L2 norm is
    /// scaled down to this value if it exceeds the budget. Fixes the iter-46
    /// "10 teach events hurt holdouts" interference pattern.
    pub correction_packet_total_clamp: f32,
    /// Direction-aware firing upper-bound distance (§10bc). When > 0, packets
    /// whose probe-to-target distance EXCEEDS this value are NOT fired even
    /// if their bucket matched the probe. Filters out packets whose target_z
    /// is direction-misaligned with the current probe. Default 0.0 disables.
    pub correction_packet_fire_max_distance: f32,
    /// Top-K direction-aware firing (§10be). When > 0, after filtering, only
    /// the K packets whose target_z is closest to probe actually fire.
    /// Default 0 disables (legacy: fire all matched).
    pub correction_packet_fire_top_k: usize,
    /// Mint-time bucket cap (§10bf). When > 0, mint operations skip writing
    /// new packets if the current bucket already holds K packets. Forces
    /// store-level diversity. iter-62 root cause for U-shape: bucket
    /// concentration in larger teach sets reduces lift coverage.
    pub correction_packet_mint_bucket_cap: usize,
    /// Per-bucket mint counts accumulated during this run. Updated on each
    /// successful mint; consulted before each new mint when bucket cap > 0.
    /// Cleared at engine reset. Persisted across mints in the same process.
    pub mint_bucket_counts: std::collections::HashMap<u8, usize>,
    /// When true, a live_capture packet is minted from each turn's final
    /// probe regardless of agency-hands tag echoing (§10bg). Bypasses the
    /// LOCK/REMEMBER gate so non-counting prompts also produce packets.
    pub correction_packet_capture_every_turn: bool,
    /// Step-window fire-gate (§10bh). When Some((start, end)), packets only
    /// fire when current_step ∈ [start, end] inclusive. Outside the window
    /// the firing block returns zero force. Iter-228 motivation: lift
    /// mechanism is post-enumeration recovery shaping, so isolating early
    /// vs late phase firing tests which phase matters. None disables.
    pub correction_packet_fire_step_window: Option<(usize, usize)>,
    /// When true, only fire packets whose embedded `ph_<hash>` in packet_id
    /// matches the current prompt's hash (§10bd). Per-prompt isolation.
    /// Set by the chat loop via `set_current_prompt_hash`.
    pub correction_packet_fire_match_prompt_hash: bool,
    /// Current prompt's hash, set by the chat loop before each generation.
    /// Used by the prompt-hash filter when enabled. Empty by default.
    pub current_prompt_hash: String,
    /// Packet authority gate mode. Shadow is diagnostic-only; enforce drops
    /// weak/unknown authority candidates before force projection.
    pub correction_packet_authority_mode: CorrectionPacketAuthorityMode,
    pub last_packet_authority_score: f32,
    pub last_packet_authority_allowed: bool,
    pub last_packet_authority_reason: String,
    pub last_packet_authority_blocked_reason: String,
    /// Most recent step's minimum 64D distance from probe to any matched
    /// packet's target_z. Telemetry-only, populated when distance gate is
    /// enabled. f32::INFINITY when no firings or gate disabled.
    pub last_correction_packet_min_target_distance: f32,
    /// Most recent sampling step's normalized Shannon entropy in [0, 1]. Updated by
    /// the generation loop after each token is sampled; read by the next step's
    /// force-application path for competence-aware modulation. 0.0 default = the
    /// first step is treated as uncertain (no suppression).
    pub last_sampling_entropy_norm: f32,
    /// Average effective pull across all packets that fired this step (post-decay,
    /// post-unfold). Zero when no packets fired. Telemetry-only.
    pub last_correction_packet_effective_pull_avg: f32,
    /// Per-step competence suppression factor actually applied this step. 1.0 if
    /// no suppression (entropy >= threshold OR threshold disabled). Multiplied INTO
    /// each firing packet's effective_pull alongside decay and unfold. Telemetry.
    pub last_correction_packet_competence_factor: f32,
    /// True on a step when the relapse trigger fired and packet pulls were unfold-boosted.
    pub last_correction_packet_unfold_active: bool,
    /// `vq_encode_error` value used by the relapse trigger this step.
    pub last_correction_packet_vq_encode_error: f32,
    /// The unfold factor actually applied this step (`1.0` when unfold inactive).
    pub last_correction_packet_unfold_factor_applied: f32,
    /// Optional output path for end-of-run packet state persistence (atomic rewrite of
    /// the entire store with current fire counters). When Some, a subsequent run loading
    /// the same file via `--correction-packets-path` resumes decay/unfold from where
    /// this session left off.
    pub correction_packets_state_out: Option<PathBuf>,
}

impl PrincipiaEngine {
    #[inline]
    pub(crate) fn eval_fast(&self) -> bool {
        self.runtime_speed_profile.is_eval_fast()
    }

    #[inline]
    pub(crate) fn stdout_debug(&self) -> bool {
        self.stdout_profile.debug_enabled()
    }

    #[inline]
    pub(crate) fn stdout_telemetry(&self) -> bool {
        self.stdout_profile.telemetry_enabled()
    }

    #[inline]
    pub(crate) fn bridge_force_layer_selected(&self, layer_idx: usize) -> bool {
        match self.bridge_force_layer_policy {
            BridgeForceLayerPolicy::All => true,
            BridgeForceLayerPolicy::Single => layer_idx == self.bridge_force_layer,
        }
    }

    /// Route-memory worker influence may target an explicit transformer layer band disjoint from physics.
    #[inline]
    pub(crate) fn specialist_worker_influence_lane_layer(&self, layer_idx: usize) -> bool {
        if self.specialist_memory_workers_mode != SpecialistMemoryWorkerMode::Influence {
            return false;
        }
        match self.specialist_memory_worker_influence_layers {
            Some((lo, hi)) => layer_idx >= lo && layer_idx <= hi,
            None => self.bridge_force_layer_selected(layer_idx),
        }
    }

    pub(crate) fn finalize_worker_only_residual_from_flat_force(
        &mut self,
        state_f32: &Tensor,
        masked_delta_flat: &Tensor,
        b_sz: usize,
        seq_len: usize,
        hidden_sz: usize,
        layer_idx: usize,
        device: &Device,
        original_state: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let mask_val = if layer_idx < 31 { 1.0f32 } else { 0.02f32 };
        let mask_t = Tensor::new(mask_val, device)?;
        let masked_delta = masked_delta_flat.broadcast_mul(&mask_t)?;

        let final_delta = if seq_len > 1 {
            let zeros_ctx = Tensor::zeros((b_sz, seq_len - 1, hidden_sz), DType::F32, device)?;
            let probe_reshaped = masked_delta.reshape((b_sz, 1, hidden_sz))?;
            Tensor::cat(&[&zeros_ctx, &probe_reshaped], 1)?
        } else {
            masked_delta.reshape((b_sz, seq_len, hidden_sz))?
        };

        let mut final_delta = final_delta;
        if self.last_wobble_pressure_crossing {
            if let Ok(wobble) = Tensor::randn(0.0f32, 0.06, final_delta.shape(), device) {
                if let Ok(new_delta) = final_delta.add(&wobble) {
                    final_delta = new_delta;
                }
            }
        }

        use candle_core::D;
        if layer_idx >= 30 {
            let proposed_state = (state_f32.clone() + &final_delta)?;
            let original_norm = state_f32.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
            let proposed_norm = proposed_state.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
            let scale = (original_norm / (proposed_norm + 1e-6)?)?;
            let fixed_state = proposed_state.broadcast_mul(&scale)?;
            let repaired_delta = (&fixed_state - state_f32)?;
            Ok(repaired_delta.to_dtype(original_state.dtype())?)
        } else {
            Ok(final_delta.to_dtype(original_state.dtype())?)
        }
    }

    #[inline]
    pub(crate) fn secret_sauce_capture_enabled(
        &self,
        step: usize,
        effective_max_steps: usize,
        stop_pending: bool,
    ) -> bool {
        match self.secret_sauce_capture_policy {
            SecretSauceCapturePolicy::PerToken => true,
            SecretSauceCapturePolicy::Final => stop_pending || step + 1 >= effective_max_steps,
            SecretSauceCapturePolicy::Off => false,
        }
    }

    #[inline]
    pub(crate) fn bridge_influence_smoke_active(&self) -> bool {
        self.bridge_enabled
            && self.bridge_influence_smoke
            && self.last_nearest_ghost_id.is_some()
            && self.codec_active_for_current_prompt
    }

    #[inline]
    pub(crate) fn bridge_influence_selective_active(&self) -> bool {
        self.bridge_enabled
            && self.bridge_influence_selective
            && self.last_nearest_ghost_id.is_some()
    }

    #[inline]
    pub(crate) fn bridge_gate34_latch_active(&self) -> bool {
        self.bridge_enabled && self.bridge_gate34_latch
    }

    pub(crate) fn bridge_motif_gate_floor(&self) -> f32 {
        if self.bridge_motif_gate_floor <= 1e-6 {
            return 0.0;
        }
        let has_bridge_motif = self
            .runtime_motifs
            .iter()
            .any(|motif| motif.motif_kind == "bridge");
        if has_bridge_motif {
            self.bridge_motif_gate_floor.clamp(0.0, 0.2)
        } else {
            0.0
        }
    }

    fn trajectory_scheduled_bridge_indices(
        &self,
        candidate_indices: &[usize],
    ) -> Option<Vec<usize>> {
        if self.bridge_force_trajectory_schedule == BridgeForceTrajectorySchedule::Off {
            return None;
        }
        let mut best_step = None;
        let mut best_distance = usize::MAX;
        for idx in candidate_indices {
            let motif = &self.runtime_motifs[*idx];
            let Some(step) = hidden_trajectory_step(&motif.motif_id) else {
                continue;
            };
            let distance = step.abs_diff(self.current_step);
            if distance < best_distance {
                best_distance = distance;
                best_step = Some(step);
            }
        }
        let best_step = best_step?;
        let scheduled = candidate_indices
            .iter()
            .copied()
            .filter(|idx| {
                hidden_trajectory_step(&self.runtime_motifs[*idx].motif_id) == Some(best_step)
            })
            .collect::<Vec<_>>();
        (!scheduled.is_empty()).then_some(scheduled)
    }

    pub(crate) fn gate34_target_kind(&self) -> &'static str {
        match self.gate34_target_source.as_str() {
            "motifs" => "motif",
            "specialists" => "specialist",
            "basins" => "basin",
            _ => "unknown",
        }
    }

    pub(crate) fn compute_user_prompt_vec(
        &mut self,
        model: &ModelWrapper,
        prompt_text: &str,
        device: &Device,
    ) -> Result<()> {
        self.prompt_vec = None;
        self.prompt_vec_norm = 0.0;
        self.prompt_similarity_unavailable = true;
        self.prompt_embedding_source = "unavailable:init".to_string();

        if self.gate34_target_source == "motifs" {
            let vec = task_anchor_signature(prompt_text);
            let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm <= 1e-8 {
                self.prompt_embedding_source = "unavailable:task_anchor_near_zero_norm".to_string();
                println!(
                    "[GATE34_PROMPT_EMBED] {}",
                    serde_json::json!({
                        "source": self.prompt_embedding_source,
                        "prompt_vec_norm": 0.0,
                        "prompt_similarity_unavailable": true,
                    })
                );
                return Ok(());
            }
            self.prompt_vec = Some(vec);
            self.prompt_vec_norm = norm;
            self.prompt_similarity_unavailable = false;
            self.prompt_embedding_source = "user_prompt_task_anchor_signature".to_string();
            println!(
                "[GATE34_PROMPT_EMBED] {}",
                serde_json::json!({
                    "source": self.prompt_embedding_source,
                    "prompt_vec_norm": self.prompt_vec_norm,
                    "prompt_similarity_unavailable": self.prompt_similarity_unavailable,
                })
            );
            return Ok(());
        }

        let encoding = match model.tokenizer().encode(prompt_text, true) {
            Ok(enc) => enc,
            Err(err) => {
                self.prompt_embedding_source = format!("unavailable:tokenize:{}", err);
                println!(
                    "[GATE34_PROMPT_EMBED] {}",
                    serde_json::json!({
                        "source": self.prompt_embedding_source,
                        "prompt_vec_norm": 0.0,
                        "prompt_similarity_unavailable": true,
                    })
                );
                return Ok(());
            }
        };

        let ids = encoding.get_ids().to_vec();
        if ids.is_empty() {
            self.prompt_embedding_source = "unavailable:empty_prompt_tokens".to_string();
            println!(
                "[GATE34_PROMPT_EMBED] {}",
                serde_json::json!({
                    "source": self.prompt_embedding_source,
                    "prompt_vec_norm": 0.0,
                    "prompt_similarity_unavailable": true,
                })
            );
            return Ok(());
        }

        let token_tensor = Tensor::from_vec(ids, (encoding.len(),), device)?;
        let emb = match model.embed_tokens_forward(&token_tensor) {
            Ok(t) => t.to_dtype(DType::F32)?,
            Err(err) => {
                self.prompt_embedding_source = format!("unavailable:embed:{}", err);
                println!(
                    "[GATE34_PROMPT_EMBED] {}",
                    serde_json::json!({
                        "source": self.prompt_embedding_source,
                        "prompt_vec_norm": 0.0,
                        "prompt_similarity_unavailable": true,
                    })
                );
                return Ok(());
            }
        };

        let (_rows, _dim) = emb.dims2()?;
        let pooled = emb.mean(0)?.flatten_all()?;
        let pooled_vec = tensor_to_vec_f32(&pooled)
            .map_err(|e| anyhow::anyhow!("prompt_vec_extract_failed: {}", e))?;
        let mut vec64: Vec<f32> = pooled_vec.into_iter().take(64).collect();
        if vec64.is_empty() {
            self.prompt_embedding_source = "unavailable:pooled_empty".to_string();
            println!(
                "[GATE34_PROMPT_EMBED] {}",
                serde_json::json!({
                    "source": self.prompt_embedding_source,
                    "prompt_vec_norm": 0.0,
                    "prompt_similarity_unavailable": true,
                })
            );
            return Ok(());
        }

        let norm = vec64.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm <= 1e-8 {
            self.prompt_embedding_source = "unavailable:near_zero_norm".to_string();
            self.prompt_vec_norm = norm;
            println!(
                "[GATE34_PROMPT_EMBED] {}",
                serde_json::json!({
                    "source": self.prompt_embedding_source,
                    "prompt_vec_norm": self.prompt_vec_norm,
                    "prompt_similarity_unavailable": true,
                })
            );
            return Ok(());
        }

        for v in &mut vec64 {
            *v /= norm;
        }
        self.prompt_vec = Some(vec64);
        self.prompt_vec_norm = norm;
        self.prompt_similarity_unavailable = false;
        self.prompt_embedding_source = "user_prompt_token_mean_64".to_string();
        println!(
            "[GATE34_PROMPT_EMBED] {}",
            serde_json::json!({
                "source": self.prompt_embedding_source,
                "prompt_vec_norm": self.prompt_vec_norm,
                "prompt_similarity_unavailable": self.prompt_similarity_unavailable,
            })
        );

        Ok(())
    }

    pub(crate) fn find_topk_motif_candidates(
        &self,
        probe: &Tensor,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let probe_vec = tensor_to_vec_f32(probe)?;
        let probe_64: Vec<f64> = probe_vec.iter().take(64).map(|&x| x as f64).collect();
        let mut dists: Vec<(String, f64)> = self
            .runtime_motifs
            .iter()
            .filter(|motif| motif.routing_safety_score >= self.gate34_motif_routing_safety_floor)
            .filter(|motif| !motif.raw_signature.is_empty())
            .map(|motif| {
                let mut dist_sq: f64 = 0.0;
                for (p, g) in probe_64.iter().zip(motif.raw_signature.iter()) {
                    let diff = p - (*g as f64);
                    dist_sq += diff * diff;
                }
                (motif.motif_id.clone(), dist_sq.sqrt())
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k.max(1));
        Ok(dists)
    }

    pub(crate) fn find_topk_ghost_candidates(
        &self,
        probe: &Tensor,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        #[cfg(feature = "niodv4_bridge")]
        {
            if let Some(registry) = &self.ghost_registry {
                let probe_vec = tensor_to_vec_f32(probe)?;
                let probe_64: Vec<f64> = probe_vec.iter().take(64).map(|&x| x as f64).collect();
                let mut dists: Vec<(String, f64)> = registry
                    .candidate_basins()
                    .map(|basin| {
                        let mut dist_sq: f64 = 0.0;
                        for (p, g) in probe_64.iter().zip(basin.data.iter()) {
                            let diff = p - (*g as f64);
                            dist_sq += diff * diff;
                        }
                        (basin.id.clone(), dist_sq.sqrt())
                    })
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                dists.truncate(k.max(1));
                return Ok(dists);
            }
        }
        let _ = probe;
        let _ = k;
        Ok(Vec::new())
    }

    pub(crate) fn find_topk_specialist_candidates(
        &self,
        probe: &Tensor,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let probe_vec = tensor_to_vec_f32(probe)?;
        let probe_64: Vec<f64> = probe_vec.iter().take(64).map(|&x| x as f64).collect();
        let mut dists: Vec<(String, f64)> = self
            .runtime_recovery_ops
            .iter()
            .filter(|operator| !operator.raw_signature.is_empty())
            .map(|operator| {
                let mut dist_sq: f64 = 0.0;
                for (p, g) in probe_64.iter().zip(operator.raw_signature.iter()) {
                    let diff = p - (*g as f64);
                    dist_sq += diff * diff;
                }
                (operator.specialist_id.clone(), dist_sq.sqrt())
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k.max(1));
        Ok(dists)
    }

    pub(crate) fn is_restored_compact_promoted(motif: &RuntimeMotifField) -> bool {
        motif.motif_kind == "promoted"
            && (motif.source == "secret_sauce::motif_anchor"
                || motif.source == "secret_sauce::promoted_anchor")
            && motif.promotion_status == "restored_compact"
    }

    pub(crate) fn sentence_boundary_reached(&self, token_text: &str, token_id: u32) -> bool {
        token_text.ends_with('.')
            || token_text.ends_with('!')
            || token_text.ends_with('?')
            || token_id == 128001
            || token_text.contains('\n')
    }

    /// Print route telemetry as JSONL to stdout
    pub(crate) fn print_route_telemetry(&self, req_id: &str, prompt_hash: &str) {
        let telemetry = serde_json::json!({
            "bridge_enabled": self.bridge_enabled,
            "req_id": req_id,
            "prompt_hash": prompt_hash,
            "ghost_basins_loaded": self.ghost_basins_loaded,
            "nearest_ghost_id": self.last_nearest_ghost_id,
            "nearest_ghost_distance": self.last_nearest_ghost_distance,
            "second_nearest_ghost_distance": self.last_second_nearest_ghost_distance,
            "route_margin": self.last_route_margin,
            "projection_strategy": self.last_projection_strategy,
            "ghost_pull_delta_norm": self.last_ghost_pull_delta_norm,
            "intervention_applied": self.last_intervention_applied,
        });
        println!("{}", serde_json::to_string(&telemetry).unwrap());
    }

    /// Save route telemetry to a JSONL file
    pub(crate) fn save_route_telemetry(
        &self,
        req_id: &str,
        prompt_hash: &str,
        output_path: &str,
        token_records: &[TokenPhysics],
        telemetry_profile: TelemetryProfile,
        active_context_startup: Option<&ActiveContextRuntimeStartupTelemetryRecord>,
    ) {
        use std::io::Write;

        let file = std::fs::File::create(output_path);
        if file.is_err() {
            eprintln!(" [TELEMETRY] Failed to create file: {}", output_path);
            return;
        }
        let mut writer = std::io::BufWriter::new(file.unwrap());

        // Write header record with aggregate info
        let header = serde_json::json!({
            "bridge_enabled": self.bridge_enabled,
            "req_id": req_id,
            "prompt_hash": prompt_hash,
            "ghost_basins_loaded": self.ghost_basins_loaded,
            "nearest_ghost_id": self.last_nearest_ghost_id,
            "nearest_ghost_distance": self.last_nearest_ghost_distance,
            "second_nearest_ghost_distance": self.last_second_nearest_ghost_distance,
            "route_margin": self.last_route_margin,
            "projection_strategy": self.last_projection_strategy,
            "ghost_pull_delta_norm": self.last_ghost_pull_delta_norm,
            "intervention_applied": self.last_intervention_applied,
            "record_type": "header",
            "telemetry_profile": telemetry_profile.as_str(),
            "token_records": token_records.len(),
        });
        let _ = writeln!(writer, "{}", serde_json::to_string(&header).unwrap());

        if let Some(record) = active_context_startup {
            let _ = writeln!(writer, "{}", serde_json::to_string(record).unwrap());
        }

        // Write each token record
        for record in token_records {
            let profile_record = record.to_profile_value(telemetry_profile);
            let _ = writeln!(
                writer,
                "{}",
                serde_json::to_string(&profile_record).unwrap()
            );
        }

        let _ = writer.flush();
    }

    pub(crate) fn update_tda_shadow_monitor(&mut self, token_trace: &mut TokenPhysics) {
        if !self.tda_shadow_monitor_enabled {
            return;
        }
        let fresh_decision = self.tda_shadow_monitor.annotate(token_trace);
        token_trace.tda_shadow_breath_apply_enabled = self.tda_shadow_breath_apply;
        if self.stdout_debug() {
            if let Some(decision) = fresh_decision.as_ref() {
                println!(
                    "[TDA SHADOW] action={} breath={} h1_total={:.4} loop={:.3} margin={:.3} involution_residual={:.6}",
                    decision.action.as_str(),
                    decision.should_breathe,
                    decision.dimensions[1].total_persistence,
                    decision.signals.loop_pressure,
                    decision.signals.margin_collapse,
                    decision.involution.max_double_apply_residual_l2,
                );
            }
        }
    }

    pub(crate) fn current_sentence_mean(&self) -> Result<Option<Tensor>> {
        if self.current_sentence_embeddings.is_empty() {
            return Ok(None);
        }
        let count = self.current_sentence_embeddings.len();
        let dim = self.current_sentence_embeddings[0].dim(0)?;
        let stack = Tensor::cat(&self.current_sentence_embeddings, 0)?;
        let stack_reshaped = stack.reshape((count, dim))?;
        Ok(Some(stack_reshaped.mean(0)?.flatten_all()?))
    }

    pub(crate) fn clear_secret_sauce_priors(&mut self) {
        self.secret_sauce_hidden_prior = None;
        self.secret_sauce_sentence_prior = None;
        self.secret_sauce_momentum_prior = None;
        self.secret_sauce_version = None;
        self.secret_sauce_steps_remaining = 0;
    }

    pub(crate) fn current_task_anchor_similarity(&self, signature: &[f32]) -> f32 {
        self.last_controller_candidates
            .iter()
            .map(|candidate| candidate.task_anchor_similarity)
            .fold(0.0f32, f32::max)
            .max(
                self.runtime_motifs
                    .iter()
                    .map(|motif| cosine_similarity_slices(signature, &motif.raw_signature).max(0.0))
                    .fold(0.0f32, f32::max),
            )
    }

    pub(crate) fn update_task_anchor_similarity_snapshot(&mut self, label: &str) {
        let Some(signature) = self.current_task_anchor_signature.clone() else {
            return;
        };
        let similarity = self.current_task_anchor_similarity(&signature);
        match label {
            "start" => self.task_anchor_similarity_start = similarity,
            "hinge" => self.task_anchor_similarity_hinge = similarity,
            "24tok" => self.task_anchor_similarity_24tok = similarity,
            _ => {}
        }
        let reference = if self.task_anchor_similarity_24tok > 0.0 {
            self.task_anchor_similarity_24tok
        } else if self.task_anchor_similarity_hinge > 0.0 {
            self.task_anchor_similarity_hinge
        } else {
            similarity
        };
        self.task_anchor_drift = (self.task_anchor_similarity_start - reference).abs();
    }

    pub(crate) fn current_hinge_window_drift(&self) -> f32 {
        if self.task_anchor_similarity_start <= 0.0 {
            0.0
        } else {
            let current = if self.task_anchor_similarity_24tok > 0.0 {
                self.task_anchor_similarity_24tok
            } else if self.task_anchor_similarity_hinge > 0.0 {
                self.task_anchor_similarity_hinge
            } else {
                self.task_anchor_similarity_start
            };
            (self.task_anchor_similarity_start - current).abs()
        }
    }

    pub(crate) fn maybe_capture_hinge_window(
        &mut self,
        event: &str,
        structured_candidate_loss_reason: Option<String>,
    ) {
        if !self.restored_run_active {
            return;
        }
        let structured_context =
            self.current_turn_structure_bias >= STRUCTURED_REENTRY_PROMPT_THRESHOLD;
        let in_window = self.structured_resume_window_remaining > 0
            || self.hinge_window_records.len() < HINGE_WINDOW_MAX_RECORDS
            || matches!(event, "promotion_attempt" | "hinge_flip");
        if !structured_context || !in_window {
            return;
        }

        let live_count = self.runtime_motifs.len().max(1) as f32;
        let neutral_count = self
            .runtime_motifs
            .iter()
            .filter(|motif| motif.motif_role == "neutral")
            .count() as f32;
        let neutral_basin_occupancy = (neutral_count / live_count).clamp(0.0, 1.0);

        let routed_motif = self.last_routed_motif_id.as_ref().and_then(|id| {
            self.runtime_motifs
                .iter()
                .find(|motif| &motif.motif_id == id)
        });
        let basin_width = routed_motif
            .map(|motif| motif.radius_mean)
            .unwrap_or(self.last_live_motif_radius);
        let curvature_tension = routed_motif
            .map(|motif| motif.tension_anchor_strength)
            .unwrap_or(0.0);

        let structured_candidate = self.last_controller_candidates.iter().find(|candidate| {
            candidate.motif_role == "structured" || candidate.motif_role == "structured_candidate"
        });
        let selected = self.last_controller_candidates.first();
        let structured_candidate_separation = match (selected, structured_candidate) {
            (Some(selected), Some(candidate)) if selected.motif_id != candidate.motif_id => {
                (candidate.routing_score - selected.routing_score).abs()
            }
            _ => 0.0,
        };
        let task_anchor_similarity = self
            .current_task_anchor_signature
            .as_ref()
            .map(|signature| self.current_task_anchor_similarity(signature))
            .unwrap_or(0.0);

        let candidates = self
            .last_controller_candidates
            .iter()
            .take(ROUTING_CONTROLLER_TOP_K)
            .map(|candidate| HingeWindowCandidateSummary {
                motif_id: candidate.motif_id.clone(),
                motif_role: candidate.motif_role.clone(),
                promotion_status: candidate.promotion_status.clone(),
                distance: candidate.distance,
                routing_score: candidate.routing_score,
                task_anchor_similarity: candidate.task_anchor_similarity,
                topology_density: candidate.topology_density,
                sequential_gap_rate: candidate.sequential_gap_rate,
                tension_anchor_strength: candidate.tension_anchor_strength,
                tightness_score: candidate.tightness_score,
            })
            .collect::<Vec<_>>();

        self.hinge_window_records.push(HingeWindowTickRecord {
            step: self.current_step,
            event: event.to_string(),
            structured_streak: self.structured_streak,
            clamp_active: self.reentry_clamp_steps_remaining > 0,
            clamp_strength: self.reentry_clamp_strength,
            task_anchor_similarity,
            basin_width,
            curvature_tension,
            neutral_basin_occupancy,
            structured_candidate_separation,
            task_vector_drift: self.current_hinge_window_drift(),
            routed_motif_id: self.last_routed_motif_id.clone(),
            routed_motif_role: self.last_routed_motif_role.clone(),
            structured_candidate_loss_reason,
            candidates,
        });
        if self.hinge_window_records.len() > HINGE_WINDOW_MAX_RECORDS {
            let drain = self.hinge_window_records.len() - HINGE_WINDOW_MAX_RECORDS;
            self.hinge_window_records.drain(0..drain);
        }
    }

    pub(crate) fn reward_empathy(&mut self, delta: f32) {
        self.empathy_spike = (self.empathy_spike + delta).clamp(0.0, 2.0);
    }

    pub(crate) fn strongest_promoted_motif_vector(&self) -> Option<Tensor> {
        self.runtime_motifs
            .iter()
            .filter(|motif| motif.motif_kind == "promoted")
            .max_by(|a, b| {
                a.promotion_score
                    .partial_cmp(&b.promotion_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.member_count.cmp(&b.member_count))
            })
            .map(|motif| motif.vector.detach())
    }

    pub(crate) fn apply_restore_continuity_assist(&mut self, summary: &MotifCarryForwardSummary) {
        if summary.restored_promoted_count == 0 {
            return;
        }
        let tuning = continuity_mode_tuning(self.runtime_mode);

        let ratio_deficit =
            (MOTIF_CARRY_FORWARD_ASSIST_RATIO_THRESHOLD - summary.carry_forward_ratio).max(0.0);
        let sim_deficit =
            (MOTIF_CARRY_FORWARD_ASSIST_SIM_THRESHOLD - summary.mean_best_similarity).max(0.0);
        let deficit = ratio_deficit.max(sim_deficit);
        if deficit <= 0.0 {
            return;
        }

        let steps = ((MOTIF_RESTORE_BIAS_MIN_STEPS as f32
            + deficit * (MOTIF_RESTORE_BIAS_MAX_STEPS - MOTIF_RESTORE_BIAS_MIN_STEPS) as f32)
            * tuning.restore_steps_scale
            * self.continuity_support_scale)
            .round() as usize;
        self.motif_restore_bias_steps_remaining = self
            .motif_restore_bias_steps_remaining
            .max(steps.clamp(MOTIF_RESTORE_BIAS_MIN_STEPS, MOTIF_RESTORE_BIAS_MAX_STEPS));
        self.motif_restore_bias_strength = self.motif_restore_bias_strength.max(
            (((0.15 + deficit * 0.70) * tuning.restore_strength_scale)
                * self.continuity_support_scale)
                .clamp(0.15, 0.95),
        );
        self.secret_sauce_steps_remaining = self.secret_sauce_steps_remaining.max(
            ((SECRET_SAUCE_RESTORE_DECAY_STEPS + 4) as f32
                * tuning.restore_steps_scale
                * self.continuity_support_scale)
                .round() as usize,
        );

        if let Some(anchor) = self.strongest_promoted_motif_vector() {
            self.secret_sauce_sentence_prior = Some(anchor.detach());
        }

        println!(
            " [MOTIF_ASSIST] weak_carry ratio={:.3} sim={:.3} -> bias_steps={} bias_strength={:.2}",
            summary.carry_forward_ratio,
            summary.mean_best_similarity,
            self.motif_restore_bias_steps_remaining,
            self.motif_restore_bias_strength
        );
    }

    pub(crate) fn apply_prior_continuity_policy(&mut self, comparison: &MotifContinuityComparison) {
        match comparison.verdict.as_str() {
            "regressed" => {
                let tuning = continuity_mode_tuning(self.runtime_mode);
                let severity = (-comparison.carry_forward_delta)
                    .max(-comparison.mean_similarity_delta)
                    .clamp(0.0, 1.0);
                if severity <= 0.0 {
                    return;
                }

                let steps = ((MOTIF_REGRESSION_ASSIST_MIN_STEPS as f32
                    + severity
                        * (MOTIF_REGRESSION_ASSIST_MAX_STEPS - MOTIF_REGRESSION_ASSIST_MIN_STEPS)
                            as f32)
                    * tuning.regression_steps_scale
                    * self.continuity_support_scale)
                    .round() as usize;
                self.motif_regression_assist_steps_remaining = self
                    .motif_regression_assist_steps_remaining
                    .max(steps.clamp(
                        MOTIF_REGRESSION_ASSIST_MIN_STEPS,
                        MOTIF_REGRESSION_ASSIST_MAX_STEPS,
                    ));
                self.motif_regression_assist_strength = self.motif_regression_assist_strength.max(
                    (((0.20 + severity * 0.65) * tuning.regression_strength_scale)
                        * self.continuity_support_scale)
                        .clamp(0.20, 0.95),
                );
                self.motif_restore_bias_steps_remaining =
                    self.motif_restore_bias_steps_remaining.max(
                        ((MOTIF_RESTORE_BIAS_MIN_STEPS as f32 + severity * 48.0)
                            * tuning.restore_steps_scale
                            * self.continuity_support_scale)
                            .round() as usize,
                    );
                self.motif_restore_bias_strength = self.motif_restore_bias_strength.max(
                    (((0.20 + severity * 0.55) * tuning.restore_strength_scale)
                        * self.continuity_support_scale)
                        .clamp(0.20, 0.90),
                );
                self.secret_sauce_steps_remaining = self.secret_sauce_steps_remaining.max(
                    ((SECRET_SAUCE_RESTORE_DECAY_STEPS + 8) as f32
                        * tuning.regression_steps_scale
                        * self.continuity_support_scale)
                        .round() as usize,
                );

                if let Some(anchor) = self.strongest_promoted_motif_vector() {
                    self.secret_sauce_sentence_prior = Some(anchor.detach());
                }

                println!(
                    " [MOTIF_POLICY] prior_regression carry_delta={:.3} sim_delta={:.3} -> assist_steps={} assist_strength={:.2}",
                    comparison.carry_forward_delta,
                    comparison.mean_similarity_delta,
                    self.motif_regression_assist_steps_remaining,
                    self.motif_regression_assist_strength
                );
            }
            "improved" => {
                let tuning = continuity_mode_tuning(self.runtime_mode);
                let confidence = comparison
                    .carry_forward_delta
                    .max(comparison.mean_similarity_delta)
                    .clamp(0.0, 1.0);
                let trim = ((0.20 + confidence * 0.45)
                    * tuning.stable_release_scale
                    * self.continuity_release_scale)
                    .clamp(0.20, 0.85);
                self.motif_regression_assist_steps_remaining = 0;
                self.motif_regression_assist_strength = 0.0;
                self.motif_restore_bias_steps_remaining =
                    ((self.motif_restore_bias_steps_remaining as f32) * (1.0 - trim)).round()
                        as usize;
                self.motif_restore_bias_strength =
                    (self.motif_restore_bias_strength * (1.0 - trim)).clamp(0.0, 1.0);
                self.secret_sauce_steps_remaining = self
                    .secret_sauce_steps_remaining
                    .min(SECRET_SAUCE_RESTORE_DECAY_STEPS);
                println!(
                    " [MOTIF_POLICY] prior_improved carry_delta={:.3} sim_delta={:.3} -> trim={:.2}",
                    comparison.carry_forward_delta,
                    comparison.mean_similarity_delta,
                    trim
                );
            }
            "stable" => {
                let tuning = continuity_mode_tuning(self.runtime_mode);
                self.motif_regression_assist_steps_remaining = 0;
                self.motif_regression_assist_strength = 0.0;
                self.motif_restore_bias_steps_remaining = ((self.motif_restore_bias_steps_remaining
                    as f32)
                    * (1.0 - 0.15 * tuning.stable_release_scale * self.continuity_release_scale))
                    .round() as usize;
                self.motif_restore_bias_strength = (self.motif_restore_bias_strength
                    * (1.0 - 0.10 * tuning.stable_release_scale * self.continuity_release_scale))
                    .clamp(0.0, 1.0);
                self.secret_sauce_steps_remaining = self.secret_sauce_steps_remaining.min(
                    ((SECRET_SAUCE_RESTORE_DECAY_STEPS + 2) as f32
                        * (2.0
                            - (tuning.stable_release_scale * self.continuity_release_scale)
                                .clamp(0.5, 1.5)))
                    .round() as usize,
                );
                println!(
                    " [MOTIF_POLICY] prior_stable carry_delta={:.3} sim_delta={:.3} -> easing restore bias",
                    comparison.carry_forward_delta,
                    comparison.mean_similarity_delta
                );
            }
            _ => {}
        }
    }

    pub(crate) fn maybe_apply_hidden_request(
        &mut self,
        signal: Option<&HiddenRequestSignal>,
        current_token: usize,
    ) {
        if !self.hidden_request_inference || !self.runtime_mode.is_agency() {
            self.hidden_request_candidate = None;
            self.hidden_request_streak = 0;
            self.last_hidden_request_pressure = 0.0;
            return;
        }

        let Some(signal) = signal else {
            self.hidden_request_candidate = None;
            self.hidden_request_streak = 0;
            self.last_hidden_request_pressure = 0.0;
            return;
        };

        if current_token < 24 {
            self.hidden_request_candidate = None;
            self.hidden_request_streak = 0;
            self.last_hidden_request_pressure = 0.0;
            return;
        }

        self.last_hidden_request_pressure = signal.score;

        if self.hidden_request_candidate == Some(signal.request_type) {
            self.hidden_request_streak += 1;
        } else {
            self.hidden_request_candidate = Some(signal.request_type);
            self.hidden_request_streak = 1;
        }

        let strong_single = signal.score >= 0.09;
        let sustained = self.hidden_request_streak >= 2 && signal.score >= 0.04;
        if !(strong_single || sustained) {
            return;
        }

        let (applied, focus_gate_msg) = self.apply_request(signal.request_type, current_token);
        println!(
            "[HIDDEN_REQUEST] type={} score={:.4} mass={:.4} peak_logit={:.3} best_rank={} activated={} surface=\"{}\"",
            signal.request_type.as_str(),
            signal.score,
            signal.blocked_mass,
            signal.peak_logit,
            signal
                .best_rank
                .map(|rank| rank.to_string())
                .unwrap_or_else(|| "-".to_string()),
            applied,
            signal.peak_surface.replace('\n', "\\n")
        );
        emit_ui_event_value(
            self.ui_events_json,
            "hidden_request",
            serde_json::json!({
                "step": current_token,
                "request_type": signal.request_type.as_str(),
                "score": signal.score,
                "blocked_mass": signal.blocked_mass,
                "peak_logit": signal.peak_logit,
                "best_rank": signal.best_rank,
                "applied": applied,
                "peak_surface": signal.peak_surface,
            }),
        );
        if let Some(msg) = focus_gate_msg {
            println!("🚫 [HIDDEN_REQUEST FOCUS GATE] Injecting: {}", msg);
            emit_ui_event_value(
                self.ui_events_json,
                "hidden_request_focus_gate",
                serde_json::json!({
                    "step": current_token,
                    "request_type": signal.request_type.as_str(),
                    "message": msg.clone(),
                }),
            );
            self.pending_insight = Some(msg);
        }
        if applied {
            self.last_hidden_request = Some(signal.request_type);
            self.hidden_request_activations += 1;
            self.hidden_request_candidate = None;
            self.hidden_request_streak = 0;
        }
    }

    /// Apply a model-requested physics change (Phase 4).
    /// Visible request tags are treated as model-authored control surfaces.
    pub fn apply_request(
        &mut self,
        req: RequestType,
        current_token: usize,
    ) -> (bool, Option<String>) {
        const MAX_REQUESTS: usize = 5; // Increased for Focus Gate redirects
        const COOLDOWN: usize = 15; // Slightly reduced
        const FOCUS_GRAVITY_SCALE: f32 = 1.35;

        // Anti-spam check
        if self.request_count >= MAX_REQUESTS {
            if self.stdout_debug() {
                println!(
                    "[AUTONOMIC: BLOCKED] Max requests ({}) reached",
                    MAX_REQUESTS
                );
            }
            return (false, None);
        }
        if self.request_count > 0 && current_token < self.last_request_token + COOLDOWN {
            if self.stdout_debug() {
                println!(
                    "[AUTONOMIC: BLOCKED] Cooldown active ({} tokens left)",
                    self.last_request_token + COOLDOWN - current_token
                );
            }
            return (false, None);
        }

        // Apply the request
        match req {
            RequestType::Spike => {
                if self.stdout_debug() {
                    println!("🧠 [AUTONOMIC: SPIKE] Model requested adrenaline burst!");
                }
                self.focus_lock_remaining_ticks = 0; // cancel focus lock
                self.adrenaline = 5.0;
                self.physics_blend = 6.5;
                self.dynamic_repulsion = -3.0;
            }
            RequestType::Focus => {
                if self.stdout_debug() {
                    println!(
                        "🧠 [AUTONOMIC: FOCUS] Model requested focus lock ({} ticks)",
                        self.focus_lock_max_ticks
                    );
                }
                self.focus_lock_remaining_ticks = self.focus_lock_max_ticks;
                self.physics_blend = 0.5;
                self.dynamic_repulsion = 0.0;
                let base_focus_gravity = self.dynamic_gravity.max(self.heartbeat_gravity);
                self.dynamic_gravity =
                    (base_focus_gravity * FOCUS_GRAVITY_SCALE).clamp(self.heartbeat_gravity, 4.0);
                self.adrenaline = 0.0;
            }
            RequestType::Explore => {
                if self.stdout_debug() {
                    println!("🧠 [AUTONOMIC: EXPLORE] Model brainstorming!");
                }
                self.focus_lock_remaining_ticks = 0; // cancel focus lock
                self.physics_blend = 2.0;
                self.dynamic_repulsion = -2.0;
                self.adrenaline = 3.0;
            }
            RequestType::Reset => {
                if self.stdout_debug() {
                    println!("🧠 [AUTONOMIC: RESET] Model clearing state!");
                }
                self.focus_lock_remaining_ticks = 0; // cancel focus lock
                self.adrenaline = 0.0;
                self.physics_blend = 1.5;
                self.dynamic_repulsion = -0.5;
                self.insight_persistence = 0;
                self.pending_insight = None;
            }
            RequestType::Remember => {
                if self.stdout_debug() {
                    println!("🧠 [AUTONOMIC: REMEMBER] Model fetching semantic memory!");
                }
                self.focus_lock_remaining_ticks = 0;
                self.physics_blend = 1.0;
                self.dynamic_repulsion = 0.0;
                self.adrenaline = 1.0;
            }
        }

        self.request_count += 1;
        self.last_request_token = current_token;
        (true, None)
    }

    pub(crate) fn reset_force_telemetry(
        &mut self,
        activation_gate: f32,
        engine_status: ForceEngineStatus,
    ) {
        self.last_gravity_mag = 0.0;
        self.last_applied_ghost_mag = 0.0;
        self.last_applied_ghost_vector = None;
        self.last_goal_mag = 0.0;
        self.last_repulsion_mag = 0.0;
        self.last_motif_mag = 0.0;
        self.last_bridge_force_selection = self.bridge_force_selection.as_str().to_string();
        self.last_bridge_force_selected_count = 0;
        self.last_bridge_force_selected_ids.clear();
        self.last_bridge_force_selection_source = "none".to_string();
        self.last_bridge_force_selected_score_max = None;
        self.last_bridge_force_selected_role = None;
        self.last_bridge_force_second_score = None;
        self.last_bridge_force_selected_margin = None;
        self.last_bridge_force_role_filter = self.bridge_force_role_filter.as_str().to_string();
        self.last_bridge_force_min_margin = self.bridge_force_min_margin;
        self.last_recovery_mag = 0.0;
        self.last_absence_signal = 0.0;
        self.last_trap_score = 0.0;
        self.active_recovery_specialist_id = None;
        self.active_recovery_weight = 0.0;
        self.last_live_motif_count = 0;
        self.last_live_motif_distance = 0.0;
        self.last_live_motif_radius = 0.0;
        self.last_live_basin_pressure = 0.0;
        self.last_activation_gate = activation_gate;
        self.last_forces_applied = false;
        self.last_engine_status = engine_status;
        self.last_vq_code_assigned = None;
        self.last_vq_encode_error = 0.0;
        self.last_correction_delta_norm = 0.0;
        self.last_specialist_activated = false;
        self.last_specialist_force_applied = false;
        self.last_specialist_force_norm = 0.0;
        self.last_correction_packet_vq_code = None;
        self.last_correction_packet_fire_count = 0;
        self.last_correction_packet_live_minted_fired_count = 0;
        self.last_correction_packet_effectiveness_avg = 0.0;
        self.last_correction_packet_effective_decay_rate = self.correction_packet_decay_rate;
        self.last_correction_packet_residual_applied = 0;
        self.last_correction_packet_force_norm = 0.0;
        self.last_correction_packet_ids.clear();
        self.last_packet_authority_score = 0.0;
        self.last_packet_authority_allowed = false;
        self.last_packet_authority_reason = "not_evaluated".to_string();
        self.last_packet_authority_blocked_reason = "not_evaluated".to_string();
        self.last_correction_packet_effective_pull_avg = 0.0;
        self.last_correction_packet_unfold_active = false;
        self.last_correction_packet_vq_encode_error = 0.0;
        self.last_correction_packet_unfold_factor_applied = 1.0;
        self.last_correction_packet_competence_factor = 1.0;
        self.last_correction_packet_arbitration_mode = if self.correction_packet_arbitration_mode
            == CorrectionPacketArbitrationMode::Disabled
        {
            "disabled".to_string()
        } else {
            "no_packet".to_string()
        };
        self.last_correction_packet_arbitration_reason = "not_evaluated".to_string();
        self.last_correction_packet_arbitration_candidate_count = 0;
        self.last_correction_packet_arbitration_min_target_distance = f32::INFINITY;
        self.last_correction_packet_arbitration_force_norm_estimate = 0.0;
        self.reset_specialist_worker_shadow_telemetry();
    }

    pub(crate) fn record_correction_packet_arbitration(
        &mut self,
        choice: CorrectionPacketArbitrationChoice,
        reason: &str,
        candidate_count: usize,
        min_target_distance: f32,
        force_norm_estimate: f32,
    ) {
        if self.correction_packet_arbitration_mode == CorrectionPacketArbitrationMode::Disabled {
            return;
        }
        self.last_correction_packet_arbitration_mode = choice.as_str().to_string();
        self.last_correction_packet_arbitration_reason = reason.to_string();
        self.last_correction_packet_arbitration_candidate_count = candidate_count;
        self.last_correction_packet_arbitration_min_target_distance = min_target_distance;
        self.last_correction_packet_arbitration_force_norm_estimate = force_norm_estimate;
    }

    /// §10bd Track 2 v11: reset trajectory-routing state at the start
    /// of each turn so classification is per-turn, not session-wide.
    pub(crate) fn reset_trajectory_routing_state(&mut self) {
        self.trajectory_fire_count_sum = 0.0;
        self.trajectory_fire_count_samples = 0;
        self.trajectory_classified = None;
        self.trajectory_turn_step = 0;
        self.trajectory_last_classified_step = usize::MAX;
        self.trajectory_pending_step_fires = 0;
        // RC3: clear the sliding window and re-classification bookkeeping so each turn
        // re-classifies from scratch.
        self.trajectory_window.clear();
        self.trajectory_last_reclassify_step = 0;
        self.trajectory_reclassify_count = 0;
        self.last_trajectory_window_mean = 0.0;
        // §10bt: clear prev-step ghost cache so the §10bs gate starts
        // each turn fresh and only suppresses after bridge has fired
        // at least once in this turn.
        self.prev_step_max_ghost_mag = 0.0;
    }

    /// §10bf prompt-level codec activation gate. Called once per turn
    /// from `run_assistant_turn` after `current_prompt_hash` is set.
    /// When `codec_active_prompt_substrings` is empty, gate is not
    /// configured and `codec_active_for_current_prompt` is true (legacy).
    /// Otherwise scans the prompt for any active substring (case-
    /// insensitive); sets the per-turn flag accordingly.
    pub(crate) fn apply_codec_prompt_gate(&mut self, prompt: &str) {
        if self.codec_active_prompt_substrings.is_empty() {
            self.codec_active_for_current_prompt = true;
            return;
        }
        let prompt_lc = prompt.to_lowercase();
        self.codec_active_for_current_prompt = self
            .codec_active_prompt_substrings
            .iter()
            .any(|s| prompt_lc.contains(s.as_str()));
    }

    /// §10ck per-prompt → top-K gate. First-match-wins over the
    /// configured map; sets `current_prompt_top_k_override` for the
    /// turn. Empty map = no override (preserves legacy fire_top_k).
    pub(crate) fn apply_correction_packet_prompt_top_k_gate(&mut self, prompt: &str) {
        let match_pair = resolve_correction_packet_prompt_top_k_match(
            prompt,
            &self.correction_packet_prompt_top_k_map,
        );
        self.current_prompt_top_k_override = match_pair.map(|(_, k)| *k);
        self.current_prompt_top_k_match_substring = match_pair.map(|(s, _)| s.clone());
        self.current_prompt_source_target_override =
            resolve_correction_packet_prompt_source_target_override(
                prompt,
                &self.correction_packet_prompt_source_target_map,
            );
        self.last_correction_packet_effective_fire_top_k = self
            .current_prompt_top_k_override
            .unwrap_or(self.correction_packet_fire_top_k);
        // §10cn: out-of-distribution suppression. Only active when the
        // map is configured AND the suppress flag is on AND the prompt
        // matched no rule. Empty map → flag is a no-op (legacy).
        self.correction_packet_suppress_for_current_prompt =
            should_suppress_correction_packets_for_prompt(
                self.correction_packet_suppress_when_no_prompt_match,
                !self.correction_packet_prompt_top_k_map.is_empty(),
                match_pair.is_some(),
            );
    }

    pub(crate) fn reset_specialist_worker_shadow_telemetry(&mut self) {
        self.last_specialist_worker_enabled = self.specialist_memory_workers_mode.is_enabled()
            && !self.specialist_memory_workers.is_empty();
        self.last_specialist_worker_selected_id = None;
        self.last_specialist_worker_packet_id = None;
        self.last_specialist_worker_unicode_escape = None;
        self.last_specialist_worker_original_route_id = None;
        self.last_specialist_worker_decoded_route_id = None;
        self.last_specialist_worker_route_preserved = None;
        self.last_specialist_worker_topk_hit = None;
        self.last_specialist_worker_score = None;
        self.last_specialist_worker_source_prompt_id = None;
        self.last_specialist_worker_direction_source = None;
        self.last_specialist_worker_delta_norm_64d = None;
        self.last_specialist_worker_hidden_delta_norm = None;
        self.last_specialist_worker_influence_clamp = None;
        self.last_specialist_worker_influence_scale = None;
        self.last_specialist_worker_probe_signature_64d = None;
        self.last_specialist_worker_target_signature_64d = None;
    }

    pub(crate) fn update_specialist_worker_shadow(
        &mut self,
        probe_normalized: &Tensor,
    ) -> Result<Option<usize>> {
        self.reset_specialist_worker_shadow_telemetry();
        if !self.specialist_memory_workers_mode.is_enabled()
            || self.specialist_memory_workers.is_empty()
        {
            return Ok(None);
        }

        let probe_vec = tensor_to_vec_f32(probe_normalized)?;
        let probe_64: Vec<f32> = probe_vec.iter().take(64).copied().collect();
        if probe_64.len() < 8 {
            return Ok(None);
        }

        let mut scored: Vec<(f32, usize)> = self
            .specialist_memory_workers
            .iter()
            .enumerate()
            .map(|(idx, worker)| {
                let n = probe_64.len().min(worker.raw_signature.len());
                let dist_sq = (0..n)
                    .map(|i| {
                        let diff = probe_64[i] - worker.raw_signature[i];
                        diff * diff
                    })
                    .sum::<f32>();
                (dist_sq.sqrt(), idx)
            })
            .filter(|(dist, _)| dist.is_finite())
            .collect();
        if scored.is_empty() {
            return Ok(None);
        }

        scored.sort_by(|(left, _), (right, _)| left.total_cmp(right));
        let (distance, selected_idx) =
            if let Some(fixed_packet_id) = &self.specialist_memory_worker_fixed_packet_id {
                match scored.iter().find(|(_, idx)| {
                    self.specialist_memory_workers[*idx].packet_id == *fixed_packet_id
                }) {
                    Some((distance, idx)) => (*distance, *idx),
                    None => return Ok(None),
                }
            } else {
                scored[0]
            };
        let worker = &self.specialist_memory_workers[selected_idx];
        let route_match = match (&self.last_routed_motif_id, worker.route_preserved) {
            (Some(route_id), _) => {
                route_id == &worker.original_route_id || route_id == &worker.decoded_route_id
            }
            (None, preserved) => preserved,
        };
        let top_k = self.specialist_memory_worker_top_k.max(1);
        let topk_hit = scored.iter().take(top_k).any(|(_, idx)| {
            let candidate = &self.specialist_memory_workers[*idx];
            if let Some(route_id) = &self.last_routed_motif_id {
                route_id == &candidate.original_route_id || route_id == &candidate.decoded_route_id
            } else {
                candidate.topk_hit
            }
        });

        self.last_specialist_worker_selected_id = Some(worker.worker_id.clone());
        self.last_specialist_worker_packet_id = Some(worker.packet_id.clone());
        self.last_specialist_worker_unicode_escape = Some(worker.unicode_escape.clone());
        self.last_specialist_worker_original_route_id = Some(worker.original_route_id.clone());
        self.last_specialist_worker_decoded_route_id = Some(worker.decoded_route_id.clone());
        self.last_specialist_worker_route_preserved = Some(route_match);
        self.last_specialist_worker_topk_hit = Some(topk_hit);
        self.last_specialist_worker_score = Some((1.0 / (1.0 + distance)).max(worker.worker_score));
        self.last_specialist_worker_source_prompt_id = Some(worker.source_prompt_id.clone());
        let signature_len = probe_64.len().min(worker.raw_signature.len()).min(64);
        self.last_specialist_worker_direction_source = Some(
            self.specialist_memory_worker_influence_direction
                .telemetry_label()
                .to_string(),
        );
        self.last_specialist_worker_delta_norm_64d = Some(distance);
        self.last_specialist_worker_influence_clamp = Some(
            self.specialist_memory_worker_influence_clamp
                .clamp(0.0, 0.03),
        );
        self.last_specialist_worker_probe_signature_64d =
            Some(probe_64.iter().take(signature_len).copied().collect());
        self.last_specialist_worker_target_signature_64d = Some(
            worker
                .raw_signature
                .iter()
                .take(signature_len)
                .copied()
                .collect(),
        );
        Ok(Some(selected_idx))
    }

    pub(crate) fn apply_specialist_worker_influence(
        &mut self,
        probe: &Tensor,
        selected_idx: Option<usize>,
        probe_force: Tensor,
        layer_idx: usize,
    ) -> Result<Tensor> {
        if self.specialist_memory_workers_mode != SpecialistMemoryWorkerMode::Influence {
            return Ok(probe_force);
        }
        if let Some((lo, hi)) = self.specialist_memory_worker_influence_layers {
            if layer_idx < lo || layer_idx > hi {
                if self.last_projection_strategy.as_str() != "route_memory_worker_influence" {
                    self.last_projection_strategy =
                        "route_memory_worker_skip:layer_out_of_influence_band".to_string();
                    self.last_ghost_pull_delta_norm = 0.0;
                    self.last_intervention_applied = false;
                    self.last_recovery_mag = 0.0;
                    self.last_specialist_worker_influence_scale = None;
                }
                return Ok(probe_force);
            }
        }

        match &self.specialist_memory_worker_influence_scope {
            SpecialistMemoryWorkerInfluenceScope::AnswerWindow
                if !self.specialist_memory_worker_answer_window_active =>
            {
                self.last_projection_strategy =
                    "route_memory_worker_skip:answer_window_inactive".to_string();
                self.last_ghost_pull_delta_norm = 0.0;
                self.last_intervention_applied = false;
                self.last_recovery_mag = 0.0;
                self.last_specialist_worker_influence_scale = None;
                return Ok(probe_force);
            }
            SpecialistMemoryWorkerInfluenceScope::PreAnswer
                if !self.specialist_memory_worker_pre_answer_active =>
            {
                self.last_projection_strategy =
                    "route_memory_worker_skip:pre_answer_inactive".to_string();
                self.last_ghost_pull_delta_norm = 0.0;
                self.last_intervention_applied = false;
                self.last_recovery_mag = 0.0;
                self.last_specialist_worker_influence_scale = None;
                return Ok(probe_force);
            }
            SpecialistMemoryWorkerInfluenceScope::PreEarned
                if !self.specialist_memory_worker_pre_earned_active =>
            {
                self.last_projection_strategy =
                    "route_memory_worker_skip:pre_earned_inactive".to_string();
                self.last_ghost_pull_delta_norm = 0.0;
                self.last_intervention_applied = false;
                self.last_recovery_mag = 0.0;
                self.last_specialist_worker_influence_scale = None;
                return Ok(probe_force);
            }
            SpecialistMemoryWorkerInfluenceScope::AtAnswerBoundary
                if !self.specialist_memory_worker_at_boundary_active =>
            {
                self.last_projection_strategy =
                    "route_memory_worker_skip:at_boundary_inactive".to_string();
                self.last_ghost_pull_delta_norm = 0.0;
                self.last_intervention_applied = false;
                self.last_recovery_mag = 0.0;
                self.last_specialist_worker_influence_scale = None;
                return Ok(probe_force);
            }
            SpecialistMemoryWorkerInfluenceScope::TokenRange {
                start_token_1_based,
                end_token_1_based,
            } => {
                let ordinal = self.current_step.saturating_add(1);
                if ordinal < *start_token_1_based || ordinal > *end_token_1_based {
                    self.last_projection_strategy =
                        "route_memory_worker_skip:token_range_out_of_band".to_string();
                    self.last_ghost_pull_delta_norm = 0.0;
                    self.last_intervention_applied = false;
                    self.last_recovery_mag = 0.0;
                    self.last_specialist_worker_influence_scale = None;
                    return Ok(probe_force);
                }
            }
            _ => {}
        }
        let Some(selected_idx) = selected_idx else {
            self.last_projection_strategy = "route_memory_worker_skip:no_selection".to_string();
            self.last_ghost_pull_delta_norm = 0.0;
            self.last_intervention_applied = false;
            self.last_recovery_mag = 0.0;
            return Ok(probe_force);
        };
        let max_norm = self
            .specialist_memory_worker_influence_clamp
            .clamp(0.0, 0.03);
        if max_norm <= 1e-6 {
            self.last_projection_strategy = "route_memory_worker_skip:clamp_zero".to_string();
            self.last_ghost_pull_delta_norm = 0.0;
            self.last_intervention_applied = false;
            self.last_recovery_mag = 0.0;
            return Ok(probe_force);
        }
        let signed_direction = self
            .specialist_memory_worker_influence_sign
            .clamp(-1.0, 1.0);
        if signed_direction.abs() <= 1e-6 {
            self.last_projection_strategy = "route_memory_worker_skip:sign_zero".to_string();
            self.last_ghost_pull_delta_norm = 0.0;
            self.last_intervention_applied = false;
            self.last_recovery_mag = 0.0;
            return Ok(probe_force);
        }

        let device = probe.device();
        let direction = match self.specialist_memory_worker_influence_direction {
            SpecialistMemoryWorkerInfluenceDirection::Target => {
                let target = self.specialist_memory_workers[selected_idx]
                    .vector
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_device(device)?;
                if target.dims() != probe.dims() {
                    self.last_projection_strategy =
                        "route_memory_worker_skip:dim_mismatch".to_string();
                    self.last_ghost_pull_delta_norm = 0.0;
                    self.last_intervention_applied = false;
                    self.last_recovery_mag = 0.0;
                    return Ok(probe_force);
                }
                (target - probe.clone())?
            }
            SpecialistMemoryWorkerInfluenceDirection::Residual64 => {
                let probe_vec = tensor_to_vec_f32(probe)?;
                let worker = &self.specialist_memory_workers[selected_idx];
                let signature_len = probe_vec.len().min(worker.raw_signature.len()).min(64);
                if signature_len < 8 {
                    self.last_projection_strategy =
                        "route_memory_worker_skip:residual64_too_short".to_string();
                    self.last_ghost_pull_delta_norm = 0.0;
                    self.last_intervention_applied = false;
                    self.last_recovery_mag = 0.0;
                    return Ok(probe_force);
                }
                let mut residual_64: Vec<f32> = (0..signature_len)
                    .map(|idx| worker.raw_signature[idx] - probe_vec[idx])
                    .collect();
                normalize(&mut residual_64);
                project_bridge_vector_to_hidden(&residual_64, self.hidden_dim, device)?
            }
            SpecialistMemoryWorkerInfluenceDirection::Delta64 => {
                let worker = &self.specialist_memory_workers[selected_idx];
                let signature_len = worker.raw_signature.len().min(64);
                if signature_len < 8 {
                    self.last_projection_strategy =
                        "route_memory_worker_skip:delta64_too_short".to_string();
                    self.last_ghost_pull_delta_norm = 0.0;
                    self.last_intervention_applied = false;
                    self.last_recovery_mag = 0.0;
                    return Ok(probe_force);
                }
                let mut delta_64 = worker
                    .raw_signature
                    .iter()
                    .take(signature_len)
                    .copied()
                    .collect::<Vec<_>>();
                normalize(&mut delta_64);
                project_bridge_vector_to_hidden(&delta_64, self.hidden_dim, device)?
            }
        };
        if direction.dims() != probe.dims() {
            self.last_projection_strategy = "route_memory_worker_skip:dim_mismatch".to_string();
            self.last_ghost_pull_delta_norm = 0.0;
            self.last_intervention_applied = false;
            self.last_recovery_mag = 0.0;
            return Ok(probe_force);
        }

        let raw_norm = direction.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        if raw_norm <= 1e-6 || !raw_norm.is_finite() {
            self.last_projection_strategy = "route_memory_worker_skip:zero_delta".to_string();
            self.last_ghost_pull_delta_norm = 0.0;
            self.last_intervention_applied = false;
            self.last_recovery_mag = 0.0;
            return Ok(probe_force);
        }

        let scale = if raw_norm > max_norm {
            max_norm / raw_norm
        } else {
            1.0
        };
        let signed_scale = scale * signed_direction;
        let scale_t = Tensor::from_vec(vec![signed_scale], (1,), device)?;
        let worker_delta = direction.broadcast_mul(&scale_t)?;
        let clamped_norm = (raw_norm * signed_scale.abs()).min(max_norm);
        self.last_ghost_pull_delta_norm = clamped_norm;
        self.last_intervention_applied = clamped_norm > 1e-6;
        self.last_projection_strategy = "route_memory_worker_influence".to_string();
        self.last_recovery_mag = clamped_norm;
        self.last_specialist_worker_hidden_delta_norm = Some(raw_norm);
        self.last_specialist_worker_influence_scale = Some(signed_scale);
        Ok((probe_force + worker_delta)?)
    }

    pub(crate) fn clear_ghost_pressure_telemetry(&mut self) {
        self.last_ghost_pre_norm = 0.0;
        self.last_ghost_gain = self.params.ghost_gravity as f32;
        self.last_wobble_pressure_crossing = false;
    }

    pub(crate) fn update_ghost_pressure_telemetry(&mut self, ghost_pre_norm: f32) {
        let previous = self.last_ghost_pre_norm;
        self.last_ghost_pre_norm = ghost_pre_norm;
        self.last_ghost_gain = self.params.ghost_gravity as f32;
        self.last_wobble_pressure_crossing = ghost_pre_norm >= NIODOO_WOBBLE_PRESSURE_THRESHOLD
            && previous < NIODOO_WOBBLE_PRESSURE_THRESHOLD;
    }

    pub(crate) fn update_bridge_telemetry(&mut self, probe_normalized: &Tensor) -> Result<()> {
        #[cfg(feature = "niodv4_bridge")]
        {
            if let Some(_registry) = &self.ghost_registry {
                let (nearest_id, min_dist, second_min_dist) =
                    self.find_nearest_ghost_info(probe_normalized)?;
                let prev_id = self.last_nearest_ghost_id.clone();

                self.last_nearest_ghost_id = nearest_id.clone();
                self.last_nearest_ghost_distance = min_dist as f32;
                self.last_second_nearest_ghost_distance = second_min_dist as f32;
                if second_min_dist < f64::MAX {
                    self.last_route_margin = (second_min_dist - min_dist) as f32;
                } else {
                    self.last_route_margin = 0.0;
                }
                self.last_bridge_route_probe_64d = tensor_to_vec_f32(probe_normalized)?
                    .into_iter()
                    .take(64)
                    .collect();

                // apply_steering runs once per layer per token; only tick counters once per token.
                let step = self.current_step as i64;
                if step != self.last_bridge_counter_step {
                    self.last_bridge_counter_step = step;
                    if nearest_id.is_some() && nearest_id == prev_id {
                        self.last_ghost_id_run_length =
                            self.last_ghost_id_run_length.saturating_add(1);
                    } else {
                        self.last_ghost_id_run_length = if nearest_id.is_some() { 1 } else { 0 };
                        if prev_id.is_some() && nearest_id != prev_id {
                            self.last_ghost_switch_cooldown_remaining =
                                self.bridge_cooldown_after_switch;
                        }
                    }
                }
            }
        }
        let _ = probe_normalized; // Avoid unused warning when feature is OFF
        Ok(())
    }

    pub(crate) fn find_nearest_ghost_info(
        &self,
        probe: &Tensor,
    ) -> Result<(Option<String>, f64, f64)> {
        #[cfg(feature = "niodv4_bridge")]
        {
            if let Some(registry) = &self.ghost_registry {
                let probe_vec = tensor_to_vec_f32(probe)?;
                // Establishing baseline: project via truncation to 64D
                let probe_64: Vec<f64> = probe_vec.iter().take(64).map(|&x| x as f64).collect();

                let mut min_dist = f64::MAX;
                let mut nearest_id = None;
                let mut second_min_dist = f64::MAX;

                for basin in registry.candidate_basins() {
                    let mut dist_sq: f64 = 0.0;
                    for (p, g) in probe_64.iter().zip(basin.data.iter()) {
                        let diff = p - (*g as f64);
                        dist_sq += diff * diff;
                    }
                    let dist = dist_sq.sqrt();
                    if dist < min_dist {
                        second_min_dist = min_dist;
                        min_dist = dist;
                        nearest_id = Some(basin.id.clone());
                    } else if dist < second_min_dist {
                        second_min_dist = dist;
                    }
                }
                return Ok((nearest_id, min_dist, second_min_dist));
            }
        }
        let _ = probe;
        Ok((None, 0.0, 0.0))
    }

    pub(crate) fn get_bridge_ghost_vector(
        &self,
        nearest_id: &str,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        #[cfg(feature = "niodv4_bridge")]
        {
            if let Some(registry) = &self.ghost_registry {
                if let Some(basin) = registry.find_basin(nearest_id) {
                    // Project 64D basin vector back to hidden dim
                    return Ok(Some(project_bridge_vector_to_hidden(
                        &basin.data,
                        self.hidden_dim,
                        device,
                    )?));
                }
            }
        }
        let _ = nearest_id;
        let _ = device;
        Ok(None)
    }

    pub(crate) fn normalized_alignment(
        &self,
        probe_normalized: &Tensor,
        target: &Tensor,
    ) -> candle_core::Result<f32> {
        Ok((probe_normalized * target)?.sum_all()?.to_scalar::<f32>()?)
    }

    pub(crate) fn normalized_topology_pressure(&self, operator: &RuntimeRecoveryOperator) -> f32 {
        let betti_0 = operator.betti_0.max(0.0);
        let betti_1 = operator.betti_1.max(0.0);
        let betti_0_log = betti_0.ln_1p();
        let betti_1_log = betti_1.ln_1p();
        let connectivity_term =
            (betti_0_log / (1.0 + betti_0_log + betti_1_log)).clamp(0.0, 1.0) * 0.12;
        let cycle_term = (betti_1_log / (1.0 + betti_0_log + betti_1_log)).clamp(0.0, 1.0) * 0.45;
        let flip_term = (operator.flip_rate * 2.5).clamp(0.0, 0.3);
        let energy_term = (operator.max_pre_energy * 1.25).clamp(0.0, 0.3);
        let tension_term = (operator.tension_point * 0.75).clamp(0.0, 0.25);
        let orbit_gap = if operator.orbit_count > 0.0 {
            (1.0 / (1.0 + operator.orbit_count)).clamp(0.0, 1.0) * 0.2
        } else {
            0.2
        };
        (connectivity_term + cycle_term + flip_term + energy_term + tension_term + orbit_gap)
            .clamp(0.0, 1.0)
    }

    pub(crate) fn synthesize_absence_signal(
        &self,
        operator: &RuntimeRecoveryOperator,
        live_motif_probe: &LiveMotifProbeStats,
    ) -> f32 {
        let stress_term = (self.stress_level / 15.0).clamp(0.0, 1.0);
        let boredom_term = (self.boredom_level / 4.0).clamp(0.0, 1.0) * 0.35;
        let topology_term = self.normalized_topology_pressure(operator);
        let live_trap_term = live_motif_probe.trap_pressure * 0.55;
        let live_fragmentation_term = live_motif_probe.fragmentation * 0.20;
        let live_radius_term = if live_motif_probe.nearest_radius > 0.0 {
            (1.0 / (1.0 + live_motif_probe.nearest_radius * 10.0)).clamp(0.0, 1.0) * 0.12
        } else {
            0.0
        };
        (operator.absence_signal * 0.7
            + stress_term * 0.35
            + boredom_term
            + topology_term
            + live_trap_term
            + live_fragmentation_term
            + live_radius_term)
            .clamp(0.0, 1.8)
    }

    pub(crate) fn compute_bridge_forces(
        &mut self,
        probe: &Tensor,
        probe_normalized: &Tensor,
        activation_gate: f32,
        layer_idx: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let device = probe.device();
        let mut motif_force = Tensor::zeros(probe.shape(), probe.dtype(), device)?;
        let mut recovery_force = Tensor::zeros(probe.shape(), probe.dtype(), device)?;

        self.last_motif_mag = 0.0;
        self.last_bridge_force_selection = self.bridge_force_selection.as_str().to_string();
        self.last_bridge_force_selected_count = 0;
        self.last_bridge_force_selected_ids.clear();
        self.last_bridge_force_selection_source = "none".to_string();
        self.last_bridge_force_selected_score_max = None;
        self.last_bridge_force_selected_role = None;
        self.last_bridge_force_second_score = None;
        self.last_bridge_force_selected_margin = None;
        self.last_bridge_force_role_filter = self.bridge_force_role_filter.as_str().to_string();
        self.last_bridge_force_min_margin = self.bridge_force_min_margin;
        self.last_recovery_mag = 0.0;
        self.last_absence_signal = 0.0;
        self.last_trap_score = 0.0;

        if layer_idx < self.physics_start_layer || layer_idx > self.physics_end_layer {
            return Ok((motif_force, recovery_force));
        }

        let live_motif_probe = self
            .live_motif_probe_stats(probe_normalized)
            .unwrap_or_default();
        self.last_live_motif_count = live_motif_probe.live_motif_count;
        self.last_live_motif_distance = live_motif_probe.nearest_distance;
        self.last_live_motif_radius = live_motif_probe.nearest_radius;
        self.last_live_basin_pressure = live_motif_probe.trap_pressure;
        let _ = self.update_specialist_worker_shadow(probe_normalized);

        if !self.runtime_motifs.is_empty() {
            let structured_context =
                self.current_turn_structure_bias >= STRUCTURED_REENTRY_PROMPT_THRESHOLD;
            let live_median_radius = self.live_median_radius();
            let routed_motif_id = self
                .active_routing_cache()
                .map(|cache| cache.motif_id.as_str());
            let has_non_compact_structured = self.runtime_motifs.iter().any(|motif| {
                motif.motif_role == "structured"
                    && motif.promotion_status != "restored_compact"
                    && motif.source != "secret_sauce::sentence_context"
            });
            let mut motif_scores = Vec::with_capacity(self.runtime_motifs.len());
            let mut motif_alignments = Vec::with_capacity(self.runtime_motifs.len());
            for motif in &self.runtime_motifs {
                let alignment = self.normalized_alignment(probe_normalized, &motif.vector)?;
                motif_alignments.push(alignment);
                let structural_drag = motif.max_pre_energy * 0.35 + motif.flip_rate * 1.75;
                let orbital_confidence =
                    (motif.orbit_count / (1.0 + motif.orbit_count)).clamp(0.0, 1.0);
                let structured_bonus = if structured_context && motif.motif_role == "structured" {
                    0.55 + motif.topology_density * 0.20 + motif.tightness_score * 0.25
                } else {
                    0.0
                };
                let conversational_penalty =
                    if structured_context && motif.motif_role == "conversational" {
                        0.35 + ((motif.radius_mean - live_median_radius).max(0.0) * 3.5)
                            .clamp(0.0, 0.35)
                    } else {
                        0.0
                    };
                let topology_penalty = motif.sequential_gap_rate * 0.22
                    + motif.conflict_ratio * 0.28
                    + motif.mixed_ratio * 0.08;
                let routed_bonus = if routed_motif_id == Some(motif.motif_id.as_str()) {
                    0.65 + motif.routing_safety_score * 0.20
                } else {
                    0.0
                };
                let restored_compact_penalty = if structured_context
                    && has_non_compact_structured
                    && motif.promotion_status == "restored_compact"
                {
                    0.45
                } else {
                    0.0
                };
                let (
                    structured_bonus,
                    conversational_penalty,
                    topology_penalty,
                    routed_bonus,
                    restored_compact_penalty,
                    routing_safety_bonus,
                ) = if self.ablate_conflict_routing {
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                } else {
                    (
                        structured_bonus,
                        conversational_penalty,
                        topology_penalty,
                        routed_bonus,
                        restored_compact_penalty,
                        motif.routing_safety_score * 0.18,
                    )
                };
                let raw_score = alignment * 4.5
                    + motif.persistence_score * 1.4
                    + motif.readiness_score
                    + motif.injection_strength * 0.5
                    + orbital_confidence * 0.5
                    + structured_bonus
                    + routed_bonus
                    + routing_safety_bonus
                    - structural_drag
                    - topology_penalty
                    - conversational_penalty
                    - restored_compact_penalty;
                motif_scores.push(raw_score);
            }

            let mut candidate_indices = self
                .runtime_motifs
                .iter()
                .enumerate()
                .filter(|(_, motif)| self.bridge_force_role_filter.accepts(&motif.motif_role))
                .map(|(idx, _)| idx)
                .collect::<Vec<_>>();
            let trajectory_schedule_source = if let Some(scheduled) =
                self.trajectory_scheduled_bridge_indices(&candidate_indices)
            {
                candidate_indices = scheduled;
                Some("trajectory_nearest_step")
            } else {
                None
            };
            let mut ranked_candidates = candidate_indices
                .iter()
                .copied()
                .filter(|idx| motif_scores[*idx].is_finite())
                .collect::<Vec<_>>();
            ranked_candidates.sort_by(|a, b| {
                motif_scores[*b]
                    .partial_cmp(&motif_scores[*a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let top_score = ranked_candidates.first().map(|idx| motif_scores[*idx]);
            let second_score = ranked_candidates.get(1).map(|idx| motif_scores[*idx]);
            let score_margin = match (top_score, second_score) {
                (Some(top), Some(second)) => Some(top - second),
                (Some(_), None) => None,
                _ => None,
            };
            let margin_passes = self.bridge_force_min_margin <= 0.0
                || score_margin
                    .map(|margin| margin >= self.bridge_force_min_margin)
                    .unwrap_or(false);

            let (selected_indices, selection_source): (Vec<usize>, &'static str) = if !margin_passes
            {
                (Vec::new(), "margin_gate")
            } else {
                match self.bridge_force_selection {
                    BridgeForceSelection::All => (
                        candidate_indices,
                        trajectory_schedule_source.unwrap_or("all"),
                    ),
                    BridgeForceSelection::Routed => {
                        if let Some(source) = trajectory_schedule_source {
                            match ranked_candidates.first().copied() {
                                Some(index) => (vec![index], source),
                                None => (Vec::new(), "none"),
                            }
                        } else {
                            let routed_index = self.active_routing_cache().and_then(|cache| {
                                self.runtime_motifs.iter().position(|motif| {
                                    motif.motif_id == cache.motif_id
                                        && self.bridge_force_role_filter.accepts(&motif.motif_role)
                                })
                            });
                            if let Some(index) = routed_index {
                                (vec![index], "routing_cache")
                            } else {
                                match ranked_candidates.first().copied() {
                                    Some(index) => (vec![index], "current_score"),
                                    None => (Vec::new(), "none"),
                                }
                            }
                        }
                    }
                }
            };
            let selected_scores = selected_indices
                .iter()
                .map(|idx| motif_scores[*idx])
                .collect::<Vec<_>>();
            let selected_weights = stable_softmax(&selected_scores);
            self.last_bridge_force_selection = self.bridge_force_selection.as_str().to_string();
            self.last_bridge_force_selected_count = selected_indices.len();
            self.last_bridge_force_selected_ids = selected_indices
                .iter()
                .take(8)
                .map(|idx| self.runtime_motifs[*idx].motif_id.clone())
                .collect();
            self.last_bridge_force_selection_source = selection_source.to_string();
            self.last_bridge_force_selected_score_max = selected_scores
                .iter()
                .copied()
                .filter(|score| score.is_finite())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.last_bridge_force_selected_role = selected_indices
                .first()
                .map(|idx| self.runtime_motifs[*idx].motif_role.clone());
            self.last_bridge_force_second_score = second_score;
            self.last_bridge_force_selected_margin = score_margin;
            self.last_bridge_force_role_filter = self.bridge_force_role_filter.as_str().to_string();
            self.last_bridge_force_min_margin = self.bridge_force_min_margin;

            for (idx, weight) in selected_indices.iter().zip(selected_weights.iter()) {
                let motif = &self.runtime_motifs[*idx];
                let alignment = motif_alignments[*idx];
                if *weight <= 1e-4 {
                    continue;
                }

                let direction = (&motif.vector - probe_normalized)?;
                let direction_norm = direction.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
                if direction_norm <= 1e-6 {
                    continue;
                }

                let affinity = ((alignment + 1.0) * 0.5).clamp(0.0, 1.0);
                let basin_tightness =
                    (1.0 / (1.0 + motif.radius_mean + motif.radius_std * 4.0)).clamp(0.15, 1.5);
                let force_scale = self.motif_force_scale
                    * activation_gate
                    * *weight
                    * (0.25 + affinity)
                    * motif.persistence_score.max(0.0)
                    * (0.25 + motif.readiness_score.max(0.0))
                    * if motif.motif_kind == "promoted"
                        && self.motif_restore_bias_steps_remaining > 0
                    {
                        (1.0 + self.motif_restore_bias_strength).clamp(1.0, 1.95)
                    } else {
                        1.0
                    }
                    * (1.0 + self.empathy_spike * 0.35).clamp(1.0, 1.7)
                    * basin_tightness;
                let scale_t = Tensor::new(force_scale / direction_norm.max(1e-6), device)?;
                motif_force = (motif_force + direction.broadcast_mul(&scale_t)?)?;
            }

            // =================================================================
            // TASK-ANCHOR CLAMPING: "Shut Up and Calculate"
            // When the routed motif is promoted/structured in a structured context,
            // apply a corrective pull when the probe drifts from the task geometry.
            // Uses the engine's current_task_anchor_signature directly (always available
            // in structured context) rather than relying on motif-attached anchors.
            // =================================================================
            let structured_context =
                self.current_turn_structure_bias >= STRUCTURED_REENTRY_PROMPT_THRESHOLD;
            if structured_context {
                // Only clamp when we have a task anchor and a routed structured motif
                if let Some(ref task_anchor) = self.current_task_anchor_signature {
                    if let Some(cache) = self.active_routing_cache() {
                        if let Some(routed) = self
                            .runtime_motifs
                            .iter()
                            .find(|m| m.motif_id == cache.motif_id)
                        {
                            let is_structured = routed.motif_kind == "promoted"
                                || routed.motif_role == "structured"
                                || routed.motif_role == "structured_candidate";

                            if is_structured {
                                if let Ok(probe_64d) =
                                    compress_hidden_state_to_64d(probe_normalized)
                                {
                                    let similarity = cosine_similarity_f32(&probe_64d, task_anchor);
                                    // Fire when probe drifts below 30% similarity to task geometry
                                    let clamp_threshold = 0.30;
                                    if similarity < clamp_threshold {
                                        let drift = (clamp_threshold - similarity).clamp(0.0, 1.0);
                                        // Strong linear scaling: max 4.0
                                        let clamp_strength = (drift
                                            * self.motif_force_scale
                                            * activation_gate
                                            * 4.0)
                                            .clamp(0.0, 4.0);
                                        if clamp_strength > 0.01 {
                                            let direction = (&routed.vector - probe_normalized)?;
                                            let direction_norm = direction
                                                .sqr()?
                                                .sum_all()?
                                                .sqrt()?
                                                .to_scalar::<f32>()?;
                                            if direction_norm > 1e-6 {
                                                let scale_t = Tensor::new(
                                                    clamp_strength / direction_norm,
                                                    device,
                                                )?;
                                                let clamp_force =
                                                    direction.broadcast_mul(&scale_t)?;
                                                motif_force = (motif_force + clamp_force)?;
                                                self.last_task_anchor_clamp =
                                                    Some((similarity, clamp_strength));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            self.last_motif_mag = motif_force
                .sqr()?
                .sum_all()?
                .sqrt()?
                .to_scalar::<f32>()
                .unwrap_or(0.0);
        }

        if !self.runtime_recovery_ops.is_empty() {
            let stress_term = (self.stress_level / 15.0).clamp(0.0, 1.0);
            let boredom_term = (self.boredom_level / 4.0).clamp(0.0, 1.0);
            let trap_score = (stress_term
                + boredom_term
                + self.last_motif_mag * 0.05
                + live_motif_probe.trap_pressure * 0.85
                + live_motif_probe.fragmentation * 0.20
                - self.empathy_spike * 0.15)
                .clamp(0.0, 2.5);
            self.last_trap_score = trap_score;
            let mut recovery_scores = Vec::with_capacity(self.runtime_recovery_ops.len());
            let mut recovery_alignments = Vec::with_capacity(self.runtime_recovery_ops.len());
            let mut recovery_absence_signals = Vec::with_capacity(self.runtime_recovery_ops.len());
            for operator in &self.runtime_recovery_ops {
                let alignment = self.normalized_alignment(probe_normalized, &operator.vector)?;
                let absence_signal = self.synthesize_absence_signal(operator, &live_motif_probe);
                let topology_pressure = self.normalized_topology_pressure(operator);
                let raw_score = alignment * 5.0
                    + operator.readiness_score * 1.25
                    + operator.persistence_score * 1.25
                    + absence_signal * 0.75
                    + topology_pressure * 0.5
                    + live_motif_probe.trap_pressure * 0.65;
                recovery_scores.push(raw_score);
                recovery_alignments.push(alignment);
                recovery_absence_signals.push(absence_signal);
            }

            let recovery_weights = stable_softmax(&recovery_scores);
            let mut active_specialist_id: Option<String> = None;
            let mut active_specialist_weight = 0.0f32;
            for (((operator, alignment), absence_signal), weight) in self
                .runtime_recovery_ops
                .iter()
                .zip(recovery_alignments.iter())
                .zip(recovery_absence_signals.iter())
                .zip(recovery_weights.iter())
            {
                if *weight > active_specialist_weight {
                    active_specialist_weight = *weight;
                    active_specialist_id = Some(operator.specialist_id.clone());
                }
                if *weight <= 1e-4 {
                    continue;
                }

                let alignment_gate = ((*alignment + 1.0) * 0.5).clamp(0.0, 1.0);
                let active_absence = *absence_signal * *weight * (0.25 + alignment_gate);
                self.last_absence_signal += active_absence;

                let direction = (&operator.vector - probe_normalized)?;
                let direction_norm = direction.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
                if direction_norm <= 1e-6 {
                    continue;
                }

                let affinity = alignment_gate;
                let basin_term = (1.0 / (1.0 + operator.basin_variance * 100.0)).clamp(0.1, 1.0);
                let radius_term = if operator.influence_radius > 0.0 {
                    (1.0 / (1.0 + operator.influence_radius * 10.0)).clamp(0.1, 1.0)
                } else {
                    1.0
                };
                let force_scale = self.recovery_force_scale
                    * activation_gate
                    * *weight
                    * (active_absence * (1.0 + trap_score))
                    * (0.25 + affinity)
                    * basin_term
                    * radius_term
                    * (0.25 + operator.readiness_score.max(0.0))
                    * (0.25 + operator.persistence_score.max(0.0))
                    * (1.0 + live_motif_probe.trap_pressure * 0.40).clamp(1.0, 1.6)
                    * (1.0 + self.empathy_spike * 0.15).clamp(1.0, 1.3);
                let scale_t = Tensor::new(force_scale / direction_norm.max(1e-6), device)?;
                recovery_force = (recovery_force + direction.broadcast_mul(&scale_t)?)?;
            }

            self.last_recovery_mag = recovery_force
                .sqr()?
                .sum_all()?
                .sqrt()?
                .to_scalar::<f32>()
                .unwrap_or(0.0);
            self.last_absence_signal = self.last_absence_signal.clamp(0.0, 3.0);
            self.active_recovery_specialist_id = active_specialist_id.clone();
            self.active_recovery_weight = active_specialist_weight;
            let step = self.current_step as i64;
            if step != self.last_recovery_counter_step {
                self.last_recovery_counter_step = step;
                if active_specialist_id.is_some()
                    && active_specialist_id == self.last_recovery_specialist_id
                {
                    self.specialist_run_length = self.specialist_run_length.saturating_add(1);
                } else {
                    self.specialist_run_length = if active_specialist_id.is_some() { 1 } else { 0 };
                    self.last_recovery_specialist_id = active_specialist_id;
                }
            }
        }

        Ok((motif_force, recovery_force))
    }

    pub(crate) fn refresh_live_hidden_bridge_vectors(&mut self) -> Result<()> {
        if self.runtime_motifs.is_empty() && self.runtime_recovery_ops.is_empty() {
            return Ok(());
        }

        let device = self.charge_tensor.device();
        let live_hidden_bank = collect_live_hidden_bank(
            &self.sentence_history,
            &self.current_sentence_embeddings,
            device,
        )?;
        if live_hidden_bank.is_empty() {
            return Ok(());
        }

        for motif in self.runtime_motifs.iter_mut() {
            if let Some(remapped) = reconstruct_hidden_from_live_bank(
                &motif.raw_signature,
                self.hidden_dim,
                device,
                &live_hidden_bank,
            )? {
                motif.vector = remapped;
                motif.live_hidden_remapped = true;
            }
        }

        for operator in self.runtime_recovery_ops.iter_mut() {
            if let Some(remapped) = reconstruct_hidden_from_live_bank(
                &operator.raw_signature,
                self.hidden_dim,
                device,
                &live_hidden_bank,
            )? {
                operator.vector = remapped;
            }
        }

        Ok(())
    }

    pub(crate) fn compute_secret_sauce_restore_force(
        &mut self,
        probe: &Tensor,
        layer_idx: usize,
    ) -> candle_core::Result<Tensor> {
        if self.secret_sauce_steps_remaining == 0
            || layer_idx < self.physics_start_layer
            || layer_idx > self.physics_end_layer
        {
            return Tensor::zeros(probe.shape(), probe.dtype(), probe.device());
        }

        let device = probe.device();
        let mut restore_force = Tensor::zeros(probe.shape(), probe.dtype(), device)?;
        let decay = if self.secret_sauce_decay_steps == 0 {
            0.0
        } else {
            (self.secret_sauce_steps_remaining as f32 / self.secret_sauce_decay_steps as f32)
                .clamp(0.0, 1.0)
        };

        if let Some(hidden_prior) = &self.secret_sauce_hidden_prior {
            let prior = hidden_prior.to_device(device)?.to_dtype(DType::F32)?;
            let direction = (&prior - probe)?;
            let norm = direction.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            if norm > 1e-6 {
                let scale =
                    Tensor::new((decay * SECRET_SAUCE_RESTORE_HIDDEN_WEIGHT) / norm, device)?;
                restore_force = (restore_force + direction.broadcast_mul(&scale)?)?;
            }
        }

        if let Some(sentence_prior) = &self.secret_sauce_sentence_prior {
            let prior = sentence_prior.to_device(device)?.to_dtype(DType::F32)?;
            let prior_norm_scalar = prior.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            if prior_norm_scalar > 1e-6 {
                let prior_scale = Tensor::new(1.0 / prior_norm_scalar, device)?;
                let prior_normalized = prior.broadcast_mul(&prior_scale)?;
                let probe_norm_scalar = probe.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
                let probe_normalized = if probe_norm_scalar > 1e-6 {
                    let probe_scale = Tensor::new(1.0 / probe_norm_scalar, device)?;
                    probe.broadcast_mul(&probe_scale)?
                } else {
                    probe.zeros_like()?
                };
                let alignment = probe_normalized
                    .broadcast_mul(&prior_normalized)?
                    .sum_all()?
                    .to_scalar::<f32>()?;
                let alignment_gate = ((alignment + 1.0) * 0.5)
                    .clamp(SECRET_SAUCE_RESTORE_SENTENCE_ALIGNMENT_FLOOR, 1.0);
                let direction = (&prior_normalized - &probe_normalized)?;
                let norm = direction.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
                if norm > 1e-6 {
                    let scale = Tensor::new(
                        (decay * SECRET_SAUCE_RESTORE_SENTENCE_WEIGHT * alignment_gate) / norm,
                        device,
                    )?;
                    restore_force = (restore_force + direction.broadcast_mul(&scale)?)?;
                }
            }
        }

        if let Some(momentum_prior) = &self.secret_sauce_momentum_prior {
            let prior = momentum_prior.to_device(device)?.to_dtype(DType::F32)?;
            let norm = prior.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            if norm > 1e-6 {
                let scale = Tensor::new(
                    (decay * SECRET_SAUCE_RESTORE_MOMENTUM_WEIGHT) / norm,
                    device,
                )?;
                restore_force = (restore_force + prior.broadcast_mul(&scale)?)?;
            }
        }

        Ok(restore_force)
    }

    /// Returns true when the codec-mediated specialist correction force should be applied
    /// this step: --specialist-correction-apply is on AND a rule-based specialist is loaded
    /// AND the trained rave codec is loaded. When true, apply_forces bypasses idle/coast
    /// early returns so the correction can land even with no other engine active.
    #[cfg(feature = "niodv4_bridge")]
    pub(crate) fn specialist_correction_active(&self) -> bool {
        self.specialist_correction_apply
            && self.vq_specialist.is_some()
            && rave_codec_global().is_some()
    }

    #[cfg(not(feature = "niodv4_bridge"))]
    pub(crate) fn specialist_correction_active(&self) -> bool {
        false
    }

    /// Returns true when a correction-packet store is loaded with at least one packet AND
    /// the rave codec + codebook are loaded. Like `specialist_correction_active`, this
    /// bypasses the idle/coast early returns so packet recall can land even with no other
    /// engine active.
    #[cfg(feature = "niodv4_bridge")]
    pub(crate) fn correction_packets_active(&self) -> bool {
        self.correction_packets
            .as_ref()
            .map(|s| !s.is_empty())
            .unwrap_or(false)
            && self.vq_codebook.is_some()
            && rave_codec_global().is_some()
    }

    #[cfg(not(feature = "niodv4_bridge"))]
    pub(crate) fn correction_packets_active(&self) -> bool {
        false
    }

    /// Bucket-mean/VQ-mediated correction-packet recall force.
    ///
    /// 1. Compress probe to 64D bucket-mean coordinates.
    /// 2. Quantize z via the codebook to a vq_code (8 bits).
    /// 3. Look up packets in the store keyed to that vq_code.
    /// 4. For each firing packet (probe-to-target distance > packet.distance_threshold):
    ///    a. delta_64 = packet.forward(z) -- pull-toward-target with norm = pull_strength.
    ///       When `correction_packet_payload_blend > 0` and the packet carries
    ///       `payload_z_64d`, an orthogonal payload component is mixed into this same
    ///       norm budget.
    ///    b. force_4096 = bucket-expand(delta_64), L2-clamped to
    ///       `correction_packet_clamp`.
    ///    c. Add force_4096 to probe_force.
    ///    d. Record packet fire (fire_count, last_fire_step) and packet_id telemetry.
    ///
    /// Telemetry is updated cumulatively across firings: `last_correction_packet_fire_count`
    /// is the number of firings, `last_correction_packet_force_norm` is the sum of clamped
    /// per-packet 4096D norms. Integrity invariant: when fire_count > 0,
    /// `last_intervention_applied=true` and `last_ghost_pull_delta_norm` is max-merged with
    /// the cumulative norm.
    ///
    /// Returns the input `probe_force` unchanged on any failure path (codec/codebook not
    /// loaded, no firing packets, encode/decode error, all forces below 1e-6, dim
    /// mismatch, NaN). Best-effort, never fatal.
    /// Runtime path uses **bucket-mean** 64D coordinates (matches `compress_hidden_state_to_64d`,
    /// which is what the codebook was trained on AND what `route_memory_workers.hidden_64d` is
    /// captured in). Mixing codec-encoder coords here was a coordinate-mismatch bug: the codebook
    /// would be queried with vectors from the wrong distribution and never recall the right bucket.
    /// The codec is intentionally NOT used in this path; force projection is bucket-expansion
    /// (each bucket-mean dim replicated across its 64 contiguous hidden dims).
    #[cfg(feature = "niodv4_bridge")]
    pub(crate) fn try_apply_correction_packet_force(
        &mut self,
        probe: &Tensor,
        probe_force: Tensor,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        // §10bf prompt-level codec activation gate. When the gate is
        // configured and the current prompt doesn't match any active
        // substring, skip force application entirely. Distinct from
        // the per-trajectory v11 gate which only changes mode within
        // an already-active turn.
        if !self.codec_active_for_current_prompt {
            self.record_correction_packet_arbitration(
                CorrectionPacketArbitrationChoice::NoPacket,
                "codec_prompt_gate",
                0,
                f32::INFINITY,
                0.0,
            );
            return Ok(probe_force);
        }
        // DEEP_DIVE_ROADMAP P1-B mint-readiness lock gate. When the active
        // selection's readiness is above threshold, the trajectory is in a
        // stable basin — further correction would overwrite the earned answer.
        // Hard skip per the gentleness-revolution finding (baseline_60: zero
        // post-readiness corrections → 80% mint-ready).
        //
        // Two selection sources are checked, in order:
        //   1. routing_cache (set by run_periodic_controller, requires
        //      structured prompt with structure_bias >= 0.42). Active in
        //      structured-task workflows.
        //   2. active_recovery_specialist_id (set at main.rs:12837 by the
        //      RuntimeRecoveryOperator softmax loop, fires whenever
        //      runtime_recovery_ops is non-empty). Active in the narrative
        //      beta artifact triage workflow where the periodic controller
        //      is dormant. iter-58 fix: original P1-B (iter-57) keyed only on
        //      routing_cache, so the gate was unreachable in narrative flow.
        let readiness_threshold = self.correction_packet_readiness_lock_threshold;
        if readiness_threshold > 0.0 {
            let mut active_readiness: f32 = 0.0;
            let mut lock_source: &'static str = "";

            if let Some(motif_id) = self
                .routing_cache
                .as_ref()
                .filter(|cache| cache.expires_at_step >= self.current_step)
                .map(|cache| cache.motif_id.clone())
            {
                let r = self
                    .runtime_motifs
                    .iter()
                    .find(|motif| motif.motif_id == motif_id)
                    .map(|motif| motif.readiness_score)
                    .unwrap_or(0.0);
                if r > active_readiness {
                    active_readiness = r;
                    lock_source = "routing_cache_motif";
                }
            }

            if let Some(spec_id) = self.active_recovery_specialist_id.as_ref() {
                let r = self
                    .runtime_recovery_ops
                    .iter()
                    .find(|op| &op.specialist_id == spec_id)
                    .map(|op| op.readiness_score)
                    .unwrap_or(0.0);
                if r > active_readiness {
                    active_readiness = r;
                    lock_source = "active_recovery_specialist";
                }
            }

            self.last_readiness_lock_source = lock_source.to_string();
            self.last_readiness_lock_score = active_readiness;
            if active_readiness > readiness_threshold {
                self.readiness_lock_skip_count += 1;
                self.record_correction_packet_arbitration(
                    CorrectionPacketArbitrationChoice::NoPacket,
                    "readiness_locked",
                    0,
                    f32::INFINITY,
                    0.0,
                );
                return Ok(probe_force);
            }
        }
        // §10bs/§10bu ghost-force-aware packet suppression with §10bx
        // post-bridge mode. Two behaviors:
        //   - Default (post_bridge_mode=false): block packets whenever
        //     bridge is/was active this turn (max of prev-step cache
        //     and live last_applied_ghost_mag).
        //   - post_bridge_mode=true: block packets ONLY at layer 0 of
        //     each step (where last_applied_ghost_mag was just zeroed
        //     by reset_force_telemetry, so the probe is still pre-
        //     bridge for this step). At layer 1+, last_applied_ghost_mag
        //     reflects the layer-0 bridge fire, so the probe HAS been
        //     bridge-shifted — let packets fire to correct on the
        //     post-bridge probe per algorithm doc.
        if self.correction_packet_suppress_when_bridge_force_above > 0.0 {
            let bridge_thr = self.correction_packet_suppress_when_bridge_force_above;
            let bridge_was_active_this_turn = self
                .prev_step_max_ghost_mag
                .max(self.last_applied_ghost_mag)
                > bridge_thr;
            let pre_bridge_this_step = self.last_applied_ghost_mag <= bridge_thr;
            if bridge_was_active_this_turn {
                if self.correction_packet_post_bridge_mode {
                    // Post-bridge mode: block ONLY layer 0 (pre-bridge
                    // probe state). Layer 1+ has bridge-shifted probe
                    // and packets are allowed to correct.
                    if pre_bridge_this_step {
                        return Ok(probe_force);
                    }
                } else {
                    // Default mode: block all layers when bridge is/was
                    // active (full suppression — matches bridge_only).
                    self.record_correction_packet_arbitration(
                        CorrectionPacketArbitrationChoice::NoPacket,
                        "bridge_force_suppression",
                        0,
                        f32::INFINITY,
                        0.0,
                    );
                    return Ok(probe_force);
                }
            }
        }
        // Compute bucket-mean probe vector (matches captured worker.hidden_64d distribution).
        let probe_64_vec: Vec<f32> = match compress_hidden_state_to_64d(probe) {
            Ok(v) if v.len() == 64 => v,
            _ => return Ok(probe_force),
        };
        let mut probe_64 = [0f32; 64];
        probe_64.copy_from_slice(&probe_64_vec);

        let current_step = self.current_step as u64;
        // RC1 outcome-feedback knobs (read once; default-off => no behavior change).
        let outcome_feedback = self.correction_packet_outcome_feedback;
        let outcome_gain = self.correction_packet_outcome_gain;
        let outcome_alpha = self.correction_packet_outcome_ema_alpha.clamp(0.0, 1.0);
        let outcome_floor = self.correction_packet_outcome_floor;
        let outcome_ceil = self
            .correction_packet_outcome_ceil
            .max(self.correction_packet_outcome_floor);
        let residual_gain = self.correction_packet_residual_gain;
        // RC1: settle the PREVIOUS step's armed fires against the current probe — which
        // IS their post-force hidden state — before this step fires. This folds each
        // fired packet's measured target-distance change into its effectiveness EMA.
        if outcome_feedback {
            if let Some(store) = self.correction_packets.as_mut() {
                store.settle_outcomes(&probe_64, outcome_alpha);
            }
        }
        // RC6: per-trajectory decay routing. Mirror the top-k router: when trajectory
        // routing is on and the turn is classified, use the per-class decay rate
        // (0.0 = no override => fall back to the global rate). The decay math and the
        // per-packet LOCK override (decay_rate=Some(1.0)) are untouched downstream, so
        // earned-answer preservation cannot regress.
        let decay_rate = if self.correction_packet_trajectory_routing {
            match self.trajectory_classified.as_deref() {
                Some("competent")
                    if self.correction_packet_trajectory_decay_competent > 0.0 =>
                {
                    self.correction_packet_trajectory_decay_competent
                }
                Some("drifting")
                    if self.correction_packet_trajectory_decay_drifting > 0.0 =>
                {
                    self.correction_packet_trajectory_decay_drifting
                }
                _ => self.correction_packet_decay_rate,
            }
        } else {
            self.correction_packet_decay_rate
        };
        self.last_correction_packet_effective_decay_rate = decay_rate;
        let decay_arg = if decay_rate > 0.0 && decay_rate < 1.0 {
            Some(decay_rate)
        } else {
            None
        };
        let unfold_threshold = self.correction_packet_unfold_encode_error_threshold;
        let unfold_factor = self.correction_packet_unfold_factor.max(0.0);

        // Compute relapse signals before the firing loop so each fire's effective_pull
        // can be unfold-boosted in lockstep with decay. Two independent triggers, OR'd:
        //   1. encode_error: probe drifted out of the codebook's training distribution
        //   2. mistake_reflex_retry: model just had to retry — explicit "I had to fix
        //      this" signal stronger than the OOD heuristic alone (§10ae)
        let (encode_error, encode_error_relapse) = {
            let codebook = match self.vq_codebook.as_ref() {
                Some(c) => c,
                None => return Ok(probe_force),
            };
            let vq_code = codebook.encode(&probe_64);
            let err = codebook.encode_error(&probe_64, vq_code);
            let triggered = unfold_threshold > 0.0 && err > unfold_threshold && unfold_factor > 1.0;
            (err, triggered)
        };
        let retry_threshold = self.correction_packet_unfold_on_retry_count;
        // Resolve the retry-source factor with max-combine fallback. When the CLI
        // override is > 1.0 we use the larger of (override, unfold_factor); when not
        // set, we fall back to the engine global. This lets retry-relapse boost
        // more strongly than OOD-relapse without ever boosting LESS than it.
        let retry_factor_override = self.correction_packet_unfold_retry_factor;
        let retry_factor = if retry_factor_override > 1.0 {
            retry_factor_override.max(unfold_factor)
        } else {
            unfold_factor
        };
        let retry_relapse = retry_threshold > 0
            && self.last_mistake_reflex_retry_count >= retry_threshold
            && retry_factor > 1.0;
        let relapse_active = encode_error_relapse || retry_relapse;
        // Per-source factors, max-combined into the applied factor. encode-error
        // always uses the global unfold_factor; retry uses retry_factor (which is
        // ≥ unfold_factor by construction). When both fire, the larger one wins.
        let encode_factor_applied = if encode_error_relapse {
            unfold_factor
        } else {
            1.0
        };
        let retry_factor_applied = if retry_relapse { retry_factor } else { 1.0 };
        let applied_unfold_factor = encode_factor_applied.max(retry_factor_applied);

        // Competence-aware suppression (§10aw, §10av): when the trajectory is
        // already in a "correct geometry" region, reduce pull to avoid overriding
        // correct trajectories. Two triggers OR'd:
        //   1. Entropy-gated: previous step's sampling entropy below threshold.
        //      Note: at temp=0.0 most tasks have low entropy → fires on every step.
        //   2. Density-gated: current step's firing count above threshold. The
        //      §10av "competence preservation threshold ~150" maps here.
        // Default factor=1.0 and both thresholds=0/0.0 mean disabled (legacy).
        let entropy_threshold = self.correction_packet_competence_entropy_threshold;
        let density_threshold = self.correction_packet_competence_density_threshold;
        let entropy_competent =
            entropy_threshold > 0.0 && self.last_sampling_entropy_norm < entropy_threshold;
        // Density evaluation happens AFTER firings are computed below; we hold
        // a placeholder competence_factor and resolve it once firings.len() is
        // known.
        let competence_suppress_factor = self
            .correction_packet_competence_suppress_factor
            .clamp(0.0, 1.0);

        // §10bh step-window fire-gate: skip the entire firing block when
        // current_step is outside [start, end]. Iter-228 motivation:
        // post-enumeration recovery shaping. Outside window = no firing,
        // no probe perturbation, deterministic baseline-equivalent behavior.
        if let Some((win_start, win_end)) = self.correction_packet_fire_step_window {
            if current_step < win_start as u64 || current_step > win_end as u64 {
                self.record_correction_packet_arbitration(
                    CorrectionPacketArbitrationChoice::NoPacket,
                    "step_window",
                    0,
                    f32::INFINITY,
                    0.0,
                );
                return Ok(probe_force);
            }
        }
        // Distance gate threshold (§10ay). Fold into competence resolution below.
        let distance_threshold = self.correction_packet_competence_distance_threshold;
        // §10bc direction-aware firing upper bound.
        let fire_max_distance = self.correction_packet_fire_max_distance;
        // §10cn: out-of-distribution suppression. When the prompt-K map
        // is configured but no substring matched AND the suppress flag
        // is set, skip packets entirely — treat as OOD for the trained
        // reflex. Validated on §10.m families (v2_090 hurts -0.17 without
        // this gate).
        if self.correction_packet_suppress_for_current_prompt {
            self.record_correction_packet_arbitration(
                CorrectionPacketArbitrationChoice::NoPacket,
                "prompt_no_match_suppression",
                0,
                f32::INFINITY,
                0.0,
            );
            return Ok(probe_force);
        }
        // §10be top-K direction-aware filter. Resolution priority:
        //   1. §10ck prompt-substring override (per-turn, set at gate eval).
        //   2. §10cq trajectory-router per-class override (per-step, only
        //      after classifier has settled and the per-class K > 0).
        //   3. global `correction_packet_fire_top_k`.
        let fire_top_k = if let Some(k) = self.current_prompt_top_k_override {
            k
        } else if self.correction_packet_trajectory_routing {
            // Once classified, route by class; preserves §10cp legacy when
            // both per-class K's are 0 (no override).
            match self.trajectory_classified.as_deref() {
                Some("competent") if self.correction_packet_trajectory_top_k_competent > 0 => {
                    self.correction_packet_trajectory_top_k_competent
                }
                Some("drifting") if self.correction_packet_trajectory_top_k_drifting > 0 => {
                    self.correction_packet_trajectory_top_k_drifting
                }
                _ => self.correction_packet_fire_top_k,
            }
        } else {
            self.correction_packet_fire_top_k
        };
        self.last_correction_packet_effective_fire_top_k = fire_top_k;
        // §10bd prompt-hash filter: only fire packets whose source prompt
        // matches the current prompt. Empty hash means filter is disabled
        // even if flag is set.
        let match_prompt_hash =
            self.correction_packet_fire_match_prompt_hash && !self.current_prompt_hash.is_empty();
        let current_ph_marker = if match_prompt_hash {
            format!("ph_{}", self.current_prompt_hash)
        } else {
            String::new()
        };
        let source_target_marker = self
            .current_prompt_source_target_override
            .as_ref()
            .map(|target| format!("target_id={target}"));
        // min_target_distance is computed during the firing filter so we can
        // gate suppression on probe proximity to known-correct geometry. Starts
        // at infinity and is reduced as we encounter closer packet targets.
        let mut min_target_distance: f32 = f32::INFINITY;

        // Each tuple: (packet_id, delta_64, effective_pull_used, distance, payload_blended).
        // Packet authority is evaluated here, immediately before hidden-force
        // projection, so enforce mode can drop weak/unknown candidates.
        let mut authority_decisions: Vec<crate::bridge::PacketAuthorityDecision> = Vec::new();
        if self
            .correction_packets
            .as_ref()
            .map(|s| s.is_empty())
            .unwrap_or(true)
        {
            self.record_correction_packet_arbitration(
                CorrectionPacketArbitrationChoice::NoPacket,
                "no_packet_store",
                0,
                f32::INFINITY,
                0.0,
            );
            return Ok(probe_force);
        }
        if self.vq_codebook.is_none() {
            self.record_correction_packet_arbitration(
                CorrectionPacketArbitrationChoice::NoPacket,
                "no_codebook",
                0,
                f32::INFINITY,
                0.0,
            );
            return Ok(probe_force);
        }

        let store = self.correction_packets.as_ref().expect("checked above");
        let codebook = self.vq_codebook.as_ref().expect("checked above");
        let vq_code = codebook.encode(&probe_64);
        let payload_blend = self.correction_packet_payload_blend.clamp(0.0, 1.0);
        let firings_with_dist: Vec<(String, [f32; 64], f32, f32, bool)> = store
            .forward_with_decay(vq_code, &probe_64, decay_arg)
            .into_iter()
            .filter_map(|(packet, _delta, decayed_pull)| {
                if let Some(marker) = source_target_marker.as_ref() {
                    if !packet.source_label.to_lowercase().contains(marker.as_str()) {
                        return None;
                    }
                }
                // Apply unfold AFTER decay was computed; recompute the delta with the
                // unfold-boosted effective pull so the magnitude reflects both knobs.
                // Per-packet unfold_factor and unfold_retry_factor independently
                // override engine globals, so an earned packet stamped with
                // `Some(1.0)` for both ignores both relapse boosts while
                // scaffolding around it (None) gets engine-global factors.
                // When both relapse sources fire, the larger per-packet factor wins.
                let encode_factor = if encode_error_relapse {
                    packet.effective_unfold_factor(unfold_factor)
                } else {
                    1.0
                };
                let retry_factor_packet = if retry_relapse {
                    packet.effective_unfold_retry_factor(retry_factor)
                } else {
                    1.0
                };
                let per_packet_factor = encode_factor.max(retry_factor_packet);
                let effective_pull = decayed_pull * per_packet_factor;
                // Compute probe-to-target distance once: used for both
                // the competence-aware trigger AND the §10bc direction-
                // aware fire-max-distance filter.
                let mut sq = 0f32;
                for i in 0..64 {
                    let d = probe_64[i] - packet.target_z_64d[i];
                    sq += d * d;
                }
                let dist = sq.sqrt();
                if dist.is_finite() && dist < min_target_distance {
                    min_target_distance = dist;
                }
                // §10bc: skip packets whose target is too far from probe
                // (direction-misaligned). Below the per-packet
                // distance_threshold (already filtered by forward_with_pull)
                // means probe is on top of target — no firing.
                // Above this max threshold means target is too unrelated —
                // also no firing. Together: a "ring of relevance".
                if fire_max_distance > 0.0 && dist > fire_max_distance {
                    return None;
                }
                // §10bd: source-aware filter. Skip packets whose embedded
                // ph_<hash> doesn't match the current prompt's hash.
                if match_prompt_hash && !packet.packet_id.contains(&current_ph_marker) {
                    return None;
                }
                let authority_decision = if self.correction_packet_authority_mode
                    == CorrectionPacketAuthorityMode::Off
                {
                    crate::bridge::PacketAuthorityDecision::allow_all("authority_gate_off")
                } else {
                    crate::bridge::decide_packet_authority(
                        packet,
                        crate::bridge::PacketAuthorityContext {
                            source_target_override: self
                                .current_prompt_source_target_override
                                .as_deref(),
                            prompt_family_matched: self
                                .current_prompt_top_k_match_substring
                                .is_some(),
                            current_prompt_hash: if self.current_prompt_hash.is_empty() {
                                None
                            } else {
                                Some(self.current_prompt_hash.as_str())
                            },
                            route_margin: self.last_route_margin,
                            nearest_ghost_present: self.last_nearest_ghost_id.is_some(),
                            target_distance: dist,
                        },
                    )
                };
                authority_decisions.push(authority_decision.clone());
                if self.correction_packet_authority_mode == CorrectionPacketAuthorityMode::Enforce
                    && !authority_decision.allowed
                {
                    return None;
                }
                let delta =
                    packet.forward_with_payload_blend(&probe_64, effective_pull, payload_blend)?;
                let payload_blended = payload_blend > 1e-6 && packet.payload_z_64d.is_some();
                Some((
                    packet.packet_id.clone(),
                    delta,
                    effective_pull,
                    dist,
                    payload_blended,
                ))
            })
            .collect();
        if self.correction_packet_authority_mode != CorrectionPacketAuthorityMode::Off
            && !authority_decisions.is_empty()
        {
            let best_allowed = authority_decisions
                .iter()
                .filter(|decision| decision.allowed)
                .max_by(|a, b| {
                    a.score
                        .partial_cmp(&b.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            let best_any = authority_decisions.iter().max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let decision = best_allowed
                .or(best_any)
                .expect("authority decision exists");
            self.last_packet_authority_score = decision.score;
            self.last_packet_authority_allowed = decision.allowed;
            self.last_packet_authority_reason = decision.reason.clone();
            self.last_packet_authority_blocked_reason = decision.blocked_reason.clone();
        }
        // §10be top-K direction-aware filter: keep only the K packets with
        // smallest probe-to-target distance. Sort ascending by distance,
        // truncate. When top_k >= len or top_k == 0, no-op.
        let mut fwd = firings_with_dist;
        if fire_top_k > 0 && fwd.len() > fire_top_k {
            fwd.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
            fwd.truncate(fire_top_k);
        }
        // RC1: capture each candidate's pre-fire probe->target distance (tuple field
        // .3) before it is dropped, so fires can be armed for next-step measurement.
        let fired_pre_dist: Vec<(String, f32)> = if outcome_feedback {
            fwd.iter().map(|t| (t.0.clone(), t.3)).collect()
        } else {
            Vec::new()
        };
        let firings: Vec<(String, [f32; 64], f32, bool)> = fwd
            .into_iter()
            .map(|(packet_id, delta, pull, _dist, payload_blended)| {
                (packet_id, delta, pull, payload_blended)
            })
            .collect();
        // Resolve the competence factor now that firing count is known.
        // Three combine modes (§10az/§10ba):
        //   "or" (default): any enabled trigger engages binary suppression
        //   "and": all enabled triggers must agree
        //   "continuous": graded suppression based on max trigger strength —
        //     factor = lerp(1.0, suppress_factor, strength) where strength ∈
        //     [0, 1] is max(density_strength, distance_strength) and each
        //     strength is the trigger value normalized to its threshold.
        let density_competent = density_threshold > 0 && firings.len() >= density_threshold;
        let distance_competent =
            distance_threshold > 0.0 && min_target_distance < distance_threshold;
        let entropy_enabled = entropy_threshold > 0.0;
        let density_enabled = density_threshold > 0;
        let distance_enabled = distance_threshold > 0.0;
        let any_enabled = entropy_enabled || density_enabled || distance_enabled;
        // §10bd Track 2 v11.5: per-trajectory adaptive routing. The
        // gate function is called per-LAYER (~32 calls per decode
        // token). On a current_step CHANGE we sum the previous step's
        // per-layer firings into trajectory_fire_count_sum as one
        // sample, then reset the per-step accumulator. firings.len()
        // for THIS layer is added to the per-step accumulator. This
        // captures total per-token firing density (sum across layers)
        // which the §10az v8 threshold was implicitly tuned against
        // (per-token telemetry's correction_packet_fire_count is the
        // last layer's value, but mean per-token total fires is a
        // proportional signal).
        let cur_step = self.current_step;
        let configured_mode = self.correction_packet_competence_combine_mode.clone();
        let routed_mode: String = if self.correction_packet_trajectory_routing {
            if self.trajectory_last_classified_step != cur_step {
                // Step boundary — finalize previous step's pending sum.
                if self.trajectory_pending_step_fires > 0 {
                    self.trajectory_fire_count_sum += self.trajectory_pending_step_fires as f32;
                    self.trajectory_fire_count_samples += 1;
                    self.trajectory_turn_step += 1;
                    let threshold = self.correction_packet_trajectory_fire_count_threshold;
                    // RC3: maintain a sliding window of recent per-step fire counts.
                    let window_len = if self.correction_packet_trajectory_window_len > 0 {
                        self.correction_packet_trajectory_window_len
                    } else {
                        self.correction_packet_trajectory_classify_step.max(1)
                    };
                    self.trajectory_window
                        .push_back(self.trajectory_pending_step_fires as f32);
                    while self.trajectory_window.len() > window_len {
                        self.trajectory_window.pop_front();
                    }
                    if self.trajectory_classified.is_none()
                        && self.trajectory_turn_step
                            >= self.correction_packet_trajectory_classify_step
                    {
                        // Initial one-shot classification (lifetime mean) — unchanged.
                        let mean_fire_count = self.trajectory_fire_count_sum
                            / self.trajectory_fire_count_samples as f32;
                        let label = if mean_fire_count > threshold {
                            "competent"
                        } else {
                            "drifting"
                        };
                        self.trajectory_classified = Some(label.to_string());
                        self.trajectory_last_reclassify_step = self.trajectory_turn_step;
                    } else if self.correction_packet_trajectory_reclassify_interval > 0
                        && self.trajectory_classified.is_some()
                        && self.trajectory_window.len() >= window_len
                        && self
                            .trajectory_turn_step
                            .saturating_sub(self.trajectory_last_reclassify_step)
                            >= self.correction_packet_trajectory_reclassify_interval
                    {
                        // RC3: mid-turn re-classification from the window mean with a
                        // hysteresis band so suppression mode / top-K track recent
                        // firing instead of a label frozen at classify_step.
                        let window_mean: f32 = self.trajectory_window.iter().sum::<f32>()
                            / self.trajectory_window.len() as f32;
                        let hyst = self.correction_packet_trajectory_hysteresis.max(0.0);
                        let cur = self.trajectory_classified.clone();
                        let new_label: Option<&str> = match cur.as_deref() {
                            Some("competent") if window_mean < threshold - hyst => Some("drifting"),
                            Some("drifting") if window_mean > threshold + hyst => Some("competent"),
                            _ => None,
                        };
                        if let Some(nl) = new_label {
                            self.trajectory_classified = Some(nl.to_string());
                            self.trajectory_reclassify_count += 1;
                        }
                        self.trajectory_last_reclassify_step = self.trajectory_turn_step;
                        self.last_trajectory_window_mean = window_mean;
                    }
                }
                self.trajectory_pending_step_fires = 0;
                self.trajectory_last_classified_step = cur_step;
            }
            self.trajectory_pending_step_fires = firings.len();
            match self.trajectory_classified.as_deref() {
                Some("competent") => "and".to_string(),
                _ => configured_mode,
            }
        } else {
            configured_mode
        };
        let mode = routed_mode.as_str();
        let competence_factor = if !any_enabled {
            1.0
        } else if mode == "continuous" {
            // Graded suppression. Each trigger contributes a strength in [0,1]:
            //   density: firings.len() / density_threshold (saturates at 1.0)
            //   distance: 1 - min_distance/distance_threshold (saturates at 1.0)
            //   entropy: 1 - last_entropy/entropy_threshold (saturates at 1.0)
            // Combined competence = max of enabled trigger strengths.
            // factor = 1.0 - (1.0 - suppress_factor) * combined_strength.
            // At strength=1.0 → factor = suppress_factor (binary equivalent).
            // At strength=0.0 → factor = 1.0 (no suppression).
            let mut combined_strength = 0.0f32;
            if density_enabled {
                let s = (firings.len() as f32 / density_threshold as f32).clamp(0.0, 1.0);
                if s > combined_strength {
                    combined_strength = s;
                }
            }
            if distance_enabled && min_target_distance.is_finite() {
                let s = (1.0 - min_target_distance / distance_threshold).clamp(0.0, 1.0);
                if s > combined_strength {
                    combined_strength = s;
                }
            }
            if entropy_enabled {
                let s = (1.0 - self.last_sampling_entropy_norm / entropy_threshold).clamp(0.0, 1.0);
                if s > combined_strength {
                    combined_strength = s;
                }
            }
            let depth = (1.0 - competence_suppress_factor).clamp(0.0, 1.0);
            (1.0 - depth * combined_strength).clamp(0.0, 1.0)
        } else if mode == "and" {
            let active = (!entropy_enabled || entropy_competent)
                && (!density_enabled || density_competent)
                && (!distance_enabled || distance_competent);
            if active {
                competence_suppress_factor
            } else {
                1.0
            }
        } else {
            // Default: OR — any enabled trigger engages binary suppression
            let active = entropy_competent || density_competent || distance_competent;
            if active {
                competence_suppress_factor
            } else {
                1.0
            }
        };
        self.last_correction_packet_min_target_distance = min_target_distance;
        // Telemetry: record the unfold context regardless of whether packets fired.
        self.last_correction_packet_vq_encode_error = encode_error;
        self.last_correction_packet_unfold_active = relapse_active;
        self.last_correction_packet_unfold_factor_applied = applied_unfold_factor;
        self.last_correction_packet_competence_factor = competence_factor;
        let arbitration_force_norm_estimate =
            firings.len() as f32 * self.correction_packet_clamp.max(0.0);
        let arbitration_decision = if self.correction_packet_arbitration_mode
            == CorrectionPacketArbitrationMode::Disabled
        {
            CorrectionPacketArbitrationDecision {
                choice: CorrectionPacketArbitrationChoice::PacketForce,
                reason: "disabled",
            }
        } else {
            choose_correction_packet_arbitration(
                self.correction_packet_arbitration_mode,
                CorrectionPacketArbitrationInput {
                    candidate_count: firings.len(),
                    min_target_distance,
                    competence_factor,
                    healthy_factor_threshold: self
                        .correction_packet_arbitration_healthy_factor_threshold,
                    stale_distance_threshold: self
                        .correction_packet_arbitration_stale_distance_threshold,
                },
            )
        };
        self.record_correction_packet_arbitration(
            arbitration_decision.choice,
            arbitration_decision.reason,
            firings.len(),
            min_target_distance,
            arbitration_force_norm_estimate,
        );
        match arbitration_decision.choice {
            CorrectionPacketArbitrationChoice::NoPacket
            | CorrectionPacketArbitrationChoice::PacketShadow => {
                return Ok(probe_force);
            }
            CorrectionPacketArbitrationChoice::PacketForce => {}
        }
        if firings.is_empty() {
            return Ok(probe_force);
        }

        let clamp = self.correction_packet_clamp.max(0.0);
        if clamp == 0.0 {
            return Ok(probe_force);
        }

        let hidden_dim = self.hidden_dim;
        let mut new_probe_force = probe_force;
        let mut total_norm: f32 = 0.0;
        let mut fire_count: usize = 0;
        let mut fired_ids: Vec<String> = Vec::with_capacity(firings.len());
        let mut effective_pull_sum: f32 = 0.0;
        let mut payload_blend_fire_count: usize = 0;
        // RC1: immutable handle for per-packet effectiveness lookups during projection,
        // plus accumulators for the mean-EMA telemetry. Held read-only across the loop.
        let outcome_store = self.correction_packets.as_ref();
        let mut effectiveness_sum: f32 = 0.0;
        let mut effectiveness_count: usize = 0;
        let mut residual_applied: usize = 0;

        for (packet_id, delta, effective_pull, payload_blended) in firings {
            // Bucket-expansion projection: each of the 64 dims spreads over its bucket of
            // hidden_dim/64 contiguous slots. Magnitude of resulting 4096D vector is
            // |delta_64| * sqrt(bucket_size). Clamp on the 4096D norm.
            let bucket_size = hidden_dim.div_ceil(64).max(1);
            let mut force_4096 = vec![0f32; hidden_dim];
            // RC2: if a residual shape is attached, rotate each block's force toward the
            // true within-block direction instead of a flat smear. None => legacy flat.
            let residual_slice: Option<&[f32]> = if residual_gain > 0.0 {
                match outcome_store.and_then(|s| s.residual_shape(&packet_id)) {
                    Some(shape) if shape.len() == hidden_dim => Some(shape),
                    _ => None,
                }
            } else {
                None
            };
            for (idx, slot) in force_4096.iter_mut().enumerate() {
                let bucket = (idx / bucket_size).min(63);
                let base = delta[bucket];
                *slot = match residual_slice {
                    Some(shape) => base * (1.0 + residual_gain * shape[idx]),
                    None => base,
                };
            }
            if residual_slice.is_some() {
                residual_applied += 1;
            }
            let raw_norm: f32 = force_4096.iter().map(|x| x * x).sum::<f32>().sqrt();
            if !raw_norm.is_finite() || raw_norm < 1e-6 {
                continue;
            }
            let (mut scaled_force, applied_norm_pre) = if raw_norm > clamp {
                let scale = clamp / raw_norm;
                for slot in force_4096.iter_mut() {
                    *slot *= scale;
                }
                (force_4096, clamp)
            } else {
                (force_4096, raw_norm)
            };
            // Competence-aware suppression (§10aw): scale post-clamp force by
            // competence_factor (1.0 = no change; <1.0 = reduce pull on
            // already-correct trajectories). This is the architectural unblock
            // pointed at by the iter-30..35 diagnostic chain.
            let applied_norm = if competence_factor < 1.0 - 1e-6 {
                for slot in scaled_force.iter_mut() {
                    *slot *= competence_factor;
                }
                applied_norm_pre * competence_factor
            } else {
                applied_norm_pre
            };
            // RC1: measured-effectiveness overlay. `factor = (1 + ema*gain)` clamped to
            // [floor, ceil]; 1.0 when the packet has never been measured or feedback is
            // off, so this is a strict no-op by default. A packet that has reduced
            // target distance (ema>0) keeps/gains force against decay; one that hasn't
            // (ema<0) is damped toward `floor`.
            let ema_factor = if outcome_feedback {
                match outcome_store.and_then(|s| s.effectiveness(&packet_id)) {
                    Some(e) => {
                        effectiveness_sum += e;
                        effectiveness_count += 1;
                        (1.0 + e * outcome_gain).clamp(outcome_floor, outcome_ceil)
                    }
                    None => 1.0,
                }
            } else {
                1.0
            };
            let applied_norm = if outcome_feedback && (ema_factor - 1.0).abs() > 1e-6 {
                for slot in scaled_force.iter_mut() {
                    *slot *= ema_factor;
                }
                applied_norm * ema_factor
            } else {
                applied_norm
            };
            let force_t = match Tensor::from_vec(scaled_force, (hidden_dim,), device) {
                Ok(t) => t,
                Err(_) => continue,
            };
            if force_t.dims() != new_probe_force.dims() {
                continue;
            }
            new_probe_force = match (new_probe_force + force_t) {
                Ok(t) => t,
                Err(_) => return Ok(Tensor::zeros(probe.shape(), probe.dtype(), device)?),
            };
            total_norm += applied_norm;
            fire_count += 1;
            fired_ids.push(packet_id);
            effective_pull_sum += effective_pull;
            if payload_blended {
                payload_blend_fire_count += 1;
            }
        }

        if fire_count == 0 {
            return Ok(new_probe_force);
        }

        // §10bb total-clamp: bound cumulative force across all firings so
        // store size doesn't cause linear interference scaling. When
        // total_clamp > 0 and the current cumulative L2 norm exceeds it,
        // scale the entire new_probe_force down. Doesn't affect per-packet
        // semantics, only the per-step total budget.
        let total_clamp = self.correction_packet_total_clamp.max(0.0);
        let final_total_norm = if total_clamp > 0.0 {
            // Recompute actual L2 norm of new_probe_force (the per-packet
            // accounting of `total_norm` sums clamped vector norms, which is
            // an UPPER BOUND on the actual L2 of the sum if vectors point in
            // different directions). Use the tensor's true norm.
            let actual_l2 = match (new_probe_force
                .sqr()
                .and_then(|t| t.sum_all())
                .and_then(|t| t.to_dtype(DType::F32))
                .and_then(|t| t.to_scalar::<f32>()))
            {
                Ok(sum_sq) => sum_sq.sqrt(),
                Err(_) => total_norm,
            };
            if actual_l2.is_finite() && actual_l2 > total_clamp {
                let scale = total_clamp / actual_l2;
                let scale_t = match Tensor::new(scale, device)
                    .and_then(|t| t.to_dtype(new_probe_force.dtype()))
                {
                    Ok(t) => t,
                    Err(_) => return Ok(new_probe_force),
                };
                new_probe_force = match new_probe_force.broadcast_mul(&scale_t) {
                    Ok(t) => t,
                    Err(_) => return Ok(Tensor::zeros(probe.shape(), probe.dtype(), device)?),
                };
                total_clamp
            } else {
                actual_l2
            }
        } else {
            total_norm
        };

        let live_minted_fired = if let Some(store) = self.correction_packets.as_ref() {
            store.record_fires_by_id(&fired_ids, current_step);
            // RC5: how many of this step's fires were packets minted this session.
            store.count_live_minted_fired(&fired_ids)
        } else {
            0
        };
        self.last_correction_packet_fire_count = fire_count;
        self.last_correction_packet_live_minted_fired_count = live_minted_fired;
        // RC1: record mean effectiveness over fired packets, then arm this step's fires
        // so the NEXT step's probe (their post-force state) settles their outcome.
        self.last_correction_packet_effectiveness_avg = if effectiveness_count > 0 {
            effectiveness_sum / effectiveness_count as f32
        } else {
            0.0
        };
        self.last_correction_packet_residual_applied = residual_applied;
        if outcome_feedback && !fired_ids.is_empty() {
            if let Some(store) = self.correction_packets.as_mut() {
                for (id, pre) in &fired_pre_dist {
                    if fired_ids.iter().any(|f| f == id) {
                        store.arm_outcome(id, *pre, current_step);
                    }
                }
            }
        }
        self.last_correction_packet_force_norm = final_total_norm;
        self.last_correction_packet_ids = fired_ids;
        if fire_count > 0 {
            self.last_correction_packet_vq_code = Some(vq_code);
        }
        self.last_correction_packet_effective_pull_avg = effective_pull_sum / fire_count as f32;
        self.last_intervention_applied = true;
        // Max-merge with prior interventions so a stronger specialist push isn't clobbered.
        self.last_ghost_pull_delta_norm = self.last_ghost_pull_delta_norm.max(total_norm);
        // Strategy label only set if not already claimed by an earlier intervention this step.
        if self.last_projection_strategy.is_empty()
            || self.last_projection_strategy == "none"
            || self.last_projection_strategy == "simple"
        {
            self.last_projection_strategy = if payload_blend_fire_count > 0 {
                "vq_correction_packet_payload".to_string()
            } else {
                "vq_correction_packet".to_string()
            };
        }
        self.last_forces_applied = true;
        Ok(new_probe_force)
    }

    pub(crate) fn build_packet_hybrid_metadata(
        &self,
        route_code: u8,
        text_fact: Option<&str>,
        agency_transition: Option<&str>,
        force_policy: &str,
        force_pull_strength: f32,
        force_distance_threshold: f32,
        force_decay_rate: Option<f32>,
        force_unfold_factor: Option<f32>,
        force_unfold_retry_factor: Option<f32>,
        answer_lock_boundary: &str,
    ) -> CorrectionPacketHybridMetadata {
        let route_motif_id = self
            .last_routed_motif_id
            .clone()
            .or_else(|| self.last_bridge_force_selected_ids.first().cloned());
        let nearest_ghost_distance = self
            .last_nearest_ghost_distance
            .is_finite()
            .then_some(self.last_nearest_ghost_distance);
        let second_nearest_ghost_distance = self
            .last_second_nearest_ghost_distance
            .is_finite()
            .then_some(self.last_second_nearest_ghost_distance);
        let route_margin = self
            .last_route_margin
            .is_finite()
            .then_some(self.last_route_margin);
        let projection_strategy = {
            let value = self.last_projection_strategy.trim();
            if value.is_empty() || value.eq_ignore_ascii_case("none") {
                None
            } else {
                Some(value.to_string())
            }
        };
        let ghost_pull_delta_norm = self
            .last_ghost_pull_delta_norm
            .is_finite()
            .then_some(self.last_ghost_pull_delta_norm);

        CorrectionPacketHybridMetadata {
            text_fact: text_fact
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string),
            payload_z_64d: None,
            route_code: Some(format!("vq_{route_code:03}")),
            route_motif_id,
            target_ghost_id: self.last_nearest_ghost_id.clone(),
            nearest_ghost_distance,
            second_nearest_ghost_distance,
            route_margin,
            agency_transition: agency_transition
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string),
            force_policy: (!force_policy.trim().is_empty()).then_some(force_policy.to_string()),
            force_pull_strength: Some(force_pull_strength),
            force_distance_threshold: Some(force_distance_threshold),
            force_decay_rate,
            force_unfold_factor,
            force_unfold_retry_factor,
            answer_lock_boundary: (!answer_lock_boundary.trim().is_empty())
                .then_some(answer_lock_boundary.to_string()),
            projection_strategy,
            ghost_pull_delta_norm,
        }
    }

    /// RC5: insert a just-minted packet into the live in-memory firing store so it
    /// can fire on a LATER step of the SAME process (closing the mint->insert->fire
    /// loop that previously required a process restart). `record` is the exact JSONL
    /// value returned by `write_correction_packet_record`; rebuilding the packet from
    /// it via `from_json_value` guarantees same-session firing is byte-identical to
    /// next-session firing. No-op when `correction_packet_live_mint` is disabled. A
    /// round-trip failure is logged and skipped — the file write already persisted the
    /// packet, so a future process still recovers it.
    #[cfg(feature = "niodv4_bridge")]
    fn insert_minted_packet_live(&mut self, record: serde_json::Value) {
        if !self.correction_packet_live_mint {
            return;
        }
        // RC2: clone the residual shape (if captured) before borrowing the store.
        let residual = if self.correction_packet_residual_gain > 0.0 {
            self.last_probe_residual_shape_4096.clone()
        } else {
            None
        };
        match crate::bridge::CorrectionPacket::from_json_value(record) {
            Ok(packet) => {
                let pid = packet.packet_id.clone();
                let store = self
                    .correction_packets
                    .get_or_insert_with(crate::bridge::CorrectionPacketStore::new);
                store.insert_live(packet);
                if let Some(shape) = residual {
                    store.set_residual_shape(&pid, shape);
                }
            }
            Err(e) => {
                eprintln!(
                    " [CORRECTION_PACKET] RC5 live-mint insert skipped (record round-trip failed): {e}"
                );
            }
        }
    }

    /// No-op when the correction-packet store is compiled out (no `niodv4_bridge`):
    /// the file write still happened, there is just no live firing store to insert into.
    #[cfg(not(feature = "niodv4_bridge"))]
    fn insert_minted_packet_live(&mut self, _record: serde_json::Value) {}

    /// Append one CorrectionPacket JSONL record to `correction_packets_out` capturing the
    /// most recent bucket-mean probe as the target. No-op when the writer is disabled
    /// (`correction_packets_out=None`), no probe was captured, or the codebook is missing.
    /// The output file is opened in append mode so repeated runs accumulate packets;
    /// pair with `--correction-packets-path <same-file>` on a subsequent run to load.
    ///
    /// This closes the "preserve correction across fresh process boundaries" North Star
    /// bullet end-to-end: the runtime mints packets from the live session, the file
    /// persists, and the next process loads them via the existing reader path.
    pub fn flush_correction_packet_capture(
        &mut self,
        prompt: &str,
        req_id: &str,
        agency_transition: Option<&str>,
    ) -> std::io::Result<bool> {
        let path = match self.correction_packets_out.as_ref() {
            Some(p) => p.clone(),
            None => return Ok(false),
        };
        let probe_64 = match self.last_probe_bucket_mean_64 {
            Some(arr) => arr,
            None => return Ok(false),
        };
        // Without a codebook we cannot key the packet — emit nothing rather than write a
        // record the receiver cannot route.
        #[cfg(feature = "niodv4_bridge")]
        let vq_code: u8 = match self.vq_codebook.as_ref() {
            Some(cb) => cb.encode(&probe_64),
            None => return Ok(false),
        };
        #[cfg(not(feature = "niodv4_bridge"))]
        let vq_code: u8 = 0;

        // §10bf bucket cap also applies to live-capture mints
        let cap = self.correction_packet_mint_bucket_cap;
        if cap > 0 {
            let count = *self.mint_bucket_counts.get(&vq_code).unwrap_or(&0);
            if count >= cap {
                return Ok(false);
            }
        }

        let prompt_hash = hash_str(prompt);
        let packet_id = format!(
            "live_capture::req_{req_id}::ph_{prompt_hash}::step_{:05}",
            self.current_step
        );
        let source_label = format!(
            "live_capture req_id={req_id} prompt_hash={prompt_hash} step={}",
            self.current_step
        );
        let text_fact = compact_agency_payload(prompt, 120);
        let hybrid_metadata = self.build_packet_hybrid_metadata(
            vq_code,
            Some(text_fact.as_str()),
            agency_transition,
            "live_capture",
            self.correction_packet_out_pull_strength,
            self.correction_packet_out_distance_threshold,
            None,
            None,
            None,
            "live_capture",
        );
        let record = write_correction_packet_record(
            &path,
            &packet_id,
            vq_code,
            &probe_64,
            self.correction_packet_out_pull_strength,
            self.correction_packet_out_distance_threshold,
            &source_label,
            self.current_step as u64,
            None, // inherit engine global decay rate
            None, // inherit engine global unfold factor
            None, // end-of-run captures don't carry an agency payload key
            None, // inherit engine global retry factor
            Some(&hybrid_metadata),
            self.correction_packet_out_unicode_v3,
        )?;
        // §10bf: increment count
        *self.mint_bucket_counts.entry(vq_code).or_insert(0) += 1;
        // RC5: also insert into the live store so it can fire later this process.
        self.insert_minted_packet_live(record);
        Ok(true)
    }

    /// Increment the per-payload-key contradiction counter and return the resulting
    /// effective multiplier for the next earned packet's pull strength. The
    /// multiplier is `lock_contradiction_multiplier × min(count, cap)`, clamped to
    /// at least 1.0. Each contradiction event for the same key escalates the boost,
    /// up to the configured cap — so a user who keeps correcting the same key
    /// progressively dominates that basin without unbounded growth.
    pub fn record_contradiction_for_key(&mut self, payload_key: &str) -> f32 {
        if payload_key.is_empty() {
            return self.correction_packet_lock_contradiction_multiplier;
        }
        let key = payload_key.to_ascii_lowercase();
        let count = self
            .contradiction_counts
            .entry(key)
            .and_modify(|c| *c = c.saturating_add(1))
            .or_insert(1);
        let cap = self.correction_packet_adaptive_contradiction_cap.max(1);
        let scaled = (*count).min(cap) as f32;
        (self.correction_packet_lock_contradiction_multiplier * scaled).max(1.0)
    }

    /// Append one CorrectionPacket JSONL record from a `[REQUEST: LOCK]` payload — the
    /// "earned answer" event. Like `mint_remember_correction_packets` but uses the
    /// higher `correction_packet_lock_pull_strength` so future probes that drift back
    /// into the earned bucket get pulled harder toward the locked answer state.
    ///
    /// `pull_multiplier > 1.0` indicates a contradiction-flavored packet. When
    /// passed, the packet uses the `lock_correction::` packet_id prefix and the
    /// `earned-correction:` source_label so downstream consumers can distinguish it
    /// from a normal earned packet, and the stamped pull is
    /// `lock_pull_strength × pull_multiplier`. Callers compute the multiplier via
    /// `record_contradiction_for_key` so the value scales with how many times the
    /// same key has been contradicted (capped by
    /// `--correction-packet-adaptive-contradiction-cap`).
    ///
    /// `packet_id = lock::req_<r>::ph_<h>::lh_<lhash>::step_<n>` (or
    /// `lock_correction::...` when multiplier > 1.0). `source_label = earned:
    /// <payload>` (or `earned-correction: <payload>`). Returns true on success.
    pub fn mint_lock_correction_packet(
        &mut self,
        lock_payload: &str,
        prompt: &str,
        req_id: &str,
        pull_multiplier: f32,
        agency_transition: Option<&str>,
    ) -> std::io::Result<bool> {
        let payload_trimmed = lock_payload.trim();
        if payload_trimmed.is_empty() {
            return Ok(false);
        }
        let path = match self.correction_packets_out.as_ref() {
            Some(p) => p.clone(),
            None => return Ok(false),
        };
        let probe_64 = match self.last_probe_bucket_mean_64 {
            Some(arr) => arr,
            None => return Ok(false),
        };
        #[cfg(feature = "niodv4_bridge")]
        let vq_code: u8 = match self.vq_codebook.as_ref() {
            Some(cb) => cb.encode(&probe_64),
            None => return Ok(false),
        };
        #[cfg(not(feature = "niodv4_bridge"))]
        let vq_code: u8 = 0;

        // §10bf bucket cap: skip if this bucket is already at the per-bucket
        // mint cap. Forces store-level diversity to address iter-62 U-shape.
        let cap = self.correction_packet_mint_bucket_cap;
        if cap > 0 {
            let count = *self.mint_bucket_counts.get(&vq_code).unwrap_or(&0);
            if count >= cap {
                return Ok(false);
            }
        }

        let prompt_hash = hash_str(prompt);
        let payload_hash = hash_str(payload_trimmed);
        let safe_multiplier = if pull_multiplier.is_finite() {
            pull_multiplier.max(0.0)
        } else {
            1.0
        };
        let is_contradiction = safe_multiplier > 1.0 + 1e-6;
        let (id_prefix, label_prefix, pull_strength) = if is_contradiction {
            (
                "lock_correction",
                "earned-correction",
                self.correction_packet_lock_pull_strength * safe_multiplier,
            )
        } else {
            (
                "lock",
                "earned",
                self.correction_packet_lock_pull_strength * safe_multiplier.max(1.0),
            )
        };
        let packet_id = format!(
            "{id_prefix}::req_{req_id}::ph_{prompt_hash}::lh_{payload_hash}::step_{:05}",
            self.current_step
        );
        let source_label = format!("{label_prefix}: {}", payload_trimmed);
        let key = agency_payload_key(payload_trimmed);
        let key_opt = if key.is_empty() {
            None
        } else {
            Some(key.as_str())
        };
        let hybrid_metadata = self.build_packet_hybrid_metadata(
            vq_code,
            Some(payload_trimmed),
            agency_transition,
            if is_contradiction {
                "lock_earned_contradiction"
            } else {
                "lock_earned"
            },
            pull_strength,
            self.correction_packet_out_distance_threshold,
            Some(1.0),
            Some(1.0),
            Some(1.0),
            "lock_payload",
        );
        let record = write_correction_packet_record(
            &path,
            &packet_id,
            vq_code,
            &probe_64,
            pull_strength,
            self.correction_packet_out_distance_threshold,
            &source_label,
            self.current_step as u64,
            // Stamp decay_rate=Some(1.0) so this earned-answer packet ignores the
            // engine's global decay even when scaffolding around it is decayed
            // aggressively. The "preserve earned answers before drift" semantics
            // is encoded here at mint time, not at runtime.
            Some(1.0),
            // Stamp unfold_factor=Some(1.0) so this earned-answer packet also
            // ignores relapse boosting — it's already preserved at full pull and
            // doesn't need extra amplification when the model drifts. Mirror of
            // the decay_rate=Some(1.0) preservation semantics.
            Some(1.0),
            // Stamp the agency_payload_key so semantic invalidation can match this
            // packet when a future LOCK contradicts a payload sharing the same key.
            key_opt,
            // Stamp unfold_retry_factor=Some(1.0) so this earned-answer packet
            // ignores the retry-source relapse boost too (mirror of the
            // unfold_factor preservation flag — earned answers are immune to BOTH
            // OOD-relapse boost and retry-relapse boost).
            Some(1.0),
            Some(&hybrid_metadata),
            self.correction_packet_out_unicode_v3,
        )?;
        // §10bf: increment bucket count on successful mint
        *self.mint_bucket_counts.entry(vq_code).or_insert(0) += 1;
        // RC5: also insert into the live store so it can fire later this process.
        self.insert_minted_packet_live(record);
        Ok(true)
    }

    /// Append one CorrectionPacket JSONL record per accepted REMEMBER tag emitted in
    /// the just-finished assistant turn. Each packet's target is the latest bucket-mean
    /// probe (the model's hidden state at end-of-turn). The packet_id and source_label
    /// embed the REMEMBER payload's hash so multiple REMEMBERs from the same turn don't
    /// collide.
    ///
    /// This is the "user as the living correction signal" North Star primitive: the
    /// REMEMBER tag is the user-facing trigger that mints durable, codec-keyed reflex
    /// memory. Returns the number of packets minted (0 when writer disabled, no probe
    /// captured, or the codebook is missing).
    pub fn mint_remember_correction_packets(
        &mut self,
        remember_payloads: &[String],
        prompt: &str,
        req_id: &str,
        agency_transition: Option<&str>,
    ) -> std::io::Result<usize> {
        if remember_payloads.is_empty() {
            return Ok(0);
        }
        let path = match self.correction_packets_out.as_ref() {
            Some(p) => p.clone(),
            None => return Ok(0),
        };
        let probe_64 = match self.last_probe_bucket_mean_64 {
            Some(arr) => arr,
            None => return Ok(0),
        };
        #[cfg(feature = "niodv4_bridge")]
        let vq_code: u8 = match self.vq_codebook.as_ref() {
            Some(cb) => cb.encode(&probe_64),
            None => return Ok(0),
        };
        #[cfg(not(feature = "niodv4_bridge"))]
        let vq_code: u8 = 0;

        // §10bf bucket cap: skip the entire batch if this bucket is already
        // at the per-bucket cap. All payloads in a single REMEMBER batch
        // share the same probe → same vq_code, so apply once.
        let cap = self.correction_packet_mint_bucket_cap;
        if cap > 0 {
            let count = *self.mint_bucket_counts.get(&vq_code).unwrap_or(&0);
            if count >= cap {
                return Ok(0);
            }
        }

        let prompt_hash = hash_str(prompt);
        let mut written = 0usize;
        for payload in remember_payloads {
            let payload_trimmed = payload.trim();
            if payload_trimmed.is_empty() {
                continue;
            }
            // Within-batch: also enforce cap as we go (each new packet
            // contributes to the same bucket).
            if cap > 0 {
                let count = *self.mint_bucket_counts.get(&vq_code).unwrap_or(&0);
                if count >= cap {
                    break;
                }
            }
            let payload_hash = hash_str(payload_trimmed);
            let packet_id = format!(
                "remember::req_{req_id}::ph_{prompt_hash}::rh_{payload_hash}::step_{:05}",
                self.current_step
            );
            let source_label = format!("remember: {}", payload_trimmed);
            let key = agency_payload_key(payload_trimmed);
            let key_opt = if key.is_empty() {
                None
            } else {
                Some(key.as_str())
            };
            let hybrid_metadata = self.build_packet_hybrid_metadata(
                vq_code,
                Some(payload_trimmed),
                agency_transition,
                "remember_scaffold",
                self.correction_packet_out_pull_strength,
                self.correction_packet_out_distance_threshold,
                None,
                None,
                None,
                "remember_payload",
            );
            let record = write_correction_packet_record(
                &path,
                &packet_id,
                vq_code,
                &probe_64,
                self.correction_packet_out_pull_strength,
                self.correction_packet_out_distance_threshold,
                &source_label,
                self.current_step as u64,
                // Scaffolding inherits the engine's global decay rate (default 1.0
                // = no decay; configurable via --correction-packet-decay-rate).
                None,
                // Scaffolding inherits the engine's global unfold factor too — gets
                // boosted on relapse like any other packet without per-packet override.
                None,
                // Stamp the agency_payload_key so semantic invalidation works on
                // REMEMBER scaffolding too if a contradicting LOCK arrives later.
                key_opt,
                // Scaffolding inherits the engine's global retry factor too — gets
                // retry-boosted on retry-relapse like any other packet without
                // per-packet override.
                None,
                Some(&hybrid_metadata),
                self.correction_packet_out_unicode_v3,
            )?;
            written += 1;
            // §10bf: increment bucket count
            *self.mint_bucket_counts.entry(vq_code).or_insert(0) += 1;
            // RC5: also insert into the live store so it can fire later this process.
            self.insert_minted_packet_live(record);
        }
        Ok(written)
    }

    /// Codec-mediated specialist correction force.
    ///
    /// `z = encode(probe)`; `delta = specialist.forward(z)`; `z' = z + [delta_x, delta_y, 0...]`;
    /// `force = decode(z') - decode(z)`. The force is L2-clamped to
    /// `specialist_correction_clamp` and added to `probe_force`. Telemetry
    /// (`last_specialist_force_applied`, `last_specialist_force_norm`,
    /// `last_intervention_applied`, `last_ghost_pull_delta_norm`,
    /// `last_projection_strategy`) is updated only when the force is actually added —
    /// closing the hollow-flag class flagged in CLAUDE.md.
    ///
    /// Silently returns the input `probe_force` unchanged on any failure path
    /// (codec not loaded, specialist not loaded, encode/decode error, force below 1e-6,
    /// shape mismatch, or NaN). Force application is best-effort, never fatal.
    #[cfg(feature = "niodv4_bridge")]
    pub(crate) fn try_apply_specialist_correction_force(
        &mut self,
        probe: &Tensor,
        probe_force: Tensor,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        if !self.specialist_correction_apply {
            return Ok(probe_force);
        }
        let codec = match rave_codec_global() {
            Some(c) => c,
            None => return Ok(probe_force),
        };

        // Encode probe with the trained encoder (NOT bucket-mean) for a high-fidelity 64D latent.
        let probe_2d = probe.unsqueeze(0)?; // (1, 4096)
        let z = match codec.encode(&probe_2d) {
            Ok(z) => z,
            Err(_) => return Ok(probe_force),
        };
        let z_vec: Vec<f32> = match z.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
            Ok(v) => v,
            Err(_) => return Ok(probe_force),
        };

        // Run specialist on the 64D latent (immutable self borrow scoped to this match).
        let delta = match self.vq_specialist.as_ref() {
            Some(s) => s.forward(&z_vec),
            None => return Ok(probe_force),
        };
        let two_d_norm = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
        if !two_d_norm.is_finite() || two_d_norm < 1e-6 {
            return Ok(probe_force);
        }

        // z' = z + [delta_x, delta_y, 0, ...]
        let mut delta_64 = vec![0f32; 64];
        delta_64[0] = delta[0];
        delta_64[1] = delta[1];
        let delta_t = Tensor::from_vec(delta_64, (1, 64), device)?;
        let z_prime = (&z + &delta_t)?;

        let decoded_z = match codec.decode(&z) {
            Ok(t) => t,
            Err(_) => return Ok(probe_force),
        };
        let decoded_zp = match codec.decode(&z_prime) {
            Ok(t) => t,
            Err(_) => return Ok(probe_force),
        };

        let raw_force = (decoded_zp.flatten_all()? - decoded_z.flatten_all()?)?;
        let force_norm: f32 = raw_force.sqr()?.sum_all()?.sqrt()?.to_scalar()?;
        if !force_norm.is_finite() || force_norm < 1e-6 {
            return Ok(probe_force);
        }

        let clamp = self.specialist_correction_clamp.max(0.0);
        let (final_force, applied_norm) = if clamp == 0.0 {
            return Ok(probe_force);
        } else if force_norm > clamp {
            let scale = clamp / force_norm;
            let s = Tensor::new(scale, device)?;
            (raw_force.broadcast_mul(&s)?, clamp)
        } else {
            (raw_force, force_norm)
        };

        if final_force.dims() != probe_force.dims() {
            return Ok(probe_force);
        }

        let new_probe_force = (probe_force + final_force)?;
        self.last_specialist_force_applied = true;
        self.last_specialist_force_norm = applied_norm;
        self.last_intervention_applied = true;
        self.last_ghost_pull_delta_norm = applied_norm;
        self.last_projection_strategy = "vq_specialist_force".to_string();
        self.last_forces_applied = true;
        Ok(new_probe_force)
    }
}

impl PhysicsEngine for PrincipiaEngine {
    fn apply_forces(
        &mut self,
        state: &Tensor,
        layer_idx: usize,
        ghost_vector: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let device = &state.device();
        let _original_dtype = state.dtype();
        let state_f32 = state.to_dtype(DType::F32)?;
        // Shape: [batch, seq, hidden]
        let (b_sz, seq_len, hidden_sz) = state_f32.dims3()?;

        // Dimension Safety Check
        if hidden_sz != self.hidden_dim {
            return Err(candle_core::Error::Msg(
                format!(
                    "Dim mismatch: state hidden_sz={} != engine hidden_dim={}",
                    hidden_sz, self.hidden_dim
                )
                .into(),
            ));
        }

        // Self-Reflective Rule Evolution
        if self.current_step % 10 == 0 {
            let _ = self.evolve_physics_rules();
        }

        let pressure_gate = pressure_activation_gate(self.last_ghost_pre_norm);
        let request_gate = visible_request_activation_gate(
            self.visible_request_gate,
            self.current_step,
            self.last_request_token,
            self.last_hidden_request,
            self.adrenaline,
            self.focus_lock_remaining_ticks,
        );
        let bridge_gate_floor = self.bridge_motif_gate_floor();
        let activation_gate = pressure_gate.max(request_gate).max(bridge_gate_floor);
        let engine_status = if activation_gate > 1e-6 {
            ForceEngineStatus::Active
        } else if self.sentence_history.is_empty() && self.goal_embedding.is_none() {
            ForceEngineStatus::Idle
        } else {
            ForceEngineStatus::Coasting
        };
        let (start_layer, _) = self.get_physics_layer_range();
        if layer_idx == start_layer {
            self.reset_force_telemetry(activation_gate, engine_status);
        }

        let bridge_lane = self.bridge_force_layer_selected(layer_idx);
        let worker_lane = self.specialist_worker_influence_lane_layer(layer_idx);
        if !(bridge_lane || worker_lane)
            && !self.specialist_correction_active()
            && !self.correction_packets_active()
        {
            return Ok(Tensor::zeros(
                (b_sz, seq_len, hidden_sz),
                DType::F32,
                device,
            )?);
        }

        // ISOLATE LAST TOKEN (The Probe)
        // We only calculate forces for the active particle
        let probe = state_f32.i((.., seq_len - 1, ..))?.flatten_all()?;

        // Capture latest bucket-mean probe for any runtime path that needs a
        // stable 64D key: packet minting and REMEMBER vault saves.
        if self.correction_packets_out.is_some() || self.vault_client.is_some() {
            if let Ok(probe64_vec) = compress_hidden_state_to_64d(&probe) {
                if probe64_vec.len() == 64 {
                    let mut arr = [0f32; 64];
                    arr.copy_from_slice(&probe64_vec);
                    self.last_probe_bucket_mean_64 = Some(arr);
                }
            }
        }
        // RC2: capture the within-block residual SHAPE (what bucket-mean discards) so a
        // minted packet can later steer with a true 4096D direction, not a flat smear.
        // Only when residual projection is enabled and a writer is configured.
        if self.correction_packet_residual_gain > 0.0 && self.correction_packets_out.is_some() {
            if let (Ok(probe_full), Some(mean64)) =
                (probe.to_vec1::<f32>(), self.last_probe_bucket_mean_64)
            {
                let hidden_dim = probe_full.len();
                if hidden_dim > 0 {
                    let bucket_size = hidden_dim.div_ceil(64).max(1);
                    let mut shape = vec![0f32; hidden_dim];
                    let mut norm_sq = 0f32;
                    for (idx, &v) in probe_full.iter().enumerate() {
                        let b = (idx / bucket_size).min(63);
                        let r = v - mean64[b];
                        shape[idx] = r;
                        norm_sq += r * r;
                    }
                    let norm = norm_sq.sqrt();
                    self.last_probe_residual_shape_4096 = if norm.is_finite() && norm > 1e-6 {
                        for s in shape.iter_mut() {
                            *s /= norm;
                        }
                        Some(shape)
                    } else {
                        None
                    };
                }
            }
        }

        // Normalize probe for metric calculations
        let probe_norm_scalar = probe.sqr()?.sum_all()?.sqrt()?;
        let probe_normalized = probe.broadcast_div(&probe_norm_scalar)?;

        // Update bridge telemetry (nearest ghost basins) - Always capture route state
        let _ = self.update_bridge_telemetry(&probe_normalized);
        // Shadow route-memory diagnostics are observational and must be visible even
        // when the generation path exits before applying any force.
        let selected_worker_idx = self
            .update_specialist_worker_shadow(&probe_normalized)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        if !bridge_lane && worker_lane {
            let mut probe_force = Tensor::zeros(probe.shape(), probe.dtype(), probe.device())?;
            probe_force = self
                .apply_specialist_worker_influence(
                    &probe,
                    selected_worker_idx,
                    probe_force,
                    layer_idx,
                )
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            self.last_forces_applied = true;
            return self.finalize_worker_only_residual_from_flat_force(
                &state_f32,
                &probe_force,
                b_sz,
                seq_len,
                hidden_sz,
                layer_idx,
                device,
                state,
            );
        }

        // VQ codec + phase2 specialist OBSERVATIONAL telemetry. Codebook encode runs whenever
        // a codebook is loaded — it's a 64D nearest-centroid lookup that costs nothing and gives
        // the audit pipeline a direct probe→bucket signal without needing the offline §10eg
        // Gaussian projection workaround in scripts/audit_probe_bucket_alignment.py. The
        // specialist forward is the part that actually requires the rule-based specialist;
        // when only the codebook is loaded, last_correction_delta_norm/last_specialist_activated
        // stay at their reset defaults. Records what the rule WOULD do in 2D-coord space without
        // claiming any actual hidden-state force was applied. Force application happens later,
        // gated behind --specialist-correction-apply, which is the only place
        // last_intervention_applied/last_ghost_pull_delta_norm/last_projection_strategy can be set
        // by this path. This split closes the hollow-flag bug where 2D-coord-norm was reported
        // as ghost_pull_delta_norm without the returned tensor ever being modified.
        #[cfg(feature = "niodv4_bridge")]
        if let Some(ref codebook) = self.vq_codebook {
            if let Ok(probe64) = compress_hidden_state_to_64d(&probe) {
                let vq_code = codebook.encode(&probe64);
                let vq_err = codebook.encode_error(&probe64, vq_code);
                self.last_vq_code_assigned = Some(vq_code);
                self.last_vq_encode_error = vq_err;
                if let Some(ref specialist) = self.vq_specialist {
                    let delta = specialist.forward(&probe64);
                    let delta_norm = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
                    self.last_correction_delta_norm = delta_norm;
                    self.last_specialist_activated = delta_norm > 1e-6;
                }
            }
        }

        if self.sentence_history.is_empty()
            && self.goal_embedding.is_none()
            && !self.bridge_influence_smoke_active()
            && !self.bridge_influence_selective_active()
            && !self.bridge_gate34_latch_active()
            && self.specialist_memory_workers_mode != SpecialistMemoryWorkerMode::Influence
            && request_gate <= 1e-6
            && bridge_gate_floor <= 1e-6
            && !self.specialist_correction_active()
            && !self.correction_packets_active()
        {
            return Ok(state.zeros_like()?);
        }

        let mut probe_force = Tensor::zeros(probe.shape(), probe.dtype(), probe.device())?;

        // Codec-mediated specialist correction force (rule-based phase2 specialist).
        // Runs first so it stacks with any other bridge forces below. Updates
        // last_intervention_applied / last_ghost_pull_delta_norm only when the force is
        // actually added, closing the hollow-flag gap left by the observational hook above.
        #[cfg(feature = "niodv4_bridge")]
        if self.specialist_correction_apply {
            probe_force =
                self.try_apply_specialist_correction_force(&probe, probe_force, device)?;
        }

        // VQ-keyed correction-packet recall (the "scar tissue → reflex" path).
        // Stacks with any specialist correction force already applied. Each firing packet
        // contributes a clamped 4096D pull-toward-target force; per-packet contributions
        // sum into probe_force.
        #[cfg(feature = "niodv4_bridge")]
        if self.correction_packets_active() {
            probe_force = self.try_apply_correction_packet_force(&probe, probe_force, device)?;
        }

        // =================================================================
        //  STRATEGY #2: PRESSURE-GATED FORCE APPLICATION
        //  Geometry, not token index, decides when the steering loop pushes.
        // =================================================================

        if activation_gate <= 1e-6
            && !self.bridge_influence_smoke_active()
            && !self.bridge_influence_selective_active()
            && !self.bridge_gate34_latch_active()
            && self.specialist_memory_workers_mode != SpecialistMemoryWorkerMode::Influence
            && bridge_gate_floor <= 1e-6
            && !self.specialist_correction_active()
            && !self.correction_packets_active()
        {
            return Ok(Tensor::zeros(
                (b_sz, seq_len, hidden_sz),
                DType::F32,
                device,
            )?);
        }

        self.last_forces_applied = true;

        // 1. Calculate Gravitydden]

        // Novel: Inject PINN loss
        if let Some(pinn) = &self.pinn_loss {
            let pinn_adj = self
                .adjust_pinn_with_lpm(pinn)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            // pinn_adj is likely [hidden]
            probe_force = (probe_force + pinn_adj)?;
        }

        // 1. Calculate Gravity
        if !self.sentence_history.is_empty() {
            // FIX: Ignore the MOST RECENT particle to prevent "Green Sky" / Self-Attraction Loop
            let n = self.sentence_history.len();
            let effective_n = if n > 0 { n - 1 } else { 0 };

            if effective_n > 0 {
                let hist_pos_vec: Vec<Tensor> = self
                    .sentence_history
                    .iter()
                    .take(effective_n)
                    .map(|p| p.position.clone())
                    .collect();
                let hist_pos = Tensor::stack(&hist_pos_vec, 0)?; // [N-1, hidden]

                let hist_mass_vec: Vec<f32> = self
                    .sentence_history
                    .iter()
                    .take(effective_n)
                    .map(|p| {
                        let clean = p.text.trim();
                        // NIODOO v1.0: "Smart Filter"
                        // Skip punctuation/short words (len < 3) during active flight
                        if clean.len() < 3 {
                            0.0
                        } else {
                            p.current_mass(self.current_step, &self.params)
                        }
                    })
                    .collect();
                let hist_mass = Tensor::from_vec(hist_mass_vec, (effective_n, 1), device)?;

                // Quantum MAE / CFD
                let _ = self.process_photon_subsamples(&probe, effective_n);

                let probe_expanded = probe.unsqueeze(0)?.broadcast_as(hist_pos.shape())?;
                // println!(" [DBG] hist_pos: {:?}, Dtype: {:?}", hist_pos.dims(), hist_pos.dtype());
                // println!(" [DBG] probe_expanded Dtype: {:?}", probe_expanded.dtype());
                let r_vec = (hist_pos - probe_expanded)?;
                // println!(" [DBG] r_vec Dtype: {:?}", r_vec.dtype());
                let dist_sq = r_vec.sqr()?.sum_keepdim(1)?;
                // println!(" [DBG] dist_sq: {:?}", dist_sq.dims());

                let epsilon_t = Tensor::from_vec(vec![1e-6 as f32], (1,), device)?;
                let gravity_t = Tensor::from_vec(vec![self.dynamic_gravity as f32], (1,), device)?;
                // F = G * m1 * m2 / r^2

                let num = hist_mass.broadcast_mul(&gravity_t)?;
                let den = dist_sq.broadcast_add(&epsilon_t)?;
                let force_mags = num.broadcast_div(&den)?;
                let dist = den.sqrt()?;
                let force_vectors = r_vec.broadcast_mul(&(force_mags / dist)?)?;

                let summed_gravity = force_vectors.sum(0)?; // [hidden]
                let gate_t = Tensor::from_vec(vec![activation_gate], (1,), device)?;
                let scaled_gravity = summed_gravity.broadcast_mul(&gate_t)?;

                // TELEMETRY: Capture gravity magnitude
                self.last_gravity_mag = scaled_gravity
                    .sqr()?
                    .sum_all()?
                    .sqrt()?
                    .to_scalar::<f32>()
                    .unwrap_or(0.0);

                probe_force = (probe_force + scaled_gravity)?;
            }
        }

        // 1.5. Ghost Vector Gravity (The "Niodoo" Attractor)
        if self.specialist_memory_workers_mode != SpecialistMemoryWorkerMode::Influence {
            if let Some(ghost) = ghost_vector {
                // ... (Ghost Logic)
                let ghost_f32 = ghost.to_device(device)?.to_dtype(DType::F32)?;
                let ghost_flat = ghost_f32.flatten_all()?;
                // Gravity = G * m_ghost * m_probe / r^2
                // Scalar mul usually supports f64, but explicit is safer
                let ghost_g_t = Tensor::from_vec(vec![self.last_ghost_gain], (1,), device)?;
                let g_force = ghost_flat.broadcast_mul(&ghost_g_t)?;

                // TELEMETRY: Capture ghost force magnitude
                self.last_applied_ghost_mag = g_force
                    .sqr()?
                    .sum_all()?
                    .sqrt()?
                    .to_scalar::<f32>()
                    .unwrap_or(0.0);
                // §10bt cache for the §10bs gate. Take max-of-turn so
                // the gate sees the strongest ghost signal observed so
                // far in this turn, even on layer-0 of a new decode
                // step (where last_applied_ghost_mag was just zeroed).
                if self.last_applied_ghost_mag > self.prev_step_max_ghost_mag {
                    self.prev_step_max_ghost_mag = self.last_applied_ghost_mag;
                }
                // Preserve any prior intervention magnitude (e.g. specialist correction force
                // applied earlier in this apply_forces call). Use max so the recorded
                // intervention norm reflects the strongest registered intervention this step,
                // and so a zero ghost mag does not clobber a real specialist push.
                self.last_ghost_pull_delta_norm = self
                    .last_ghost_pull_delta_norm
                    .max(self.last_applied_ghost_mag);
                self.last_intervention_applied = self.last_ghost_pull_delta_norm > 1e-6;
                self.last_applied_ghost_vector = Some(
                    tensor_to_vec_f32(&g_force)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?,
                );

                probe_force = (probe_force + g_force)?;
            }
        }

        #[cfg(feature = "niodv4_bridge")]
        if self.bridge_influence_smoke_active() {
            if let Some(id) = self.last_nearest_ghost_id.clone() {
                if let Some(ghost_target) = self
                    .get_bridge_ghost_vector(&id, device)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?
                {
                    let ghost_flat = ghost_target
                        .flatten_all()?
                        .to_dtype(DType::F32)?
                        .to_device(device)?;
                    // Direction toward nearest basin in hidden space.
                    let delta = (ghost_flat - probe.clone())?;
                    let raw_norm = delta.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
                    if raw_norm > 1e-6 {
                        let max_norm: f32 = self.bridge_influence_smoke_clamp.clamp(0.0, 0.03);
                        let scale = if raw_norm > max_norm {
                            max_norm / raw_norm
                        } else {
                            1.0
                        };
                        let scale_t = Tensor::from_vec(vec![scale], (1,), device)?;
                        let smoke_delta = delta.broadcast_mul(&scale_t)?;
                        let clamped_norm = (raw_norm * scale).min(max_norm);
                        probe_force = (probe_force + smoke_delta)?;
                        self.last_ghost_pull_delta_norm = clamped_norm;
                        self.last_intervention_applied = clamped_norm > 1e-6;
                        self.last_projection_strategy = "smoke".to_string();
                    }
                }
            }
        }

        #[cfg(feature = "niodv4_bridge")]
        if self.bridge_influence_selective_active() {
            let id_opt = self.last_nearest_ghost_id.clone();
            let route_margin = self.last_route_margin;
            let run_length = self.last_ghost_id_run_length;
            let cooldown = self.last_ghost_switch_cooldown_remaining;
            let threshold = self.bridge_margin_threshold;
            let stability_k = self.bridge_stability_k;

            let skip_reason: Option<&'static str> = if id_opt.is_none() {
                Some("no_id")
            } else if cooldown > 0 {
                Some("cooldown")
            } else if route_margin < threshold {
                Some("margin")
            } else if run_length < stability_k {
                Some("stability")
            } else {
                None
            };

            if let Some(reason) = skip_reason {
                self.last_ghost_pull_delta_norm = 0.0;
                self.last_intervention_applied = false;
                self.last_projection_strategy = format!("selective_skip:{}", reason);
            } else if let Some(id) = id_opt {
                if let Some(ghost_target) = self
                    .get_bridge_ghost_vector(&id, device)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?
                {
                    let ghost_flat = ghost_target
                        .flatten_all()?
                        .to_dtype(DType::F32)?
                        .to_device(device)?;
                    let delta = (ghost_flat - probe.clone())?;
                    let raw_norm = delta.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
                    if raw_norm > 1e-6 {
                        let base_clamp = self.bridge_influence_smoke_clamp.clamp(0.0, 0.03);
                        let max_norm = if self.bridge_scale_by_margin {
                            let f = (route_margin / (2.0 * threshold.max(1e-6))).clamp(0.0, 1.0);
                            base_clamp * f
                        } else {
                            base_clamp
                        };
                        if max_norm > 1e-6 {
                            let scale = if raw_norm > max_norm {
                                max_norm / raw_norm
                            } else {
                                1.0
                            };
                            let scale_t = Tensor::from_vec(vec![scale], (1,), device)?;
                            let smoke_delta = delta.broadcast_mul(&scale_t)?;
                            let clamped_norm = (raw_norm * scale).min(max_norm);
                            probe_force = (probe_force + smoke_delta)?;
                            self.last_ghost_pull_delta_norm = clamped_norm;
                            self.last_intervention_applied = clamped_norm > 1e-6;
                            self.last_projection_strategy = "selective".to_string();
                        } else {
                            self.last_ghost_pull_delta_norm = 0.0;
                            self.last_intervention_applied = false;
                            self.last_projection_strategy =
                                "selective_skip:scaled_zero".to_string();
                        }
                    } else {
                        self.last_ghost_pull_delta_norm = 0.0;
                        self.last_intervention_applied = false;
                        self.last_projection_strategy = "selective_skip:zero_delta".to_string();
                    }
                }
            }
        }

        #[cfg(feature = "niodv4_bridge")]
        if self.bridge_gate34_latch_active() {
            let step_i64 = self.current_step as i64;
            if self.gate34_last_step != step_i64 {
                self.gate34_last_step = step_i64;
                if self.gate34_phase == Gate34Phase::Warmup {
                    let topk_candidates = match self.gate34_target_source.as_str() {
                        "motifs" => self.find_topk_motif_candidates(
                            &probe_normalized,
                            self.gate34_acquire_top_k,
                        ),
                        "specialists" => self.find_topk_specialist_candidates(
                            &probe_normalized,
                            self.gate34_acquire_top_k,
                        ),
                        _ => self.find_topk_ghost_candidates(
                            &probe_normalized,
                            self.gate34_acquire_top_k,
                        ),
                    };
                    if let Ok(topk) = topk_candidates {
                        for (id, dist) in topk {
                            let dist = dist as f32;
                            *self.gate34_candidate_counts.entry(id.clone()).or_insert(0) += 1;
                            *self
                                .gate34_candidate_margin_sum
                                .entry(id.clone())
                                .or_insert(0.0) += self.last_route_margin;
                            let best = self
                                .gate34_candidate_best_margin
                                .entry(id.clone())
                                .or_insert(f32::MIN);
                            if self.last_route_margin > *best {
                                *best = self.last_route_margin;
                            }
                            if dist.is_finite() {
                                *self
                                    .gate34_candidate_distance_sum
                                    .entry(id.clone())
                                    .or_insert(0.0) += dist;
                                *self
                                    .gate34_candidate_distance_sq_sum
                                    .entry(id.clone())
                                    .or_insert(0.0) += dist * dist;
                                let min_entry = self
                                    .gate34_candidate_distance_min
                                    .entry(id.clone())
                                    .or_insert(f32::INFINITY);
                                if dist < *min_entry {
                                    *min_entry = dist;
                                }
                                let max_entry = self
                                    .gate34_candidate_distance_max
                                    .entry(id)
                                    .or_insert(f32::NEG_INFINITY);
                                if dist > *max_entry {
                                    *max_entry = dist;
                                }
                            }
                        }
                    }
                    self.gate34_warmup_step_count = self.gate34_warmup_step_count.saturating_add(1);
                    if self.gate34_warmup_step_count >= self.gate34_warmup_steps {
                        if self.gate34_candidate_counts.is_empty() {
                            self.gate34_phase = Gate34Phase::Released;
                            self.gate34_release_reason = Some("no_target_vector".to_string());
                            self.last_projection_strategy =
                                "gate34_released:no_target_vector".to_string();
                        } else {
                            let max_count = self
                                .gate34_candidate_counts
                                .values()
                                .copied()
                                .max()
                                .unwrap_or(1) as f32;
                            let mut max_mean = 1e-6f32;
                            let mut max_best = 1e-6f32;
                            let mut min_prompt_cos = f32::INFINITY;
                            let mut max_prompt_cos = f32::NEG_INFINITY;
                            let mut max_inverse_distance = 1e-6f32;
                            let probe_vec = tensor_to_vec_f32(&probe_normalized)
                                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                            let probe_64: Vec<f32> = probe_vec.into_iter().take(64).collect();
                            let prompt_vec_opt = self.prompt_vec.clone();
                            let mut records: Vec<Gate34CandidateRecord> = Vec::new();

                            for (id, count) in &self.gate34_candidate_counts {
                                let mean = self
                                    .gate34_candidate_margin_sum
                                    .get(id)
                                    .copied()
                                    .unwrap_or(0.0)
                                    / (*count as f32).max(1.0);
                                let best = self
                                    .gate34_candidate_best_margin
                                    .get(id)
                                    .copied()
                                    .unwrap_or(0.0);
                                max_mean = max_mean.max(mean);
                                max_best = max_best.max(best);

                                let mut distance = f32::MAX;
                                let mut prompt_cos = 0.0f32;
                                let mut motif_role: Option<String> = None;
                                let mut routing_safety_score: Option<f32> = None;
                                let mut injection_strength: Option<f32> = None;
                                let mut persistence_score: Option<f32> = None;
                                let mut readiness_score: Option<f32> = None;
                                let mut window_bias = 0.0f32;

                                if self.gate34_target_source == "motifs" {
                                    if let Some(motif) =
                                        self.runtime_motifs.iter().find(|m| m.motif_id == *id)
                                    {
                                        // Motif candidate distance uses the same 64D codec geometry
                                        // as latch distance: compress(probe) vs motif.raw_signature.
                                        if !motif.raw_signature.is_empty() {
                                            let probe_compressed = compress_tensor_to_dim(
                                                &probe,
                                                motif.raw_signature.len(),
                                            )
                                            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                                            if probe_compressed.len() == motif.raw_signature.len() {
                                                let mut dist_sq = 0.0f32;
                                                for i in 0..motif.raw_signature.len() {
                                                    let d = probe_compressed[i]
                                                        - motif.raw_signature[i];
                                                    dist_sq += d * d;
                                                }
                                                distance = dist_sq.sqrt();
                                            }
                                        }
                                        if let Some(prompt_vec) = &prompt_vec_opt {
                                            let anchor = motif
                                                .task_anchor_signature
                                                .as_ref()
                                                .unwrap_or(&motif.raw_signature);
                                            if prompt_vec.len() == anchor.len() {
                                                let mut dot = 0.0f32;
                                                let mut anchor_norm_sq = 0.0f32;
                                                for i in 0..anchor.len() {
                                                    dot += prompt_vec[i] * anchor[i];
                                                    anchor_norm_sq += anchor[i] * anchor[i];
                                                }
                                                let anchor_norm = anchor_norm_sq.sqrt();
                                                if anchor_norm > 1e-8 {
                                                    prompt_cos = dot / anchor_norm;
                                                } else {
                                                    self.prompt_similarity_unavailable = true;
                                                }
                                            } else {
                                                self.prompt_similarity_unavailable = true;
                                            }
                                        } else {
                                            self.prompt_similarity_unavailable = true;
                                        }
                                        motif_role = Some(motif.motif_role.clone());
                                        routing_safety_score = Some(motif.routing_safety_score);
                                        injection_strength = Some(motif.injection_strength);
                                        persistence_score = Some(motif.persistence_score);
                                        readiness_score = Some(motif.readiness_score);
                                        if id.contains("stable_window") {
                                            window_bias = 0.30;
                                        } else if id.contains("hinge_window") {
                                            window_bias = -0.50;
                                        }
                                    } else {
                                        self.prompt_similarity_unavailable = true;
                                    }
                                } else if self.gate34_target_source == "specialists" {
                                    if let Some(operator) = self
                                        .runtime_recovery_ops
                                        .iter()
                                        .find(|op| op.specialist_id == *id)
                                    {
                                        if !operator.raw_signature.is_empty() {
                                            let dim =
                                                operator.raw_signature.len().min(probe_64.len());
                                            if dim > 0 {
                                                let mut dist_sq = 0.0f32;
                                                for i in 0..dim {
                                                    let d = probe_64[i] - operator.raw_signature[i];
                                                    dist_sq += d * d;
                                                }
                                                distance = dist_sq.sqrt();
                                            }
                                        }
                                        if let Some(prompt_vec) = &prompt_vec_opt {
                                            if prompt_vec.len() == operator.raw_signature.len() {
                                                let mut dot = 0.0f32;
                                                let mut signature_norm_sq = 0.0f32;
                                                for i in 0..operator.raw_signature.len() {
                                                    dot +=
                                                        prompt_vec[i] * operator.raw_signature[i];
                                                    signature_norm_sq += operator.raw_signature[i]
                                                        * operator.raw_signature[i];
                                                }
                                                let signature_norm = signature_norm_sq.sqrt();
                                                if signature_norm > 1e-8 {
                                                    prompt_cos = dot / signature_norm;
                                                } else {
                                                    self.prompt_similarity_unavailable = true;
                                                }
                                            } else {
                                                self.prompt_similarity_unavailable = true;
                                            }
                                        } else {
                                            self.prompt_similarity_unavailable = true;
                                        }
                                        motif_role = Some(operator.role.clone());
                                        routing_safety_score = Some(
                                            (1.0 / (1.0 + operator.basin_variance * 100.0))
                                                .clamp(0.0, 1.0),
                                        );
                                        injection_strength = Some(
                                            (1.0 / (1.0 + operator.influence_radius * 10.0))
                                                .clamp(0.1, 1.0),
                                        );
                                        persistence_score = Some(operator.persistence_score);
                                        readiness_score = Some(operator.readiness_score);
                                        if operator.role.contains("structured_candidate") {
                                            window_bias = 0.20;
                                        } else if operator.role.contains("structured") {
                                            window_bias = 0.30;
                                        } else if operator.role.contains("neutral") {
                                            window_bias = -0.25;
                                        }
                                    } else {
                                        self.prompt_similarity_unavailable = true;
                                    }
                                } else {
                                    #[cfg(feature = "niodv4_bridge")]
                                    if let Some(registry) = &self.ghost_registry {
                                        if let Some(basin) = registry.find_basin(id) {
                                            let dim = basin.data.len().min(probe_64.len());
                                            if dim > 0 {
                                                let mut dist_sq = 0.0f32;
                                                for i in 0..dim {
                                                    let d = probe_64[i] - basin.data[i];
                                                    dist_sq += d * d;
                                                }
                                                distance = dist_sq.sqrt();
                                            }
                                            if let Some(prompt_vec) = &prompt_vec_opt {
                                                if prompt_vec.len() == basin.data.len() {
                                                    let mut dot = 0.0f32;
                                                    let mut basin_norm_sq = 0.0f32;
                                                    for i in 0..basin.data.len() {
                                                        dot += prompt_vec[i] * basin.data[i];
                                                        basin_norm_sq +=
                                                            basin.data[i] * basin.data[i];
                                                    }
                                                    let basin_norm = basin_norm_sq.sqrt();
                                                    if basin_norm > 1e-8 {
                                                        prompt_cos = dot / basin_norm;
                                                    } else {
                                                        self.prompt_similarity_unavailable = true;
                                                    }
                                                } else {
                                                    self.prompt_similarity_unavailable = true;
                                                }
                                            } else {
                                                self.prompt_similarity_unavailable = true;
                                            }
                                        } else {
                                            self.prompt_similarity_unavailable = true;
                                        }
                                    }
                                }

                                let inverse_distance = if distance.is_finite() {
                                    1.0 / (1.0 + distance.max(0.0))
                                } else {
                                    0.0
                                };
                                min_prompt_cos = min_prompt_cos.min(prompt_cos);
                                max_prompt_cos = max_prompt_cos.max(prompt_cos);
                                max_inverse_distance = max_inverse_distance.max(inverse_distance);

                                records.push(Gate34CandidateRecord {
                                    candidate_ghost_id: id.clone(),
                                    gate34_target_source: self.gate34_target_source.clone(),
                                    gate34_target_kind: self.gate34_target_kind().to_string(),
                                    count: *count,
                                    count_ratio: *count as f32 / max_count.max(1.0),
                                    mean_margin: mean,
                                    best_margin: best,
                                    distance_to_probe_at_acquire: if distance.is_finite() {
                                        distance
                                    } else {
                                        0.0
                                    },
                                    prompt_ghost_cosine: prompt_cos,
                                    prompt_ghost_cosine_norm: 0.0,
                                    inverse_distance,
                                    inverse_distance_norm: 0.0,
                                    mean_margin_norm: 0.0,
                                    best_margin_norm: 0.0,
                                    motif_role,
                                    routing_safety_score,
                                    injection_strength,
                                    persistence_score,
                                    readiness_score,
                                    window_bias,
                                    acquisition_score: f32::MIN,
                                    selected: false,
                                });
                            }

                            let mut best_id: Option<String> = None;
                            let mut best_score = f32::MIN;
                            for rec in &mut records {
                                rec.mean_margin_norm = rec.mean_margin / max_mean.max(1e-6);
                                rec.best_margin_norm = rec.best_margin / max_best.max(1e-6);
                                let prompt_rel = if self.prompt_similarity_unavailable
                                    || !min_prompt_cos.is_finite()
                                    || !max_prompt_cos.is_finite()
                                {
                                    0.0
                                } else {
                                    let denom = (max_prompt_cos - min_prompt_cos).max(1e-6);
                                    ((rec.prompt_ghost_cosine - min_prompt_cos) / denom)
                                        .clamp(0.0, 1.0)
                                };
                                rec.prompt_ghost_cosine_norm = prompt_rel;
                                rec.inverse_distance_norm =
                                    rec.inverse_distance / max_inverse_distance.max(1e-6);
                                let stability_score = rec.count_ratio
                                    + rec.mean_margin_norm
                                    + 0.5 * rec.best_margin_norm
                                    + rec.inverse_distance_norm;
                                rec.acquisition_score = if self.gate34_target_source == "motifs" {
                                    stability_score * (0.5 + prompt_rel)
                                        + self.bridge_prompt_weight * prompt_rel
                                        + rec.routing_safety_score.unwrap_or(0.0) * 0.08
                                        + rec.persistence_score.unwrap_or(0.0) * 0.06
                                        + rec.readiness_score.unwrap_or(0.0) * 0.06
                                        + rec.window_bias * 0.15
                                } else {
                                    stability_score
                                        + self.bridge_prompt_weight * rec.prompt_ghost_cosine_norm
                                };
                                if rec.acquisition_score > best_score {
                                    best_score = rec.acquisition_score;
                                    best_id = Some(rec.candidate_ghost_id.clone());
                                }
                            }

                            if let Some(target_id) = best_id.clone() {
                                for rec in &mut records {
                                    rec.selected = rec.candidate_ghost_id == target_id;
                                }
                                self.gate34_acquisition_candidates = records.clone();
                                println!(
                                    "[GATE34_ACQUIRE] {}",
                                    serde_json::json!({
                                        "req_id": self.current_run_id,
                                        "prompt_hash": self.last_prompt_hash,
                                        "target_source": self.gate34_target_source.clone(),
                                        "prompt_embedding_source": self.prompt_embedding_source,
                                        "prompt_vec_norm": self.prompt_vec_norm,
                                        "prompt_similarity_unavailable": self.prompt_similarity_unavailable,
                                        "candidates": records,
                                    })
                                );
                                let target_vec_opt = match self.gate34_target_source.as_str() {
                                    "motifs" => self
                                        .runtime_motifs
                                        .iter()
                                        .find(|m| m.motif_id == target_id)
                                        // Use the live-hidden remapped motif vector for latch distance geometry.
                                        .filter(|m| m.live_hidden_remapped)
                                        .map(|m| m.vector.clone()),
                                    "specialists" => self
                                        .runtime_recovery_ops
                                        .iter()
                                        .find(|op| op.specialist_id == target_id)
                                        .map(|op| op.vector.clone()),
                                    _ => self
                                        .get_bridge_ghost_vector(&target_id, device)
                                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?,
                                };
                                if let Some(target_vec) = target_vec_opt {
                                    self.gate34_target_ghost_id = Some(target_id.clone());
                                    self.gate34_target_specialist_id =
                                        if self.gate34_target_source == "specialists" {
                                            Some(target_id.clone())
                                        } else {
                                            None
                                        };
                                    self.gate34_target_motif_id =
                                        if self.gate34_target_source == "motifs" {
                                            Some(target_id.clone())
                                        } else if self.gate34_target_source == "specialists" {
                                            self.runtime_recovery_ops
                                                .iter()
                                                .find(|op| op.specialist_id == target_id)
                                                .map(|op| op.motif_id.clone())
                                        } else {
                                            None
                                        };
                                    self.gate34_target_vector = Some(target_vec.clone());
                                    self.gate34_target_acquired_step = step_i64;
                                    self.gate34_target_margin_at_acquire = self.last_route_margin;
                                    let target_flat = target_vec
                                        .flatten_all()?
                                        .to_dtype(DType::F32)?
                                        .to_device(device)?;
                                    if target_flat.dims() != probe.dims() {
                                        self.gate34_phase = Gate34Phase::Released;
                                        self.gate34_release_reason =
                                            Some("dim_mismatch".to_string());
                                        self.last_projection_strategy =
                                            "gate34_released:dim_mismatch".to_string();
                                    } else {
                                        let dist = if self.gate34_target_source == "motifs" {
                                            if let Some(motif) = self
                                                .gate34_target_ghost_id
                                                .as_ref()
                                                .and_then(|id| {
                                                    self.runtime_motifs
                                                        .iter()
                                                        .find(|m| m.motif_id == *id)
                                                })
                                            {
                                                // Motif raw_signature is canonical 64D geometry from
                                                // niodv4/scripts/build_60k_aligned_runtime_bridge.py and
                                                // niodv4/production_candidate/60k/bridge_60k_aligned/live_hidden_60k_raw64_bridge.json.
                                                let probe_64 = compress_tensor_to_dim(
                                                    &probe,
                                                    motif.raw_signature.len(),
                                                )
                                                .map_err(|e| {
                                                    candle_core::Error::Msg(e.to_string())
                                                })?;
                                                if motif.raw_signature.is_empty()
                                                    || probe_64.len() != motif.raw_signature.len()
                                                {
                                                    (target_flat - probe.clone())?
                                                        .sqr()?
                                                        .sum_all()?
                                                        .sqrt()?
                                                        .to_scalar::<f32>()?
                                                } else {
                                                    let mut dist_sq = 0.0f32;
                                                    for i in 0..motif.raw_signature.len() {
                                                        let d =
                                                            probe_64[i] - motif.raw_signature[i];
                                                        dist_sq += d * d;
                                                    }
                                                    dist_sq.sqrt()
                                                }
                                            } else {
                                                (target_flat - probe.clone())?
                                                    .sqr()?
                                                    .sum_all()?
                                                    .sqrt()?
                                                    .to_scalar::<f32>()?
                                            }
                                        } else if self.gate34_target_source == "specialists" {
                                            if let Some(operator) = self
                                                .gate34_target_specialist_id
                                                .as_ref()
                                                .and_then(|id| {
                                                    self.runtime_recovery_ops
                                                        .iter()
                                                        .find(|op| op.specialist_id == *id)
                                                })
                                            {
                                                let probe_64 = compress_tensor_to_dim(
                                                    &probe,
                                                    operator.raw_signature.len(),
                                                )
                                                .map_err(|e| {
                                                    candle_core::Error::Msg(e.to_string())
                                                })?;
                                                if operator.raw_signature.is_empty()
                                                    || probe_64.len()
                                                        != operator.raw_signature.len()
                                                {
                                                    (target_flat - probe.clone())?
                                                        .sqr()?
                                                        .sum_all()?
                                                        .sqrt()?
                                                        .to_scalar::<f32>()?
                                                } else {
                                                    let mut dist_sq = 0.0f32;
                                                    for i in 0..operator.raw_signature.len() {
                                                        let d =
                                                            probe_64[i] - operator.raw_signature[i];
                                                        dist_sq += d * d;
                                                    }
                                                    dist_sq.sqrt()
                                                }
                                            } else {
                                                (target_flat - probe.clone())?
                                                    .sqr()?
                                                    .sum_all()?
                                                    .sqrt()?
                                                    .to_scalar::<f32>()?
                                            }
                                        } else {
                                            (target_flat - probe.clone())?
                                                .sqr()?
                                                .sum_all()?
                                                .sqrt()?
                                                .to_scalar::<f32>()?
                                        };
                                        self.gate34_target_distance_at_acquire = dist;
                                        self.gate34_current_target_distance = dist;
                                        let warmup_count = self
                                            .gate34_candidate_counts
                                            .get(&target_id)
                                            .copied()
                                            .unwrap_or(0)
                                            .max(1)
                                            as f32;
                                        let warmup_sum = self
                                            .gate34_candidate_distance_sum
                                            .get(&target_id)
                                            .copied()
                                            .unwrap_or(dist);
                                        let warmup_sq_sum = self
                                            .gate34_candidate_distance_sq_sum
                                            .get(&target_id)
                                            .copied()
                                            .unwrap_or(dist * dist);
                                        let warmup_mean = warmup_sum / warmup_count;
                                        let variance = (warmup_sq_sum / warmup_count
                                            - warmup_mean * warmup_mean)
                                            .max(0.0);
                                        self.gate34_target_warmup_distance_min = self
                                            .gate34_candidate_distance_min
                                            .get(&target_id)
                                            .copied()
                                            .unwrap_or(dist);
                                        self.gate34_target_warmup_distance_max = self
                                            .gate34_candidate_distance_max
                                            .get(&target_id)
                                            .copied()
                                            .unwrap_or(dist);
                                        self.gate34_target_warmup_distance_mean = warmup_mean;
                                        self.gate34_target_warmup_distance_std = variance.sqrt();
                                        self.gate34_last_distance_drift_score = 0.0;
                                        self.gate34_last_distance_limit_ratio = dist;
                                        self.gate34_last_distance_limit_warmup =
                                            self.gate34_target_warmup_distance_max;
                                        self.gate34_last_distance_gate_mode =
                                            "ratio_only".to_string();
                                        self.gate34_phase = Gate34Phase::Latched;
                                        self.gate34_held_step_count = 0;
                                        self.gate34_bad_margin_count = 0;
                                        self.gate34_bad_distance_count = 0;
                                        self.gate34_release_reason = None;
                                        self.last_projection_strategy =
                                            "gate34_latch_acquired".to_string();
                                    }
                                } else {
                                    self.gate34_phase = Gate34Phase::Released;
                                    self.gate34_release_reason =
                                        Some("no_target_vector".to_string());
                                    self.last_projection_strategy =
                                        "gate34_released:no_target_vector".to_string();
                                }
                            }
                        }
                    }
                }
            }

            if self.gate34_phase == Gate34Phase::Latched {
                let current_target_vec = match self.gate34_target_source.as_str() {
                    "motifs" => {
                        if let Some(target_id) = &self.gate34_target_ghost_id {
                            self.runtime_motifs
                                .iter()
                                .find(|m| m.motif_id == *target_id)
                                .filter(|m| m.live_hidden_remapped)
                                .map(|m| m.vector.clone())
                                .or_else(|| self.gate34_target_vector.clone())
                        } else {
                            self.gate34_target_vector.clone()
                        }
                    }
                    "specialists" => {
                        if let Some(target_id) = &self.gate34_target_specialist_id {
                            self.runtime_recovery_ops
                                .iter()
                                .find(|op| op.specialist_id == *target_id)
                                .map(|op| op.vector.clone())
                                .or_else(|| self.gate34_target_vector.clone())
                        } else {
                            self.gate34_target_vector.clone()
                        }
                    }
                    _ => self.gate34_target_vector.clone(),
                };
                if let Some(target_vec) = current_target_vec {
                    self.gate34_target_vector = Some(target_vec.clone());
                    let target_flat = target_vec
                        .flatten_all()?
                        .to_dtype(DType::F32)?
                        .to_device(device)?;
                    if target_flat.dims() != probe.dims() {
                        self.gate34_phase = Gate34Phase::Released;
                        self.gate34_release_reason = Some("dim_mismatch".to_string());
                        self.last_projection_strategy = "gate34_released:dim_mismatch".to_string();
                        self.last_ghost_pull_delta_norm = 0.0;
                        self.last_intervention_applied = false;
                    } else {
                        let delta = (target_flat - probe.clone())?;
                        let current_target_distance = if self.gate34_target_source == "motifs" {
                            if let Some(motif) =
                                self.gate34_target_ghost_id.as_ref().and_then(|id| {
                                    self.runtime_motifs.iter().find(|m| m.motif_id == *id)
                                })
                            {
                                let probe_64 =
                                    compress_tensor_to_dim(&probe, motif.raw_signature.len())
                                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                                if motif.raw_signature.is_empty()
                                    || probe_64.len() != motif.raw_signature.len()
                                {
                                    delta.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?
                                } else {
                                    let mut dist_sq = 0.0f32;
                                    for i in 0..motif.raw_signature.len() {
                                        let d = probe_64[i] - motif.raw_signature[i];
                                        dist_sq += d * d;
                                    }
                                    dist_sq.sqrt()
                                }
                            } else {
                                delta.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?
                            }
                        } else if self.gate34_target_source == "specialists" {
                            if let Some(operator) =
                                self.gate34_target_specialist_id.as_ref().and_then(|id| {
                                    self.runtime_recovery_ops
                                        .iter()
                                        .find(|op| op.specialist_id == *id)
                                })
                            {
                                let probe_64 =
                                    compress_tensor_to_dim(&probe, operator.raw_signature.len())
                                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                                if operator.raw_signature.is_empty()
                                    || probe_64.len() != operator.raw_signature.len()
                                {
                                    delta.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?
                                } else {
                                    let mut dist_sq = 0.0f32;
                                    for i in 0..operator.raw_signature.len() {
                                        let d = probe_64[i] - operator.raw_signature[i];
                                        dist_sq += d * d;
                                    }
                                    dist_sq.sqrt()
                                }
                            } else {
                                delta.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?
                            }
                        } else {
                            delta.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?
                        };
                        self.gate34_current_target_distance = current_target_distance;
                        if self.last_route_margin < self.gate34_release_margin_floor {
                            self.gate34_bad_margin_count =
                                self.gate34_bad_margin_count.saturating_add(1);
                        } else {
                            self.gate34_bad_margin_count = 0;
                        }
                        let motif_drift_mult = if self.gate34_target_source == "motifs" {
                            // RC data shows stable-window turns can carry meaningful hinge drift;
                            // scale the distance gate by observed live drift instead of a fixed multiplier.
                            1.0 + self.current_hinge_window_drift().clamp(0.0, 1.0) * 3.0
                        } else {
                            1.0
                        };
                        let distance_limit_ratio = (self.gate34_target_distance_at_acquire
                            * self.gate34_release_distance_mult
                            * motif_drift_mult)
                            .max(1e-6);
                        let warmup_mean = if self.gate34_target_warmup_distance_mean > 1e-6 {
                            self.gate34_target_warmup_distance_mean
                        } else {
                            self.gate34_target_distance_at_acquire
                        };
                        let warmup_std = self.gate34_target_warmup_distance_std.max(0.0);
                        let warmup_std_floor = (warmup_mean * 0.10).max(0.005);
                        let warmup_std_eff = warmup_std.max(warmup_std_floor);
                        let distance_limit_warmup =
                            warmup_mean + 3.0 * warmup_std_eff * motif_drift_mult;
                        let drift_score =
                            (current_target_distance - warmup_mean) / warmup_std_eff.max(1e-6);
                        let acquire_stable_floor = 0.10f32;
                        let use_ratio_gate =
                            self.gate34_target_distance_at_acquire >= acquire_stable_floor;
                        let distance_bad = if use_ratio_gate {
                            current_target_distance > distance_limit_ratio
                                && current_target_distance > distance_limit_warmup
                        } else {
                            current_target_distance > distance_limit_warmup
                        };
                        self.gate34_last_distance_drift_score = drift_score;
                        self.gate34_last_distance_limit_ratio = distance_limit_ratio;
                        self.gate34_last_distance_limit_warmup = distance_limit_warmup;
                        self.gate34_last_distance_gate_mode = if use_ratio_gate {
                            "ratio_and_warmup".to_string()
                        } else {
                            "warmup_only".to_string()
                        };
                        if distance_bad {
                            self.gate34_bad_distance_count =
                                self.gate34_bad_distance_count.saturating_add(1);
                        } else {
                            self.gate34_bad_distance_count = 0;
                        }

                        let release_reason = if self.gate34_bad_margin_count
                            >= self.gate34_release_patience
                        {
                            Some("margin_floor")
                        } else if self.gate34_bad_distance_count >= self.gate34_release_patience {
                            Some("distance_drift")
                        } else if self.gate34_held_step_count >= self.gate34_hold_steps {
                            Some("hold_expired")
                        } else {
                            None
                        };

                        if let Some(reason) = release_reason {
                            self.gate34_phase = Gate34Phase::Released;
                            self.gate34_release_reason = Some(reason.to_string());
                            self.last_projection_strategy = format!("gate34_released:{}", reason);
                            self.last_ghost_pull_delta_norm = 0.0;
                            self.last_intervention_applied = false;
                        } else {
                            let raw_norm = current_target_distance;
                            if raw_norm > 1e-6 {
                                let base_clamp = self.bridge_influence_smoke_clamp.clamp(0.0, 0.02);
                                let margin_floor = if self.gate34_target_source == "motifs" {
                                    let window_floor: f32 = self
                                        .gate34_target_ghost_id
                                        .as_ref()
                                        .map(|id| {
                                            if id.contains("stable_window") {
                                                0.75f32
                                            } else if id.contains("hinge_window") {
                                                0.55f32
                                            } else {
                                                0.60f32
                                            }
                                        })
                                        .unwrap_or(0.60f32);
                                    let safety_floor = self
                                        .gate34_target_ghost_id
                                        .as_ref()
                                        .and_then(|id| {
                                            self.runtime_motifs.iter().find(|m| m.motif_id == *id)
                                        })
                                        .map(|m| {
                                            (0.45f32
                                                + 0.35f32 * m.routing_safety_score.clamp(0.0, 1.0))
                                            .clamp(0.45f32, 0.85f32)
                                        })
                                        .unwrap_or(0.60f32);
                                    window_floor.max(safety_floor)
                                } else if self.gate34_target_source == "specialists" {
                                    self.gate34_target_specialist_id
                                        .as_ref()
                                        .and_then(|id| {
                                            self.runtime_recovery_ops
                                                .iter()
                                                .find(|op| op.specialist_id == *id)
                                        })
                                        .map(|op| {
                                            if op.role.contains("structured") {
                                                0.55
                                            } else {
                                                0.35
                                            }
                                        })
                                        .unwrap_or(0.40)
                                } else {
                                    0.25
                                };
                                let margin_factor =
                                    (self.last_route_margin / 0.0010).clamp(margin_floor, 1.0);
                                let hold_decay = if self.gate34_held_step_count < 32 {
                                    1.0
                                } else {
                                    0.75
                                };
                                let injection_mult = if self.gate34_target_source == "motifs" {
                                    self.gate34_target_ghost_id
                                        .as_ref()
                                        .and_then(|id| {
                                            self.runtime_motifs.iter().find(|m| m.motif_id == *id)
                                        })
                                        .map(|m| m.injection_strength.clamp(0.4, 1.0))
                                        .unwrap_or(1.0)
                                } else if self.gate34_target_source == "specialists" {
                                    self.gate34_target_specialist_id
                                        .as_ref()
                                        .and_then(|id| {
                                            self.runtime_recovery_ops
                                                .iter()
                                                .find(|op| op.specialist_id == *id)
                                        })
                                        .map(|op| {
                                            let basin_term = (1.0
                                                / (1.0 + op.basin_variance * 100.0))
                                                .clamp(0.2, 1.0);
                                            let radius_term = if op.influence_radius > 0.0 {
                                                (1.0 / (1.0 + op.influence_radius * 10.0))
                                                    .clamp(0.2, 1.0)
                                            } else {
                                                1.0
                                            };
                                            (basin_term
                                                * radius_term
                                                * op.readiness_score.max(0.25))
                                            .clamp(0.25, 1.0)
                                        })
                                        .unwrap_or(0.75)
                                } else {
                                    1.0
                                };
                                let max_norm =
                                    base_clamp * margin_factor * hold_decay * injection_mult;
                                let scale = if raw_norm > max_norm {
                                    max_norm / raw_norm
                                } else {
                                    1.0
                                };
                                let scale_t = Tensor::from_vec(vec![scale], (1,), device)?;
                                let latch_delta = delta.broadcast_mul(&scale_t)?;
                                let clamped_norm = (raw_norm * scale).min(max_norm);
                                probe_force = (probe_force + latch_delta)?;
                                self.last_ghost_pull_delta_norm = clamped_norm;
                                self.last_intervention_applied = clamped_norm > 1e-6;
                                self.last_projection_strategy = "gate34_latch".to_string();
                                self.gate34_held_step_count =
                                    self.gate34_held_step_count.saturating_add(1);
                                self.gate34_intervention_count =
                                    self.gate34_intervention_count.saturating_add(1);
                            } else {
                                self.last_ghost_pull_delta_norm = 0.0;
                                self.last_intervention_applied = false;
                                self.last_projection_strategy =
                                    "gate34_latch:zero_delta".to_string();
                            }
                        }
                    }
                } else {
                    self.gate34_phase = Gate34Phase::Released;
                    self.gate34_release_reason = Some("no_target_vector".to_string());
                    self.last_projection_strategy = "gate34_released:no_target_vector".to_string();
                    self.last_ghost_pull_delta_norm = 0.0;
                    self.last_intervention_applied = false;
                }
            }
        }

        probe_force = self
            .apply_specialist_worker_influence(&probe, selected_worker_idx, probe_force, layer_idx)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // Per-token cadence: cooldown decrements once per token, not once per layer.
        #[cfg(feature = "niodv4_bridge")]
        if self.bridge_enabled
            && self.last_ghost_switch_cooldown_remaining > 0
            && self.last_bridge_cooldown_step != self.current_step as i64
        {
            self.last_bridge_cooldown_step = self.current_step as i64;
            self.last_ghost_switch_cooldown_remaining =
                self.last_ghost_switch_cooldown_remaining.saturating_sub(1);
        }

        // 2. PINN Manifold Conservation (The "Rail")
        // Enforce that the particle stays on the semantic manifold (hypersphere shell)
        // L_cons = (||x|| - R)^2 => F_cons = -grad(L) = -2 * (||x|| - R) * x_hat
        // Expected R ~ sqrt(hidden_dim) for RMSNorm
        if self.params.pinn_enabled {
            // Simplified: Always valid if code is active
            // We reuse probe_norm_scalar calculated earlier
            let target_r = (self.hidden_dim as f64).sqrt();
            let current_r = probe_norm_scalar.to_scalar::<f32>()? as f64; // Extract F32, cast to f64 for precision

            // Stiffness k
            let k_cons = self.params.pinn_stiffness; // Moderate restoration

            let diff = current_r - target_r;
            // Force directs back to shell
            let magnitude = -k_cons * diff; // f64

            // Direction is probe_normalized (F32)
            // Must cast magnitude to f32
            let mag_tensor = Tensor::new(magnitude as f32, device)?;
            let f_cons = probe_normalized.broadcast_mul(&mag_tensor)?;
            probe_force = (probe_force + f_cons)?;
        }

        // 3. Goal Attractor
        if let Some(goal) = &self.goal_embedding {
            let goal_dev = goal.to_device(device)?.to_dtype(DType::F32)?;
            let goal_dim = goal_dev.dim(0)?;
            let probe_dim = probe.dim(0)?;

            if goal_dim == probe_dim {
                if layer_idx > 15 {
                    let r_goal = (&goal_dev - &probe)?;
                    // Use gravity_well arg to scale the goal strength (Centripetal Force)
                    let goal_strength = (self.dynamic_gravity as f64
                        * self.params.gravity_well_strength
                        * 1000.0) as f32;
                    let gs_t = Tensor::new(goal_strength, device)?;
                    let goal_force = r_goal.broadcast_mul(&gs_t)?;
                    self.last_goal_mag = goal_force
                        .sqr()?
                        .sum_all()?
                        .sqrt()?
                        .to_scalar::<f32>()
                        .unwrap_or(0.0);
                    probe_force = (probe_force + goal_force)?;
                }
            }
        }

        // 3.5 Black Hole Repulsion (The "Niodoo" Shield)
        // Applies repulsive force to specific forbidden concepts
        if !self.black_hole_embeddings.is_empty() && layer_idx > 10 {
            let repulsion_strength = self.dynamic_repulsion as f32; // e.g. -1.3 (Elastic)
            let mut total_repulsion_mag: f32 = 0.0; // TELEMETRY accumulator

            if repulsion_strength.abs() > 1e-6 {
                let rep_t = Tensor::new(repulsion_strength * 10.0, device)?; // Scale up a bit for impact
                for bh_emb in &self.black_hole_embeddings {
                    let bh_dev = bh_emb.to_device(device)?.to_dtype(DType::F32)?;
                    // F = -G * m1*m2 / r^2  (repulsion is negative G)

                    // Vector R = bh_pos - probe
                    let r_vec = (&bh_dev - &probe)?;
                    let dist_sq = r_vec.sqr()?.sum_all()?;
                    let dist_scalar = dist_sq.sqrt()?.to_scalar::<f32>()?;

                    // Only repel if close (short range force)
                    if dist_scalar < 5.0 {
                        let epsilon = Tensor::new(1e-3f32, device)?;
                        let denom = (dist_sq + epsilon)?;

                        // Force vector
                        let force_mag = rep_t.broadcast_div(&denom)?; // negative value
                        let force = r_vec.broadcast_mul(&force_mag)?;

                        // TELEMETRY: Accumulate repulsion magnitude
                        total_repulsion_mag += force
                            .sqr()?
                            .sum_all()?
                            .sqrt()?
                            .to_scalar::<f32>()
                            .unwrap_or(0.0);

                        probe_force = (probe_force + force)?;
                    }
                }
            }

            // TELEMETRY: Store total repulsion
            self.last_repulsion_mag = total_repulsion_mag;
        }

        // =====================================================================
        // PHASE 2: ORBITAL STEERING (The "Double Rainbow" Logic)
        // =====================================================================
        if self.orbital_active && self.params.orbit_speed > 0.0 && layer_idx > 15 {
            // A. Calculate "Sun" (Context Centroid)
            let head_len = self.sentence_history.len();
            if head_len > 0 {
                let lookback = 20.min(head_len);
                let mut center_of_mass_t = Tensor::zeros((self.hidden_dim,), DType::F32, device)?;
                let mut count = 0.0f32;

                for i in 0..lookback {
                    let idx = head_len - 1 - i;
                    if let Some(p) = self.sentence_history.get(idx) {
                        let pos = p
                            .position
                            .to_device(device)?
                            .to_dtype(DType::F32)?
                            .flatten_all()?;
                        center_of_mass_t = (center_of_mass_t + pos)?;
                        count += 1.0;
                    }
                }

                // Anchor to Prompt (First particle)
                if let Some(first) = self.sentence_history.front() {
                    let anchor = first
                        .position
                        .to_device(device)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?;
                    let strength =
                        (self.params.gravity * self.params.gravity_well_strength * 1000.0) as f32;
                    let s_t = Tensor::new(strength, device)?;
                    center_of_mass_t = (center_of_mass_t + anchor.broadcast_mul(&s_t)?)?;
                    count += strength;
                }

                if count > 0.0 {
                    let scale = Tensor::new(1.0 / count, device)?;
                    center_of_mass_t = center_of_mass_t.broadcast_mul(&scale)?;
                }

                // B. Gravity Vector (Sun - Probe)
                let r_vec = (&center_of_mass_t - &probe)?;
                let dist_sq = r_vec.sqr()?.sum_all()?;
                let dist_scalar = dist_sq.to_scalar::<f32>()?;

                // C. Orbital Kick
                // If far enough away (prevent singularity)
                if dist_scalar > 1.0 {
                    // Symplectic Kick: Tangential Force
                    // For high dimensional orbit, we push orthogonal to gravity?
                    // Or simply use the "Lure" logic (Attraction)?
                    // Niodoo Protocol says: "Lure".
                    // But strictly, "Orbit" means tangential velocity.
                    // Here we implement a simpler "Gravitational Assist"

                    // 1. Attraction (Centripetal)
                    // F = G * M / r^2
                    // AMPLIFIED: Forces were O(0.001), hidden states are O(20-50).
                    //            Scaling by 10000x to be LLM-visible.
                    let g_force_mag = (self.params.gravity * 10000.0) as f32 / dist_scalar;
                    let g_force_t = Tensor::new(g_force_mag, device)?;
                    let g_force = r_vec.broadcast_mul(&g_force_t)?;

                    // 2. Tangential Component (The "Orbit" Speed)
                    // We cheat: Add a spiral component orthogonal to R?
                    // Hard in 4096D.
                    // Instead, we just ADD to the probe_force directly.

                    // Use orbit_speed as a direct multiplier for the "Assist"
                    // AMPLIFIED: Scaling by 100x
                    let assist_mag = self.params.orbit_speed as f32 * 100.0;
                    let assist_t = Tensor::new(assist_mag, device)?;
                    let assist_force = g_force.broadcast_mul(&assist_t)?;

                    probe_force = (probe_force + assist_force)?;
                }
            }
        }

        // 3.75 Motif Memory + Reflexive Recovery
        // Runtime bridge forces live in the same continuous-space update as the
        // existing physics terms so logits still emerge from the shifted manifold.
        if self.specialist_memory_workers_mode != SpecialistMemoryWorkerMode::Influence
            && (!self.runtime_motifs.is_empty() || !self.runtime_recovery_ops.is_empty())
        {
            let (mut motif_force, mut recovery_force) =
                self.compute_bridge_forces(&probe, &probe_normalized, activation_gate, layer_idx)?;
            // DEEP_DIVE_ROADMAP P2-A: per-layer physics blend mask. Deep
            // layers (defaults to "off"; configured via
            // --physics-blend-deep-layer-from N --physics-blend-deep-layer-multiplier M)
            // get bridge forces scaled by M — protecting task-specific
            // route geometry that uniform aggressive blending overwrites.
            // For Llama-3.1-8B (32 layers): from=28 protects layers 28-31.
            if self.physics_blend_deep_layer_from > 0
                && layer_idx >= self.physics_blend_deep_layer_from
                && self.physics_blend_deep_layer_multiplier < 1.0
            {
                let m = Tensor::new(self.physics_blend_deep_layer_multiplier, device)?;
                motif_force = motif_force.broadcast_mul(&m)?;
                recovery_force = recovery_force.broadcast_mul(&m)?;
                self.physics_blend_deep_layer_mask_count += 1;
            }
            probe_force = (probe_force + motif_force)?;
            probe_force = (probe_force + recovery_force)?;
            // DEEP_DIVE_ROADMAP P2-C: autonomic physics adaptation. After
            // the bridge force is applied, sample the combined magnitude
            // (last_motif_mag + last_recovery_mag are populated inside
            // compute_bridge_forces) into a rolling window. When the
            // window saturates and the mean exceeds the threshold, scale
            // motif/recovery_force_scale down by 0.9× per check; when the
            // mean falls below 0.5× the threshold, restore by 1.05× up to
            // the originals. `0.0` threshold disables. The clamp prevents
            // runaway oscillation.
            if self.autonomic_physics_force_threshold > 0.0
                && self.autonomic_physics_window_size > 0
            {
                let combined_mag = self.last_motif_mag + self.last_recovery_mag;
                self.autonomic_physics_force_window.push_back(combined_mag);
                while self.autonomic_physics_force_window.len() > self.autonomic_physics_window_size
                {
                    self.autonomic_physics_force_window.pop_front();
                }
                if self.autonomic_physics_force_window.len() >= self.autonomic_physics_window_size {
                    let mean: f32 = self.autonomic_physics_force_window.iter().sum::<f32>()
                        / self.autonomic_physics_force_window.len() as f32;
                    let thr = self.autonomic_physics_force_threshold;
                    if mean > thr {
                        self.motif_force_scale = (self.motif_force_scale * 0.9).max(0.05);
                        self.recovery_force_scale = (self.recovery_force_scale * 0.9).max(0.05);
                        self.autonomic_physics_scale_down_count += 1;
                    } else if mean < thr * 0.5 {
                        let m_origin = self.autonomic_physics_motif_scale_origin;
                        let r_origin = self.autonomic_physics_recovery_scale_origin;
                        if self.motif_force_scale < m_origin {
                            self.motif_force_scale = (self.motif_force_scale * 1.05).min(m_origin);
                            self.autonomic_physics_scale_up_count += 1;
                        }
                        if self.recovery_force_scale < r_origin {
                            self.recovery_force_scale =
                                (self.recovery_force_scale * 1.05).min(r_origin);
                        }
                    }
                }
            }
        }

        if self.secret_sauce_steps_remaining > 0 {
            let restore_force = self.compute_secret_sauce_restore_force(&probe, layer_idx)?;
            probe_force = (probe_force + restore_force)?;
        }

        // 4. Langevin Dynamics
        let dt = self.params.dt as f64;
        let mu = self.params.mu;
        let sigma = self.params.sigma;

        let drift_scalar = (mu * dt) as f32;
        let drift_t = Tensor::new(drift_scalar, device)?;
        let drift = probe_force.broadcast_mul(&drift_t)?;

        let noise = Tensor::randn(0.0f32, 1.0f32, probe.shape(), device)?;
        let diffusion_scalar = (sigma * (2.0 * dt).sqrt()) as f32;
        let diff_t = Tensor::new(diffusion_scalar, device)?;
        let diffusion = noise.broadcast_mul(&diff_t)?;
        let mut delta_probe = (drift + diffusion)?;

        // 4. Momentum Injection
        let momentum_alpha = 0.15;
        if let Some(_buffer) = &self.momentum_buffer {
            // Retrieve last delta for THIS layer
            // We need to store [hidden] shaped deltas in current_surprisals or charge_tensor?
            // Actually `self.charge_tensor` is usually universe embeddings.
            // `momentum_buffer` field in struct seems unused/undefined contextually in original code logic?
            // Using `last_deltas` map for momentum history

            if let Some(last_delta) = self.last_deltas.get(&layer_idx) {
                let last_delta_dev = last_delta.to_device(device)?.to_dtype(DType::F32)?;
                // Ensure shape match (might be [1,1,hidden] from previous steps)
                let last_delta_flat = last_delta_dev.flatten_all()?;

                // Clean NaNs
                let sq = last_delta_flat.sqr()?.sum_all()?.to_scalar::<f32>()?;
                let safe_last = if sq.is_nan() || sq > 1e6 {
                    last_delta_flat.zeros_like()?
                } else {
                    last_delta_flat
                };

                let alpha_f32 = momentum_alpha as f32;
                let one_minus_alpha_t = Tensor::new(1.0 - alpha_f32, device)?;
                let alpha_t = Tensor::new(alpha_f32, device)?;
                let delta_calculated = (delta_probe.broadcast_mul(&one_minus_alpha_t)?
                    + safe_last.broadcast_mul(&alpha_t)?)?;
                delta_probe = delta_calculated;
            }
        }

        // 5. Momentum Update
        // Re-integrate Lorentz Boost and Atomic Simulation
        let momentum = self.params.momentum;
        if let Some(last_full) = self.last_deltas.get(&layer_idx) {
            let mut last_dev = last_full.to_device(device)?.to_dtype(DType::F32)?;
            // If last_dev is [batch, seq, hidden], we extract the slice for momentum calculation on the probe
            if last_dev.rank() > 1 && last_dev.dim(1).unwrap_or(0) > 0 {
                last_dev = last_dev.i((.., last_dev.dim(1)? - 1, ..))?.flatten_all()?;
            }

            let nan_check = last_dev.sqr()?.sum_all()?.to_scalar::<f32>()?;
            if nan_check.is_nan() {
                last_dev = probe.zeros_like()?;
            }

            // Lorentz Boost (relativistic stability)
            let lorentz_boost = self.compute_lorentz_boost(layer_idx)?;

            // Berkeley atomic
            let mut last_slice = last_dev.clone();
            if let Some(deepmd) = &self.deepmd_kit {
                last_slice = deepmd
                    .simulate_atomic(&last_slice)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            }

            // Explicit Type Handling for Compilation Safety
            let m_f32 = momentum as f32;
            let m_t = Tensor::new(m_f32, device)?;
            let one_minus_m_t = Tensor::new(1.0 - m_f32, device)?;

            let term1 = last_slice.broadcast_mul(&m_t)?;
            let mut term2 = delta_probe.broadcast_mul(&one_minus_m_t)?;

            // Safety on term2 before norm
            let t2_sq = term2.sqr()?.sum_all()?.to_scalar::<f32>()?;
            if t2_sq.is_nan() {
                println!(" [WARN] NaN in TERM2 at layer {} - zeroing", layer_idx);
                term2 = term2.zeros_like()?;
            }

            let delta_norm = t2_sq.sqrt();
            if delta_norm > 50.0 {
                let scale_t = Tensor::new((50.0 / delta_norm as f64) as f32, device)?;
                term2 = term2.broadcast_mul(&scale_t)?;
            }

            // Non-Reciprocal
            let term2_nr = self.apply_non_reciprocal(&term2, layer_idx)?;
            term2 = term2_nr;

            if let Some(nemo) = &self.nvidia_physicsnemo {
                term2 = nemo
                    .accelerate_500x(&term2)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            }

            let lb_t = Tensor::new(lorentz_boost as f32, device)?;
            delta_probe = term1.add(&term2)?.broadcast_mul(&lb_t)?;
        } else {
            let m_f32 = momentum as f32;
            let one_minus_m_t = Tensor::new(1.0 - m_f32, device)?;
            delta_probe = delta_probe.broadcast_mul(&one_minus_m_t)?;
        }

        // Store for next step (We store the PROBE delta, but maybe we should store full?)
        // Storing probe is enough as we only effectively use probe momentum next step
        self.last_deltas.insert(layer_idx, delta_probe.clone());

        // === SAFETY 1: EVENT HORIZON CLAMP ===
        let delta_len_sq = delta_probe.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let delta_len = delta_len_sq.sqrt();
        let safe_delta = if delta_len.is_nan() || delta_len > 100.0 {
            if delta_len > 10.0 {
                // Tighter
                let limit = Tensor::new(10.0 / delta_len as f32, device)?;
                delta_probe.broadcast_mul(&limit)?
            } else {
                println!(" [WARN] NaN in delta at layer {} - resetting", layer_idx);
                self.last_deltas.remove(&layer_idx);
                delta_probe.zeros_like()?
            }
        } else {
            delta_probe
        };

        // === BINARY MASK (LOWERED WHISPER) ===
        // L0-30: Full Force
        let mask_val = if layer_idx < 31 { 1.0f32 } else { 0.02f32 };
        let mask_t = Tensor::new(mask_val, device)?;
        let masked_delta = safe_delta.broadcast_mul(&mask_t)?;

        // === PROJECTION TO FULL TENSOR ===
        // ...

        let final_delta = if seq_len > 1 {
            let zeros_ctx = Tensor::zeros((b_sz, seq_len - 1, hidden_sz), DType::F32, device)?;
            let probe_reshaped = masked_delta.reshape((b_sz, 1, hidden_sz))?;
            Tensor::cat(&[&zeros_ctx, &probe_reshaped], 1)?
        } else {
            masked_delta.reshape((b_sz, seq_len, hidden_sz))?
        };

        if self.stdout_debug() {
            println!(" [DBG] apply_forces END layer {}", layer_idx);
        }

        // === PRESSURE-TRIGGERED MICRO-WOBBLE (Self-Correction Trigger) ===
        // Inject noise only when raw manifold pressure crosses the spike threshold.
        let mut final_delta = final_delta;
        if self.last_wobble_pressure_crossing {
            let device = final_delta.device();
            if let Ok(wobble) = Tensor::randn(0.0f32, 0.06, final_delta.shape(), device) {
                match final_delta.add(&wobble) {
                    Ok(new_delta) => {
                        final_delta = new_delta;
                        if layer_idx == 20 {
                            if self.stdout_debug() {
                                println!(
                                    " [WOBBLE] Pressure crossing at step {} (ghost_pre_norm={:.4}, threshold={:.2})",
                                    self.current_step,
                                    self.last_ghost_pre_norm,
                                    NIODOO_WOBBLE_PRESSURE_THRESHOLD
                                );
                            }
                        }
                    }
                    Err(e) => println!(" [WARN] Wobble failed: {:?}", e),
                }
            }
        }

        // === PHASE 23-B: ISO-METRIC REPAIR ===
        if layer_idx >= 30 {
            let proposed_state = (state_f32.clone() + &final_delta)?;
            let original_norm = state_f32.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
            self.layer_norms
                .insert(layer_idx, original_norm.mean_all()?.to_scalar::<f32>()?);

            let proposed_norm = proposed_state.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
            let scale = (original_norm / (proposed_norm + 1e-6)?)?;
            let fixed_state = proposed_state.broadcast_mul(&scale)?;
            let repaired_delta = (&fixed_state - state_f32)?;
            Ok(repaired_delta.to_dtype(state.dtype())?)
        } else {
            Ok(final_delta.to_dtype(state.dtype())?)
        }
    }

    fn get_physics_blend(&self) -> f32 {
        if self.braking {
            return 0.0;
        }
        self.physics_blend
    }

    fn set_physics_blend(&mut self, blend: f32) {
        self.physics_blend = blend;
    }

    fn get_physics_layer_range(&self) -> (usize, usize) {
        (self.physics_start_layer, self.physics_end_layer)
    }

    fn use_multiplicative_blend(&self) -> bool {
        self.multiplicative_blend
    }

    fn set_braking(&mut self, braking: bool) {
        self.braking = braking;
    }

    fn physics_invoke_for_early_worker_influence(&self, layer_idx: usize) -> bool {
        matches!(
            (
                self.specialist_memory_workers_mode,
                self.specialist_memory_worker_influence_layers,
            ),
            (SpecialistMemoryWorkerMode::Influence, Some((lo, hi)))
                if layer_idx >= lo && layer_idx <= hi
        ) && !(layer_idx >= self.physics_start_layer && layer_idx <= self.physics_end_layer)
    }
}

impl PrincipiaEngine {
    // 🧠 ELASTIC GRAVITY: Calculate Drift from Anchor (Sun)
    pub fn calculate_drift(&self, current_state: &Tensor) -> candle_core::Result<f32> {
        if let Some(sun) = &self.goal_embedding {
            let c_flat = current_state.flatten_all()?;
            let s_flat = sun.flatten_all()?;

            let c_norm_scalar = c_flat.sqr()?.sum_all()?.sqrt()?;
            let c_norm = c_flat.broadcast_div(&c_norm_scalar)?;

            let s_norm_scalar = s_flat.sqr()?.sum_all()?.sqrt()?;
            let s_norm = s_flat.broadcast_div(&s_norm_scalar)?;

            // Dot Product = Cosine Similarity
            let sim = (c_norm * s_norm)?.sum_all()?.to_scalar::<f32>()?;
            return Ok(1.0 - sim); // 0.0 = aligned, 1.0 = drift
        }
        Ok(0.0)
    }

    /// Phase 4: Heartbeat Tick with ADRENALINE DECAY
    /// If boredom > 0.8: Trigger "Adrenaline Shot" - 5-token decaying energy boost
    /// Creates a cognitive detour: SPIKE -> HIGH DRIFT -> SETTLE
    pub fn heartbeat_tick(&mut self, telemetry: &TokenPhysics) {
        const STRESS_THRESHOLD: f32 = 15.0;
        const BOREDOM_THRESHOLD: f32 = 2.0;
        const BUFFER_SIZE: usize = 10;

        // Adrenaline constants
        const ADRENALINE_TRIGGER_BOREDOM: f32 = 0.8;
        const ADRENALINE_INITIAL: f32 = 5.0; // Start value for decay curve
        const ADRENALINE_DECAY: f32 = 1.0; // Subtract per token
        const ADRENALINE_COOLDOWN: usize = 25; // Wait before next shot

        // Base limits
        const EMPATHY_DECAY: f32 = 0.01;

        // Focus lock constants
        const FOCUS_BLEND: f32 = 0.5;
        const FOCUS_REPULSION: f32 = 0.0;
        const FOCUS_GRAVITY_SCALE: f32 = 1.35;

        // ==========================================
        // FOCUS LOCK (runs every tick, highest priority)
        // ==========================================
        if self.focus_lock_remaining_ticks > 0 {
            self.focus_lock_remaining_ticks -= 1;
            self.physics_blend = FOCUS_BLEND;
            self.dynamic_repulsion = FOCUS_REPULSION;
            let base_focus_gravity = self.dynamic_gravity.max(self.heartbeat_gravity);
            self.dynamic_gravity =
                (base_focus_gravity * FOCUS_GRAVITY_SCALE).clamp(self.heartbeat_gravity, 4.0);
            self.adrenaline = 0.0; // keep exploration suppressed

            if self.stdout_debug() {
                println!(
                    "[FOCUS LOCK] remaining={} max={} blend={:.1} rep={:.1} gravity={:.2}",
                    self.focus_lock_remaining_ticks,
                    self.focus_lock_max_ticks,
                    FOCUS_BLEND,
                    FOCUS_REPULSION,
                    self.dynamic_gravity
                );
            }
            return;
        }

        if self.tda_shadow_breath_apply
            && telemetry.tda_shadow_decision_fresh
            && telemetry.tda_shadow_breath_requested
        {
            self.focus_lock_remaining_ticks = self.focus_lock_max_ticks.max(1);
            self.physics_blend = FOCUS_BLEND;
            self.dynamic_repulsion = FOCUS_REPULSION;
            let base_focus_gravity = self.dynamic_gravity.max(self.heartbeat_gravity);
            self.dynamic_gravity =
                (base_focus_gravity * FOCUS_GRAVITY_SCALE).clamp(self.heartbeat_gravity, 4.0);
            self.adrenaline = 0.0;

            if self.stdout_debug() {
                println!(
                    "[TDA BREATH] focus_lock={} action={} score={:.3} reason={}",
                    self.focus_lock_remaining_ticks,
                    telemetry.tda_shadow_action,
                    telemetry.tda_shadow_breath_score,
                    telemetry.tda_shadow_reason
                );
            }
            return;
        }

        // ==========================================
        // ADRENALINE PROCESSING (runs every tick)
        // ==========================================
        if self.adrenaline > 0.0 {
            // Apply adrenaline boost to physics
            let boosted_blend = self.heartbeat_blend + self.adrenaline;
            let boosted_repulsion = self.heartbeat_repulsion + (self.adrenaline * -0.5);

            self.physics_blend = boosted_blend;
            self.dynamic_repulsion = boosted_repulsion;

            if self.stdout_debug() {
                println!(
                    "[ADRENALINE] level={:.1} -> blend={:.1} rep={:.1}",
                    self.adrenaline, boosted_blend, boosted_repulsion
                );
            }

            // Decay
            self.adrenaline -= ADRENALINE_DECAY;
            if self.adrenaline <= 0.0 {
                self.adrenaline = 0.0;
                if self.stdout_debug() {
                    println!("[ADRENALINE] Wore off - returning to base");
                }
            }

            // While adrenaline active, skip normal heartbeat processing
            self.empathy_spike = (self.empathy_spike - EMPATHY_DECAY * 0.5).max(0.0);
            return;
        }

        // ==========================================
        // NORMAL HEARTBEAT PROCESSING
        // ==========================================

        // Update stress buffer
        self.stress_buffer.push_back(telemetry.total_force);
        if self.stress_buffer.len() > BUFFER_SIZE {
            self.stress_buffer.pop_front();
        }

        // Decrement cooldown
        if self.defib_cooldown > 0 {
            self.defib_cooldown -= 1;
        }

        // Calculate stress/boredom levels
        if self.stress_buffer.len() >= 3 {
            let avg_force: f32 =
                self.stress_buffer.iter().sum::<f32>() / self.stress_buffer.len() as f32;
            let surface_heuristic_flag = telemetry.surface_heuristic_flag;
            let empathy = self.empathy_spike.clamp(0.0, 2.0);

            // Stress: high force or surface anomaly
            if avg_force > STRESS_THRESHOLD || surface_heuristic_flag {
                let stress_delta = (0.2 * (1.0 - (empathy * 0.25).clamp(0.0, 0.5))).max(0.05);
                self.stress_level = (self.stress_level + stress_delta).min(1.0);
                self.boredom_level = (self.boredom_level - (0.1 + empathy * 0.02)).max(0.0);
            }
            // Boredom: low force
            else if avg_force < BOREDOM_THRESHOLD {
                let boredom_delta = (0.2 * (1.0 - (empathy * 0.20).clamp(0.0, 0.4))).max(0.05);
                self.boredom_level = (self.boredom_level + boredom_delta).min(1.0);
                self.stress_level = (self.stress_level - (0.1 + empathy * 0.03)).max(0.0);
            }
            // Normal
            else {
                self.stress_level = (self.stress_level - (0.05 + empathy * 0.02)).max(0.0);
                self.boredom_level = (self.boredom_level - (0.05 + empathy * 0.02)).max(0.0);
            }
            // ⚡ ADRENALINE SHOT: If boredom exceeds threshold and cooldown expired
            if self.boredom_level > ADRENALINE_TRIGGER_BOREDOM && self.defib_cooldown == 0 {
                self.adrenaline = ADRENALINE_INITIAL;
                self.defib_cooldown = ADRENALINE_COOLDOWN;
                if self.stdout_debug() {
                    println!(
                        "[ADRENALINE] ⚡ SHOT! boredom={:.2} -> adrenaline={:.1} (5-token boost, next in {} tokens)",
                        self.boredom_level, ADRENALINE_INITIAL, ADRENALINE_COOLDOWN
                    );
                }
            }
        }

        self.empathy_spike = (self.empathy_spike - EMPATHY_DECAY).max(0.0);

        // When no adrenaline, use base physics
        self.physics_blend = self.heartbeat_blend + self.empathy_spike * 0.15;
        self.dynamic_repulsion = self.heartbeat_repulsion + self.empathy_spike * 0.20;
    }

    // Non-Reciprocal Force (Plasma-Inspired)
    pub(crate) fn apply_non_reciprocal(
        &self,
        delta: &Tensor,
        layer_idx: usize,
    ) -> candle_core::Result<Tensor> {
        if layer_idx > 20 {
            let asym_factor = Tensor::new(0.5f32, delta.device())?;
            let noise = Tensor::randn(0.0f32, 0.1f32, delta.shape(), delta.device())?;
            let asym_delta = delta.broadcast_mul(&asym_factor)?;
            Ok((asym_delta + noise)?)
        } else {
            Ok(delta.clone())
        }
    }

    // Novel: Lorentz Boost (Relativistic Stability)
    pub(crate) fn compute_lorentz_boost(&self, _layer_idx: usize) -> candle_core::Result<f32> {
        let beta: f64 = 0.9;
        Ok((1.0 - beta * beta).sqrt() as f32)
    }

    pub(crate) fn entangle_particles(&mut self, idx1: usize, idx2: usize) -> Result<()> {
        if idx1 < self.sentence_history.len() && idx2 < self.sentence_history.len() {
            if idx1 == idx2 {
                return Ok(());
            }

            let strength = 0.8; // Simplification

            if let Some(p1) = self.sentence_history.get_mut(idx1) {
                p1.entangled_with.insert(idx2, strength);
            }
            if let Some(p2) = self.sentence_history.get_mut(idx2) {
                p2.entangled_with.insert(idx1, strength);
            }

            // Update shared state
            let p1_pos = self.sentence_history[idx1].position.clone();
            let p2_pos = self.sentence_history[idx2].position.clone();
            // FIX: Unwrap result before dividing
            let sum = p1_pos.broadcast_add(&p2_pos).map_err(anyhow::Error::new)?;
            let scale_half = Tensor::new(0.5f32, self.charge_tensor.device())?;
            let shared = sum.broadcast_mul(&scale_half).map_err(anyhow::Error::new)?;

            self.sentence_history[idx1].quantum_state = shared.clone();
            self.sentence_history[idx2].quantum_state = shared;
        }
        Ok(())
    }

    pub(crate) fn update_entangled_state(&self, p: &SentenceParticle) -> Result<Tensor> {
        // Safety: Verify shapes match before arithmetic
        if p.quantum_state.dims() != p.position.dims() {
            // Shapes mismatch - return original position unmodified
            return Ok(p.position.clone());
        }
        let nudge = (&p.quantum_state - &p.position).map_err(anyhow::Error::new)?;
        let scale = Tensor::new(0.1f32, nudge.device())?;
        let nudge_scaled = nudge.broadcast_mul(&scale).map_err(anyhow::Error::new)?;
        Ok((&p.position + nudge_scaled).map_err(anyhow::Error::new)?)
    }

    pub(crate) fn evolve_physics_rules(&mut self) -> Result<()> {
        let mut new_pop_vec: Vec<EvoEntry> = self
            .evo_population
            .iter()
            .map(|e| EvoEntry {
                fitness: e.fitness,
                params: e.params.clone(),
            })
            .collect();
        // Fix: BinaryHeap.iter() is unordered. Sort to ensure deterministic crossover pairing.
        new_pop_vec.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if new_pop_vec.is_empty() {
            return Ok(());
        }

        // CROSSOVER: Deterministic averaging of adjacent entries
        let read_only_pop = new_pop_vec.clone();
        let len = read_only_pop.len();

        for (i, entry) in new_pop_vec.iter_mut().enumerate() {
            // Deterministic neighbor selection (Ring topology)
            let other_idx = (i + 1) % len;
            let other = &read_only_pop[other_idx];

            // Converge towards mean
            entry.params.gravity = (entry.params.gravity + other.params.gravity) / 2.0;

            // Deterministic Mutation: Oscillate based on index
            let mutation = if i % 2 == 0 { 0.001 } else { -0.001 };
            entry.params.gravity += mutation;

            // Quantize (HIGGS-like)
            entry.params.gravity = self.quantize_higgs(entry.params.gravity)?;
        }

        // SELECT BEST (Deterministically)
        // Pick the one with highest fitness, or index 0 if all equal
        new_pop_vec.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.params = new_pop_vec[0].params.clone();
        Ok(())
    }

    pub(crate) fn quantize_higgs(&self, val: f64) -> Result<f64> {
        Ok((val * 100.0).round() / 100.0)
    }

    /// Compute the Ghost Particle Vector for Niodoo Protocol
    ///
    /// Formula: F_total = Norm(α·V_VAD + β·V_Memory + γ·V_Surprisal)
    ///
    /// Components:
    /// - V_VAD: Emotional state projected to hidden dim
    /// - V_Memory: Weighted average of sentence history
    /// - V_Surprisal: Mass multiplier based on information content
    /// - Goal: If present, dominates other components
    pub(crate) fn compute_ghost_vector(
        &mut self,
        input_embedding: &Tensor,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        self.last_intervention_applied = false;
        self.last_ghost_pull_delta_norm = 0.0;
        self.last_projection_strategy = "none".to_string();

        if self.stdout_debug() {
            println!(" [DBG] compute_ghost_vector start");
            std::io::stdout().flush().unwrap();
        }
        // NIODOO PROTOCOL: F_total = Norm(alpha * V_vad + beta * V_mem + gamma * V_surp)

        let mut components = Vec::new();
        let mut weights = Vec::new();

        // Component 1: VAD Emotional State (α · V_VAD)
        if let Some(_vad_head) = &self.vad_head {
            // Infer current emotional state from context
            // let (valence, arousal, dominance) = vad_head.infer_vad_from_context(&self.sentence_history);

            // Project 3D VAD to hidden dimension
            // let vad_vector = vad_head.project_vad(valence, arousal, dominance)?
            //    .to_device(device)?
            //    .to_dtype(DType::F32)?;

            // Weight by alpha_emo (emotional mass weight)
            let _alpha = self.params.alpha_emo as f32;

            // SAFETY: Disable VAD if random (no file loaded) to avoid garbage injection
            // We assume valid VAD only if we have a way to verify it, for now we skip to ensure coherence.
            // components.push(vad_vector);
            // weights.push(alpha);
            if self.stdout_debug() {
                println!(" [Niodoo] Skipping VAD (Random Noise Prevention)");
            }
        }

        // Component 2: Memory Context (Needle Physics: β · V_Memory)
        if !self.sentence_history.is_empty() {
            let (memory_vector, dynamic_mass) = self.compute_needle_physicsmod(input_embedding)?;
            let memory_vector = memory_vector.to_device(device)?.to_dtype(DType::F32)?;

            // Weight by semantic + coherence mass (β = α_sem + α_coh)
            let beta = (self.params.alpha_sem + self.params.alpha_coh) as f32 * dynamic_mass;
            components.push(memory_vector);
            weights.push(beta);
        }

        // Component 3: Goal Attractor (Overrides if present)
        // Goal has highest priority and largest weight
        if let Some(goal) = &self.goal_embedding {
            let goal_dev = goal.to_device(device)?.to_dtype(DType::F32)?;

            // Goal gets massive weight (acts as "Black Hole")
            let goal_weight = 2.0;
            components.push(goal_dev);
            weights.push(goal_weight);
        }

        // Component 3.5: Bridge Ghost Attractor (niodv4_bridge)
        #[cfg(feature = "niodv4_bridge")]
        {
            if let Some(registry) = &self.ghost_registry {
                // Find nearest ghost based on current query embedding
                let (nearest_id, dist, _) = self.find_nearest_ghost_info(input_embedding)?;
                if let Some(id) = nearest_id {
                    if let Some(bridge_vec) = self.get_bridge_ghost_vector(&id, device)? {
                        // Bridge Ghost weight is dynamically adjusted by distance
                        // closer = stronger pull
                        let bridge_weight = 1.0 / (1.0 + dist as f32);
                        components.push(bridge_vec);
                        weights.push(bridge_weight);

                        self.last_projection_strategy = "simple".to_string();
                    }
                }
            }
        }

        if components.is_empty() {
            self.clear_ghost_pressure_telemetry();
            return Ok(None);
        }

        // Component 4: Surprisal Mass Multiplier (γ)
        let avg_surprisal = if !self.current_surprisals.is_empty() {
            let sum: f32 = self.current_surprisals.iter().sum();
            sum / self.current_surprisals.len() as f32
        } else {
            1.0 // Default: no scaling
        };

        // Compute weighted average: F_total = Σ(w_i · V_i) / Σ(w_i)
        let total_weight: f32 = weights.iter().sum();
        if total_weight.abs() <= 1e-6 {
            self.clear_ghost_pressure_telemetry();
            return Ok(None);
        }
        let mut f_total = components[0].zeros_like()?;

        for (i, comp) in components.iter().enumerate() {
            let normalized_weight = weights[i] / total_weight;
            let w_t = Tensor::new(normalized_weight, comp.device())?;
            let weighted_comp = comp.broadcast_mul(&w_t)?;
            f_total = (f_total + weighted_comp)?;
        }

        // Apply Surprisal Mass Scaling (γ)
        // High surprisal = more information = larger mass (stronger Ghost)
        let mass_multiplier = 1.0 + (avg_surprisal - 1.0).abs() * 0.5;
        let mm_t = Tensor::new(mass_multiplier, f_total.device())?;
        f_total = f_total.broadcast_mul(&mm_t)?;

        if self.stdout_debug() {
            println!(
                " [Niodoo] Surprisal γ={:.2}, mass_mult={:.2}",
                avg_surprisal, mass_multiplier
            );
        }

        // Normalize to unit vector (direction only, mass applied in NakedLlama)
        let norm = f_total.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        self.update_ghost_pressure_telemetry(norm);
        if norm > 1e-6 {
            let norm_t = Tensor::new(norm, f_total.device())?;
            let normalized = f_total.broadcast_div(&norm_t)?;
            if self.stdout_debug() {
                println!(" [Niodoo] Ghost Vector Active: norm={:.4}", norm);
            }
            Ok(Some(normalized))
        } else {
            if self.stdout_debug() {
                println!(" [Niodoo] Ghost Vector too small: norm={:.6}", norm);
            }
            Ok(Some(f_total))
        }
    }

    pub(crate) fn compute_needle_physicsmod(&self, query: &Tensor) -> Result<(Tensor, f32)> {
        if self.stdout_debug() {
            println!(
                " [DBG] compute_needle_physicsmod start. Input shape: {:?}",
                query.dims()
            );
            std::io::stdout().flush().unwrap();
        }
        // 1. Calculate dynamic mass based on sentence history correlation
        // S_physics = Sim(Q, C) / Dispersion
        // Dispersion ~ 1/m_coh (Approximation)
        let mut loaded_memories = Vec::new();
        let _total_sim = 0.0;
        let mut split_masses = Vec::new();

        // Ensure query is 1D [hidden_dim]
        let q_flat = query.flatten_all()?.squeeze(0)?;
        let q_norm = q_flat.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;

        for p in &self.sentence_history {
            // Similarity
            let c_dev = p.position.to_device(query.device())?;
            let c_flat = c_dev.flatten_all()?;
            let c_norm = c_flat.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;

            // Element-wise multiply (both should be [hidden_dim])
            let dot = (&q_flat * &c_flat)?.sum_all()?.to_scalar::<f32>()?;
            let sim = if q_norm > 0.0 && c_norm > 0.0 {
                dot / (q_norm * c_norm)
            } else {
                0.0
            };

            // Physics Score S = Sim / Dispersion
            let dispersion = (1.0 - p.m_coh).max(0.01);
            let s_physics = sim / dispersion;

            if s_physics > 0.5 {
                loaded_memories.push(c_dev);
                split_masses.push(s_physics);
            }
        }

        if loaded_memories.is_empty() {
            return Ok((self.get_weighted_context_avg()?, 1.0));
        }

        // Centroid - work in 1D space [hidden_dim]
        let hidden_dim = loaded_memories[0].dims()[0];
        let mut centroid = Tensor::zeros((hidden_dim,), DType::F32, query.device())?;
        let mut total_m = 0.0;
        for (mem, m) in loaded_memories.iter().zip(split_masses.iter()) {
            let mem_flat = mem.flatten_all()?;
            let m_t = Tensor::new(*m, query.device())?;
            centroid = (centroid + mem_flat.broadcast_mul(&m_t)?)?;
            total_m += m;
        }
        if total_m > 0.0 {
            let tm_t = Tensor::new(total_m, query.device())?;
            centroid = centroid.broadcast_div(&tm_t)?;
        }
        Ok((centroid, (total_m / loaded_memories.len() as f32).min(2.0)))
    }

    pub(crate) fn compute_total_mass(
        &self,
        m_info: f32,
        m_sem: f32,
        m_coh: f32,
        m_struct: f32,
        m_quantum: f32,
        m_geometric: f32,
        m_emo: f32,
        kl_delta: f32,
    ) -> f32 {
        let params = &self.params;
        let base_mass = params.alpha_info as f32 * m_info
            + params.alpha_sem as f32 * m_sem
            + params.alpha_coh as f32 * m_coh
            + params.alpha_struct as f32 * m_struct
            + params.alpha_quantum as f32 * m_quantum
            + params.alpha_geometric as f32 * m_geometric
            + if params.use_emo {
                params.alpha_emo as f32 * m_emo
            } else {
                0.0
            };

        let multiplier = 1.0 + kl_delta.abs() * 2.0;
        (base_mass * multiplier).clamp(0.1, 100.0)
    }

    pub(crate) fn get_weighted_context_avg(&self) -> Result<Tensor> {
        if self.sentence_history.is_empty() {
            return Ok(Tensor::zeros(
                (self.hidden_dim,),
                DType::F32,
                self.charge_tensor.device(),
            )?);
        }
        if self.stdout_debug() {
            println!(
                " [DBG] get_weighted_context_avg start. Hist len: {}",
                self.sentence_history.len()
            );
            std::io::stdout().flush().unwrap();
        }
        // FIX: Ignore the MOST RECENT particle to prevent "Green Sky" / Self-Attraction Loop
        let n = self.sentence_history.len();
        let history_iter = if n > 0 {
            self.sentence_history.iter().take(n - 1)
        } else {
            self.sentence_history.iter().take(0)
        };

        let weights: Vec<f32> = history_iter
            .map(|p| p.fitness * p.m_quantum * p.m_geometric)
            .collect();
        if self.stdout_debug() {
            println!(" [DBG] Weights collected. Len: {}", weights.len());
            std::io::stdout().flush().unwrap();
        }
        let sum_w: f32 = weights.iter().sum();
        let norm_w: Vec<f32> = if sum_w > 0.0 {
            weights.iter().map(|w| w / sum_w).collect()
        } else {
            vec![1.0 / weights.len() as f32; weights.len()]
        };

        let mut avg = Tensor::zeros((self.hidden_dim,), DType::F32, self.charge_tensor.device())?;

        // FIX: Re-create the iterator to match the weights length
        let history_iter_2 = if n > 0 {
            self.sentence_history.iter().take(n - 1)
        } else {
            self.sentence_history.iter().take(0)
        };

        for (i, p) in history_iter_2.enumerate() {
            let scale = Tensor::new(norm_w[i], self.charge_tensor.device())?;
            let weighted_p = p
                .position
                .to_device(self.charge_tensor.device())?
                .broadcast_mul(&scale)?;
            avg = (avg + weighted_p)?;
        }
        Ok(avg)
    }

    pub(crate) fn compute_m_coh(&self, embedding: &Tensor) -> Result<f32> {
        if self.sentence_history.is_empty() {
            return Ok(0.5);
        }
        // Altair-inspired geometric adjustment
        let emb_adj = self.adjust_geometric(embedding)?;
        let context_avg = self.get_weighted_context_avg()?;
        let context_avg = context_avg.to_device(emb_adj.device())?;

        let emb_flat = emb_adj.flatten_all()?;
        let ctx_flat = context_avg.flatten_all()?;

        let dot = (&emb_flat * &ctx_flat)?.sum_all()?.to_scalar::<f32>()?;
        let norm_emb = emb_flat.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let norm_ctx = ctx_flat.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;

        if norm_emb == 0.0 || norm_ctx == 0.0 {
            return Ok(0.5);
        }
        let sim = dot / (norm_emb * norm_ctx + 1e-8);
        Ok(sim.max(0.0))
    }

    pub(crate) fn compute_similarity(&self, a: &Tensor, b: &Tensor) -> Result<f32> {
        let dot = (a * b)?.sum_all()?.to_scalar::<f32>()?;
        let n_a = a.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let n_b = b.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        Ok(dot / (n_a * n_b + 1e-8))
    }

    pub(crate) fn is_live_runtime_motif(motif: &RuntimeMotifField) -> bool {
        motif.motif_kind == "live" || motif.motif_kind == "promoted"
    }

    pub(crate) fn live_motif_merge_threshold(motif: &RuntimeMotifField) -> f32 {
        let crystal_bias = (motif.structured_signal * motif.tightness_score).clamp(0.0, 1.0);
        (LIVE_MOTIF_MERGE_DISTANCE_THRESHOLD + motif.radius_mean + motif.radius_std * 0.5
            - crystal_bias * 0.06)
            .clamp(0.05, 0.35)
    }

    pub(crate) fn live_motif_probe_stats(
        &self,
        probe_normalized: &Tensor,
    ) -> Result<LiveMotifProbeStats> {
        let mut stats = LiveMotifProbeStats::default();
        let mut best_distance = f32::INFINITY;
        let mut best_radius = 0.0f32;
        let mut best_pressure = 0.0f32;

        for motif in &self.runtime_motifs {
            if !Self::is_live_runtime_motif(motif) {
                continue;
            }

            stats.live_motif_count += 1;

            let sim = self.compute_similarity(probe_normalized, &motif.vector)?;
            let distance = (1.0 - sim).max(0.0);
            let radius = (motif.radius_mean + motif.radius_std + 0.02).clamp(0.02, 0.45);
            let density = ((motif.member_count.saturating_sub(1)) as f32 / 5.0).clamp(0.0, 1.0);
            let width_tightness =
                (1.0 / (1.0 + motif.radius_mean * 6.0 + motif.radius_std * 8.0)).clamp(0.15, 1.0);
            let proximity =
                (1.0 - distance / (radius + LIVE_MOTIF_MERGE_DISTANCE_THRESHOLD)).clamp(0.0, 1.0);
            let local_pressure = proximity * (0.35 + density * 0.65) * width_tightness;

            if distance < best_distance {
                best_distance = distance;
                best_radius = radius;
            }
            if local_pressure > best_pressure {
                best_pressure = local_pressure;
            }
        }

        if stats.live_motif_count > 0 {
            stats.nearest_distance = best_distance;
            stats.nearest_radius = best_radius;
            stats.trap_pressure = best_pressure.clamp(0.0, 1.5);
            stats.fragmentation =
                ((stats.live_motif_count.saturating_sub(1)) as f32 / 6.0).clamp(0.0, 1.0);
        }

        Ok(stats)
    }

    pub(crate) fn refresh_live_motif_scores(
        motif: &mut RuntimeMotifField,
        particle_fitness: f32,
        empathy_spike: f32,
        turn_structure_bias: f32,
    ) {
        let density = ((motif.member_count.saturating_sub(1)) as f32 / 5.0).clamp(0.0, 1.0);
        let empathy = (empathy_spike / 2.0).clamp(0.0, 1.0);
        let structure = motif
            .structured_signal
            .max(turn_structure_bias * 0.70)
            .clamp(0.0, 1.0);
        let tightness = motif_tightness(motif.radius_mean, motif.radius_std);
        let crystal_signal = (structure * tightness).clamp(0.0, 1.0);
        let liquid_signal = ((1.0 - structure) * (0.35 + empathy * 0.65)).clamp(0.0, 1.0);

        motif.tightness_score = tightness;
        motif.persistence_score =
            (0.05 + density * 0.54 + crystal_signal * 0.18 + liquid_signal * 0.10 + empathy * 0.08)
                .clamp(0.05, 0.95);
        motif.readiness_score = (0.10
            + density * 0.28
            + particle_fitness * 0.18
            + crystal_signal * 0.28
            + structure * 0.12
            + empathy * 0.05)
            .clamp(0.10, 0.98);
        motif.injection_strength =
            (0.05 + density * 0.14 + crystal_signal * 0.14 + structure * 0.05 + empathy * 0.04)
                .clamp(0.05, 0.52);
        motif.orbit_count = (density + crystal_signal * 0.35 + empathy * 0.10).clamp(0.0, 1.4);
    }

    pub(crate) fn strongest_structured_reentry_target(
        &self,
    ) -> Option<(Tensor, String, String, f32, f32, f32)> {
        self.runtime_motifs
            .iter()
            .filter(|motif| Self::is_live_runtime_motif(motif))
            .max_by(|a, b| {
                let score_a = (a.structured_signal * 0.40
                    + a.tightness_score * 0.25
                    + a.promotion_score * 0.20
                    + a.readiness_score * 0.10
                    + ((a.member_count.saturating_sub(1)) as f32 / 5.0).clamp(0.0, 1.0) * 0.05)
                    .clamp(0.0, 1.0);
                let score_b = (b.structured_signal * 0.40
                    + b.tightness_score * 0.25
                    + b.promotion_score * 0.20
                    + b.readiness_score * 0.10
                    + ((b.member_count.saturating_sub(1)) as f32 / 5.0).clamp(0.0, 1.0) * 0.05)
                    .clamp(0.0, 1.0);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.member_count.cmp(&b.member_count))
            })
            .map(|motif| {
                (
                    motif.vector.detach(),
                    motif.motif_kind.clone(),
                    motif.promotion_status.clone(),
                    motif.promotion_score,
                    motif.structured_signal,
                    motif.tightness_score,
                )
            })
    }

    pub(crate) fn controller_active(&self) -> bool {
        !self.ablate_periodic_controller
            && (self.restored_run_active
                || self.reentry_clamp_steps_remaining > 0
                || self.current_turn_structure_bias >= STRUCTURED_REENTRY_PROMPT_THRESHOLD)
    }

    pub(crate) fn active_routing_cache(&self) -> Option<&RoutingDecisionCache> {
        self.routing_cache
            .as_ref()
            .filter(|cache| cache.expires_at_step >= self.current_step)
    }

    pub(crate) fn live_median_radius(&self) -> f32 {
        let mut radii = self
            .runtime_motifs
            .iter()
            .filter(|motif| Self::is_live_runtime_motif(motif))
            .map(|motif| motif.radius_mean.max(0.0))
            .collect::<Vec<_>>();
        if radii.is_empty() {
            return 0.10;
        }
        radii.sort_by(|a, b| a.total_cmp(b));
        radii[radii.len() / 2]
    }

    // Dev-override helpers: use runtime arg if non-zero, else the constant.
    pub(crate) fn structured_candidate_task_sim(&self) -> f32 {
        if self.dev_structured_candidate_task_sim > 0.0 {
            self.dev_structured_candidate_task_sim
        } else {
            MOTIF_ROLE_STRUCTURED_CANDIDATE_TASK_SIM
        }
    }
    pub(crate) fn structured_candidate_bonus_scale(&self) -> f32 {
        if self.dev_structured_candidate_bonus_scale > 0.0 {
            self.dev_structured_candidate_bonus_scale
        } else {
            ROUTING_STRUCTURED_CANDIDATE_BONUS_SCALE
        }
    }
    pub(crate) fn neutral_basin_penalty_scale(&self) -> f32 {
        if self.dev_neutral_basin_penalty_scale > 0.0 {
            self.dev_neutral_basin_penalty_scale
        } else {
            ROUTING_NEUTRAL_BASIN_PENALTY_SCALE
        }
    }
    pub(crate) fn task_utility_bonus_scale(&self) -> f32 {
        if self.dev_task_utility_bonus_scale > 0.0 {
            self.dev_task_utility_bonus_scale
        } else {
            ROUTING_TASK_UTILITY_BONUS_SCALE
        }
    }
    pub(crate) fn fragmentation_discount(&self) -> f32 {
        if self.dev_fragmentation_discount > 0.0 {
            self.dev_fragmentation_discount
        } else {
            STRUCTURED_FRAGMENTATION_DISCOUNT
        }
    }
    pub(crate) fn restored_topology_floor_signal(&self) -> f32 {
        if self.dev_restored_topology_floor_signal > 0.0 {
            self.dev_restored_topology_floor_signal
        } else {
            RESTORED_TOPOLOGY_FLOOR_SIGNAL
        }
    }
    pub(crate) fn restored_topology_floor_tightness(&self) -> f32 {
        if self.dev_restored_topology_floor_tightness > 0.0 {
            self.dev_restored_topology_floor_tightness
        } else {
            RESTORED_TOPOLOGY_FLOOR_TIGHTNESS
        }
    }
    pub(crate) fn structured_candidate_escalation_topology(&self) -> f32 {
        if self.dev_structured_candidate_escalation_topology > 0.0 {
            self.dev_structured_candidate_escalation_topology
        } else {
            ROUTING_STRUCTURED_CANDIDATE_ESCALATION_TOPOLOGY
        }
    }
    pub(crate) fn structured_candidate_escalation_task(&self) -> f32 {
        if self.dev_structured_candidate_escalation_task > 0.0 {
            self.dev_structured_candidate_escalation_task
        } else {
            ROUTING_STRUCTURED_CANDIDATE_ESCALATION_TASK
        }
    }
    pub(crate) fn routing_stickiness_bonus(&self) -> f32 {
        if self.dev_routing_stickiness_bonus > 0.0 {
            self.dev_routing_stickiness_bonus
        } else {
            ROUTING_STICKINESS_BONUS
        }
    }
    pub(crate) fn routing_stickiness_ticks(&self) -> usize {
        if self.dev_routing_stickiness_ticks > 0.0 {
            self.dev_routing_stickiness_ticks as usize
        } else {
            ROUTING_STICKINESS_TICKS
        }
    }

    pub(crate) fn apply_routing_stickiness(
        &mut self,
        selected_motif_id: &str,
        selected_role: &str,
    ) {
        if matches!(selected_role, "structured" | "structured_candidate") {
            self.routing_stickiness_motif_id = Some(selected_motif_id.to_string());
            self.routing_stickiness_remaining_ticks = self.routing_stickiness_ticks();
        }
    }

    pub(crate) fn stickiness_score_for_motif(&self, motif_id: &str) -> f32 {
        if self.routing_stickiness_remaining_ticks > 0
            && self
                .routing_stickiness_motif_id
                .as_deref()
                .is_some_and(|s| s == motif_id)
        {
            self.routing_stickiness_bonus()
                * (self.routing_stickiness_remaining_ticks as f32
                    / self.routing_stickiness_ticks() as f32)
        } else {
            0.0
        }
    }

    pub(crate) fn decay_routing_stickiness(&mut self) {
        if self.routing_stickiness_remaining_ticks > 0 {
            self.routing_stickiness_remaining_ticks -= 1;
            if self.routing_stickiness_remaining_ticks == 0 {
                self.routing_stickiness_motif_id = None;
            }
        }
    }

    pub(crate) fn classify_motif_role(
        motif: &RuntimeMotifField,
        live_median_radius: f32,
        _task_anchor_similarity: f32,
        _structured_candidate_task_sim: f32,
    ) -> String {
        if motif.tightness_score >= MOTIF_ROLE_STRUCTURED_TIGHTNESS
            && motif.structured_signal >= MOTIF_ROLE_STRUCTURED_SIGNAL
        {
            "structured".to_string()
        } else if motif.tightness_score >= MOTIF_ROLE_STRUCTURED_CANDIDATE_TIGHTNESS
            && motif.structured_signal >= MOTIF_ROLE_STRUCTURED_CANDIDATE_SIGNAL
            && motif.topology_density >= 0.28
        {
            "structured_candidate".to_string()
        } else if motif.tightness_score < MOTIF_ROLE_CONVERSATIONAL_TIGHTNESS
            && motif.radius_mean >= live_median_radius.max(0.08)
        {
            "conversational".to_string()
        } else {
            "neutral".to_string()
        }
    }

    pub(crate) fn routing_task_anchor_similarity(&self, motif: &RuntimeMotifField) -> f32 {
        self.current_task_anchor_signature
            .as_ref()
            .map(|signature| cosine_similarity_slices(signature, &motif.raw_signature).max(0.0))
            .unwrap_or(0.0)
    }

    pub(crate) fn motif_topology_signature(
        &self,
        motif: &RuntimeMotifField,
    ) -> Result<(f32, f32, f32, f32, f32)> {
        let recent = self
            .sentence_history
            .iter()
            .rev()
            .take(12)
            .collect::<Vec<_>>();
        if recent.is_empty() {
            let fallback_density = (motif.tightness_score * 0.45).clamp(0.0, 1.0);
            let fallback_gap = (motif.flip_rate * 2.0 + motif.radius_std * 3.0).clamp(0.0, 1.0);
            let fallback_fragmentation = (fallback_gap * 0.75).clamp(0.0, 1.0);
            let fallback_hole =
                (fallback_gap * 0.55 + fallback_fragmentation * 0.45).clamp(0.0, 1.0);
            let fallback_tension =
                (motif.radius_std * 6.0 + motif.flip_rate * 2.2 + motif.max_pre_energy * 0.2)
                    .clamp(0.0, 1.0);
            return Ok((
                fallback_density,
                fallback_gap,
                fallback_fragmentation,
                fallback_hole,
                fallback_tension,
            ));
        }

        let threshold = (Self::live_motif_merge_threshold(motif) * 1.15).clamp(0.08, 0.42);
        let mut hit_indices = Vec::new();
        let mut hit_positions = Vec::new();
        let mut proximity_sum = 0.0f32;

        for (idx, particle) in recent.iter().rev().enumerate() {
            let sim = self.compute_similarity(&particle.position, &motif.vector)?;
            let distance = (1.0 - sim).max(0.0);
            if distance <= threshold {
                hit_indices.push(idx);
                hit_positions.push(particle.position.detach());
                proximity_sum += (1.0 - distance / threshold).clamp(0.0, 1.0);
            }
        }

        if hit_indices.is_empty() {
            let base_density = (motif.tightness_score * 0.35).clamp(0.0, 1.0);
            let fragmentation = (base_density * 0.6).clamp(0.0, 1.0);
            let gap = (1.0 - base_density + motif.flip_rate).clamp(0.0, 1.0);
            let hole = (gap * 0.5 + fragmentation * 0.5).clamp(0.0, 1.0);
            let tension = (motif.radius_std * 5.0 + motif.flip_rate * 1.8).clamp(0.0, 1.0);
            if motif.promotion_status == "restored_compact"
                && motif.structured_signal >= MOTIF_ROLE_STRUCTURED_CANDIDATE_SIGNAL
            {
                let sig_scale = self.restored_topology_floor_signal();
                let tight_scale = self.restored_topology_floor_tightness();
                let restored_floor = (motif.structured_signal * sig_scale
                    + motif.tightness_score * tight_scale)
                    .clamp(base_density, 1.0);
                let restored_gap = ((1.0 - motif.structured_signal) * 0.3).clamp(0.0, 1.0);
                let restored_frag = (restored_gap * 0.5).clamp(0.0, 1.0);
                return Ok((restored_floor, restored_gap, restored_frag, hole, tension));
            }
            return Ok((base_density, gap, fragmentation, hole, tension));
        }

        let base_density = hit_indices.len() as f32 / recent.len() as f32;
        let proximity = (proximity_sum / hit_indices.len() as f32).clamp(0.0, 1.0);
        let topology_density = (base_density * (0.55 + proximity * 0.45)).clamp(0.0, 1.0);

        let mut segments = 1usize;
        let mut gap_count = 0usize;
        for pair in hit_indices.windows(2) {
            if pair[1] > pair[0] + 1 {
                segments += 1;
                gap_count += 1;
            }
        }
        let sequential_gap_rate = if hit_indices.len() > 1 {
            gap_count as f32 / (hit_indices.len() - 1) as f32
        } else {
            (1.0 - topology_density).clamp(0.0, 1.0)
        };
        let raw_fragmentation = if hit_indices.len() > 1 {
            (segments.saturating_sub(1)) as f32 / (hit_indices.len() - 1) as f32
        } else {
            (1.0 - topology_density).clamp(0.0, 1.0) * 0.5
        };
        let fragmentation = if motif.structured_signal >= MOTIF_ROLE_STRUCTURED_CANDIDATE_SIGNAL
            && motif.tightness_score >= MOTIF_ROLE_STRUCTURED_CANDIDATE_TIGHTNESS
        {
            (raw_fragmentation * self.fragmentation_discount()).clamp(0.0, 1.0)
        } else {
            raw_fragmentation
        };
        let hole_pressure = (fragmentation * 0.55 + sequential_gap_rate * 0.45).clamp(0.0, 1.0);

        let mut tension_anchor_strength = 0.0f32;
        for triple in hit_positions.windows(3) {
            let second = ((&triple[2] - &triple[1])? - (&triple[1] - &triple[0])?)?;
            let second_norm = second.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            tension_anchor_strength = tension_anchor_strength.max(second_norm);
        }
        tension_anchor_strength =
            (tension_anchor_strength * 0.75 + motif.radius_std * 3.0 + motif.flip_rate * 1.2)
                .clamp(0.0, 1.0);

        Ok((
            topology_density,
            sequential_gap_rate,
            fragmentation,
            hole_pressure,
            tension_anchor_strength,
        ))
    }

    pub(crate) fn refresh_runtime_motif_metadata(&mut self) -> Result<()> {
        let live_median_radius = self.live_median_radius();
        for idx in 0..self.runtime_motifs.len() {
            let (
                topology_density,
                sequential_gap_rate,
                fragmentation,
                hole_pressure,
                tension_anchor_strength,
            ) = {
                let motif = &self.runtime_motifs[idx];
                self.motif_topology_signature(motif)?
            };
            let conflict_ratio =
                (fragmentation * 0.45 + hole_pressure * 0.35 + sequential_gap_rate * 0.20)
                    .clamp(0.0, 1.0);
            let mixed_ratio = {
                let motif = &self.runtime_motifs[idx];
                (1.0 - ((motif.structured_signal - 0.5).abs() * 2.0)).clamp(0.0, 1.0)
            };
            let role = {
                let motif = &self.runtime_motifs[idx];
                let imported_structured_bridge = motif.motif_kind == "bridge"
                    && motif.promotion_status == "imported"
                    && motif.structured_signal >= MOTIF_ROLE_STRUCTURED_CANDIDATE_SIGNAL
                    && matches!(
                        motif.motif_role.as_str(),
                        "structured" | "structured_candidate"
                    );
                if imported_structured_bridge {
                    motif.motif_role.clone()
                } else {
                    let task_sim = self
                        .current_task_anchor_signature
                        .as_ref()
                        .map(|sig| cosine_similarity_slices(sig, &motif.raw_signature).max(0.0))
                        .unwrap_or(0.0);
                    Self::classify_motif_role(
                        motif,
                        live_median_radius,
                        task_sim,
                        self.structured_candidate_task_sim(),
                    )
                }
            };
            let routing_safety_score =
                (1.0 - (conflict_ratio * 0.7 + mixed_ratio * 0.3)).clamp(0.0, 1.0);

            let motif = &mut self.runtime_motifs[idx];
            motif.topology_density = topology_density;
            motif.sequential_gap_rate = sequential_gap_rate;
            motif.fragmentation = fragmentation;
            motif.hole_pressure = hole_pressure;
            motif.tension_anchor_strength = tension_anchor_strength;
            motif.conflict_ratio = conflict_ratio;
            motif.mixed_ratio = mixed_ratio;
            motif.routing_safety_score = routing_safety_score;
            motif.motif_role = role;
            if motif.origin_run_id.is_empty() {
                motif.origin_run_id = if motif.motif_kind == "bridge" {
                    format!("bridge::{}", motif.source)
                } else {
                    self.current_run_id.clone()
                };
            }
            if motif.provenance_summary.is_empty() {
                motif.provenance_summary = format!(
                    "{}::{}::{}",
                    motif.source, motif.motif_kind, motif.promotion_status
                );
            }
            if motif.motif_kind == "promoted" && motif.promotion_epoch == 0 {
                motif.promotion_epoch = motif.last_updated_step.max(self.current_step);
            }
            motif.merge_key = format!(
                "{}::{}::{}",
                motif.origin_run_id, motif.promotion_epoch, motif.motif_id
            );
        }
        Ok(())
    }

    pub(crate) fn should_escalate_structured_candidate(
        &self,
        motif: &RuntimeMotifField,
        task_anchor_similarity: f32,
    ) -> bool {
        let hinge_recent = self
            .first_organic_promoted_step
            .or(self.first_recovered_promoted_step)
            .map(|step| self.current_step.saturating_sub(step) <= TASK_ANCHOR_BIND_TOKENS)
            .unwrap_or(false);
        let structured_context = self.restored_run_active
            && self.current_turn_structure_bias >= STRUCTURED_REENTRY_PROMPT_THRESHOLD
            && (self.reentry_clamp_steps_remaining > 0
                || self.structured_resume_window_remaining > 0
                || self.task_anchor_window_tokens_seen < TASK_ANCHOR_BIND_TOKENS
                || hinge_recent);
        if !structured_context || motif.motif_role != "structured_candidate" {
            return false;
        }

        let promoted_like = motif.motif_kind == "promoted"
            || matches!(
                motif.promotion_status.as_str(),
                "recovered_promoted" | "promoted" | "reinforcing"
            )
            || (motif.promotion_status == "restored_compact"
                && motif.structured_signal >= MOTIF_ROLE_STRUCTURED_CANDIDATE_SIGNAL
                && motif.tightness_score >= MOTIF_ROLE_STRUCTURED_CANDIDATE_TIGHTNESS);
        let persistent_enough = motif.controller_selected_count >= 2 || motif.member_count >= 2;

        motif.tightness_score >= ROUTING_STRUCTURED_CANDIDATE_ESCALATION_TIGHTNESS
            && motif.structured_signal >= ROUTING_STRUCTURED_CANDIDATE_ESCALATION_SIGNAL
            && motif.topology_density >= self.structured_candidate_escalation_topology()
            && task_anchor_similarity >= self.structured_candidate_escalation_task()
            && promoted_like
            && persistent_enough
    }

    pub(crate) fn effective_routing_role(
        &self,
        motif: &RuntimeMotifField,
        task_anchor_similarity: f32,
    ) -> (String, bool) {
        if self.should_escalate_structured_candidate(motif, task_anchor_similarity) {
            ("structured".to_string(), true)
        } else {
            (motif.motif_role.clone(), false)
        }
    }

    pub(crate) fn routing_score_for_motif(
        &self,
        motif: &RuntimeMotifField,
        distance: f32,
        live_median_radius: f32,
    ) -> f32 {
        if self.ablate_conflict_routing {
            return distance;
        }
        let structured_context =
            self.current_turn_structure_bias >= STRUCTURED_REENTRY_PROMPT_THRESHOLD;
        let task_anchor_similarity = self.routing_task_anchor_similarity(motif);
        let (effective_role, escalated) =
            self.effective_routing_role(motif, task_anchor_similarity);
        let wide_basin_penalty = if structured_context
            && effective_role == "conversational"
            && motif.radius_mean > live_median_radius
        {
            ((motif.radius_mean - live_median_radius) * 6.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let gap_penalty = motif.sequential_gap_rate.clamp(0.0, 1.0);
        let tightness_bonus = if structured_context
            && matches!(
                effective_role.as_str(),
                "structured" | "structured_candidate"
            ) {
            (motif.tightness_score * (0.7 + motif.topology_density * 0.3)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let task_utility_bonus = if structured_context {
            task_anchor_similarity.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let neutral_basin_penalty = if structured_context && effective_role == "neutral" {
            (1.0 - task_anchor_similarity).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let task_aligned_bonus = if structured_context
            && matches!(
                effective_role.as_str(),
                "structured_candidate" | "structured"
            ) {
            (motif.tightness_score * 0.4
                + motif.topology_density * 0.3
                + task_anchor_similarity * 0.3)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let stickiness_bonus = self.stickiness_score_for_motif(&motif.motif_id);
        let restored_compact_penalty = if structured_context
            && motif.promotion_status == "restored_compact"
            && effective_role != "structured"
        {
            0.04
        } else {
            0.0
        };
        // DEEP_DIVE_ROADMAP P2-B: when motif_routing_consensus_weight is on,
        // replace the additive conflict/mixed penalty with a softmax-shape
        // term that combines persistence (preferred) with conflict (penalized).
        // High persistence + low conflict ratio = large w_i = large negative
        // score contribution = lower total score = preferred motif.
        let conflict_penalty = if self.motif_routing_consensus_weight {
            let c_conflict = 0.08 * motif.conflict_ratio + 0.03 * motif.mixed_ratio;
            -(motif.persistence_score - c_conflict).exp()
        } else {
            0.08 * motif.conflict_ratio + 0.03 * motif.mixed_ratio
        };
        (distance
            + conflict_penalty
            + ROUTING_STRUCTURED_WIDE_PENALTY * wide_basin_penalty
            + ROUTING_GAP_PENALTY_SCALE * gap_penalty
            - ROUTING_TIGHTNESS_BONUS_SCALE * tightness_bonus
            - self.task_utility_bonus_scale() * task_utility_bonus
            - self.structured_candidate_bonus_scale() * task_aligned_bonus
            - if escalated { 0.035 } else { 0.0 }
            + self.neutral_basin_penalty_scale() * neutral_basin_penalty
            - 0.03 * motif.routing_safety_score
            + restored_compact_penalty
            - stickiness_bonus)
            .clamp(-1.0, 3.0)
    }

    pub(crate) fn run_periodic_controller(&mut self, probe: &Tensor) -> Result<()> {
        if !self.controller_active()
            || self.current_step % ROUTING_CONTROLLER_INTERVAL != 0
            || self.runtime_motifs.is_empty()
        {
            return Ok(());
        }

        self.refresh_runtime_motif_metadata()?;

        let probe_flat = probe.flatten_all()?;
        let probe_norm = probe_flat.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        if probe_norm <= 1e-6 {
            return Ok(());
        }
        let probe_normalized = probe_flat
            .broadcast_div(&Tensor::new(probe_norm, probe.device())?)?
            .detach();
        let live_median_radius = self.live_median_radius();
        let structured_context =
            self.current_turn_structure_bias >= STRUCTURED_REENTRY_PROMPT_THRESHOLD;

        let mut nearest = Vec::new();
        for motif in &self.runtime_motifs {
            let alignment = self.compute_similarity(&probe_normalized, &motif.vector)?;
            let distance = (1.0 - alignment).max(0.0);
            let task_anchor_similarity = self.routing_task_anchor_similarity(motif);
            let (effective_role, escalated) =
                self.effective_routing_role(motif, task_anchor_similarity);
            nearest.push((
                motif.motif_id.clone(),
                effective_role,
                motif.promotion_status.clone(),
                distance,
                self.routing_score_for_motif(motif, distance, live_median_radius),
                task_anchor_similarity,
                escalated,
            ));
        }
        nearest.sort_by(|a, b| a.3.total_cmp(&b.3));
        nearest.truncate(ROUTING_CONTROLLER_TOP_K);
        if nearest.is_empty() {
            return Ok(());
        }

        let nearest_id = nearest[0].0.clone();
        let nearest_distance = nearest[0].3;
        let mut routed = nearest.clone();
        routed.sort_by(|a, b| a.4.total_cmp(&b.4));
        let mut selected = routed[0].clone();

        if let Some(cache) = self.active_routing_cache().cloned() {
            if let Some(prev_motif) = self
                .runtime_motifs
                .iter()
                .find(|motif| motif.motif_id == cache.motif_id)
            {
                let prev_alignment =
                    self.compute_similarity(&probe_normalized, &prev_motif.vector)?;
                let prev_distance = (1.0 - prev_alignment).max(0.0);
                let prev_score =
                    self.routing_score_for_motif(prev_motif, prev_distance, live_median_radius);
                if selected.4 >= prev_score - 0.005 {
                    selected = (
                        prev_motif.motif_id.clone(),
                        self.effective_routing_role(
                            prev_motif,
                            self.routing_task_anchor_similarity(prev_motif),
                        )
                        .0,
                        prev_motif.promotion_status.clone(),
                        prev_distance,
                        prev_score,
                        self.routing_task_anchor_similarity(prev_motif),
                        false,
                    );
                }
            }
        }

        if nearest.len() > 1
            && (nearest[1].3 - nearest_distance).abs() <= ROUTING_TIE_BREAK_MARGIN
            && selected.0 != nearest_id
        {
            self.conflict_tie_break_count += 1;
        }

        self.last_controller_candidates = self
            .runtime_motifs
            .iter()
            .filter_map(|motif| {
                nearest
                    .iter()
                    .find(|candidate| candidate.0 == motif.motif_id)
                    .map(|candidate| ControllerCandidateRecord {
                        motif_id: motif.motif_id.clone(),
                        motif_role: candidate.1.clone(),
                        promotion_status: motif.promotion_status.clone(),
                        distance: candidate.3,
                        routing_score: candidate.4,
                        task_anchor_similarity: candidate.5,
                        topology_density: motif.topology_density,
                        sequential_gap_rate: motif.sequential_gap_rate,
                        tension_anchor_strength: motif.tension_anchor_strength,
                        tightness_score: motif.tightness_score,
                    })
            })
            .collect();

        if structured_context {
            if self
                .last_controller_candidates
                .iter()
                .any(|candidate| candidate.task_anchor_similarity > 0.0)
            {
                self.task_utility_bonus_applied += 1;
            }
            if self
                .last_controller_candidates
                .iter()
                .any(|candidate| candidate.motif_role == "neutral")
            {
                self.neutral_basin_penalty_applied += 1;
            }
        }

        self.structured_candidate_escalation_attempts +=
            nearest.iter().filter(|candidate| candidate.6).count();

        let best_structured_candidate = self
            .last_controller_candidates
            .iter()
            .filter(|candidate| {
                matches!(
                    candidate.motif_role.as_str(),
                    "structured" | "structured_candidate"
                )
            })
            .min_by(|a, b| a.routing_score.total_cmp(&b.routing_score))
            .cloned();

        let structured_candidate_loss_reason =
            if structured_context && matches!(selected.1.as_str(), "neutral" | "conversational") {
                if let Some(candidate) = &best_structured_candidate {
                    let reason = if selected.3 + 0.015 < candidate.distance {
                        "distance_deficit"
                    } else if self
                        .runtime_motifs
                        .iter()
                        .find(|motif| motif.motif_id == candidate.motif_id)
                        .map(|motif| motif.conflict_ratio)
                        .unwrap_or(0.0)
                        > self
                            .runtime_motifs
                            .iter()
                            .find(|motif| motif.motif_id == selected.0)
                            .map(|motif| motif.conflict_ratio)
                            .unwrap_or(0.0)
                            + 0.05
                    {
                        "conflict_penalty"
                    } else if candidate.sequential_gap_rate > 0.45 {
                        "topology_gap"
                    } else if candidate.task_anchor_similarity + 0.05
                        < self
                            .last_controller_candidates
                            .iter()
                            .find(|record| record.motif_id == selected.0)
                            .map(|record| record.task_anchor_similarity)
                            .unwrap_or(0.0)
                    {
                        "task_anchor_misalignment"
                    } else {
                        "role_threshold_miss"
                    };
                    *self
                        .structured_candidate_loss_reason_counts
                        .entry(reason.to_string())
                        .or_insert(0) += 1;
                    Some(reason.to_string())
                } else {
                    None
                }
            } else {
                None
            };

        for motif in self.runtime_motifs.iter_mut() {
            if nearest
                .iter()
                .any(|candidate| candidate.0 == motif.motif_id)
            {
                if motif.motif_id == selected.0 {
                    motif.controller_selected_count += 1;
                } else {
                    motif.controller_rejected_count += 1;
                }
            }
        }

        self.controller_tick_count += 1;
        if selected.1 == "structured" {
            self.controller_selected_structured_count += 1;
            if selected.6 {
                self.structured_candidate_escalation_wins += 1;
            }
            if self.structured_resume_window_remaining > 0 {
                self.structured_basin_lock_count += 1;
            }
        } else if selected.1 == "structured_candidate" {
            self.controller_selected_structured_candidate_count += 1;
            if self.structured_resume_window_remaining > 0 {
                self.structured_basin_lock_count += 1;
            }
        } else if selected.1 == "conversational" {
            self.controller_selected_conversational_count += 1;
            if self.structured_resume_window_remaining > 0 {
                self.structured_resume_conversational_hits += 1;
            }
        }
        self.last_routed_motif_id = Some(selected.0.clone());
        self.last_routed_motif_role = Some(selected.1.clone());
        self.last_routed_motif_score = selected.4;
        self.apply_routing_stickiness(&selected.0, &selected.1);
        self.decay_routing_stickiness();
        self.routing_cache = Some(RoutingDecisionCache {
            motif_id: selected.0.clone(),
            motif_role: selected.1.clone(),
            routing_score: selected.4,
            expires_at_step: self.current_step + ROUTING_CACHE_LIFETIME,
        });

        println!(
            " [CONTROLLER_ROUTE] step={} selected={} role={} status={} distance={:.3} routing_score={:.3} nearest={} tie_breaks={}",
            self.current_step,
            selected.0,
            selected.1,
            selected.2,
            selected.3,
            selected.4,
            nearest_id,
            self.conflict_tie_break_count
        );
        emit_ui_event_value(
            self.ui_events_json,
            "controller_route",
            serde_json::json!({
                "step": self.current_step,
                "selected_motif_id": selected.0,
                "selected_role": selected.1,
                "selected_status": selected.2,
                "distance": selected.3,
                "routing_score": selected.4,
                "nearest_motif_id": nearest_id,
                "conflict_tie_break_count": self.conflict_tie_break_count,
                "structured_candidate_escalated": selected.6,
            }),
        );
        if self.task_anchor_similarity_start <= 0.0 {
            self.update_task_anchor_similarity_snapshot("start");
        }
        self.maybe_capture_hinge_window("controller_tick", structured_candidate_loss_reason);

        Ok(())
    }

    pub(crate) fn update_structured_streak(&mut self, particle_structure_signal: f32) {
        if particle_structure_signal >= STRUCTURED_STREAK_SIGNAL_THRESHOLD {
            self.structured_streak += 1;
            self.max_structured_streak = self.max_structured_streak.max(self.structured_streak);
        } else if self.structured_streak > 0 {
            self.structured_streak = self.structured_streak.saturating_sub(1);
        }
    }

    pub(crate) fn apply_crystal_ratchet(motif: &mut RuntimeMotifField, structured_streak: usize) {
        if structured_streak < STRUCTURED_RATCHET_MIN_STREAK {
            return;
        }
        let streak_strength =
            ((structured_streak - STRUCTURED_RATCHET_MIN_STREAK + 1) as f32 / 4.0).clamp(0.0, 1.0);
        let ratchet =
            (motif.structured_signal * motif.tightness_score * streak_strength).clamp(0.0, 1.0);
        if ratchet <= 0.0 {
            return;
        }
        let radius_shrink = (1.0 - ratchet * 0.18).clamp(0.72, 1.0);
        let std_shrink = (1.0 - ratchet * 0.24).clamp(0.64, 1.0);
        motif.radius_mean *= radius_shrink;
        motif.radius_std *= std_shrink;
        motif.radius_m2 *= std_shrink * std_shrink;
        motif.tightness_score = motif_tightness(motif.radius_mean, motif.radius_std);
        motif.readiness_score = (motif.readiness_score + ratchet * 0.08).clamp(0.0, 1.0);
    }

    pub(crate) fn apply_reentry_clamp(
        &mut self,
        motif_kind: &str,
        promotion_status: &str,
        structured_signal: f32,
        tightness_score: f32,
    ) {
        if self.ablate_reentry_clamp {
            return;
        }
        let promoted_like = motif_kind == "promoted" || promotion_status == "recovered_promoted";
        let clamp_drive = (structured_signal * 0.45
            + tightness_score * 0.35
            + if promoted_like { 0.20 } else { 0.0 })
        .clamp(0.0, 1.0);
        if clamp_drive <= 0.0 {
            return;
        }
        let steps = (REENTRY_CLAMP_MIN_STEPS as f32
            + clamp_drive * (REENTRY_CLAMP_MAX_STEPS - REENTRY_CLAMP_MIN_STEPS) as f32)
            .round() as usize;
        self.reentry_clamp_steps_remaining = self.reentry_clamp_steps_remaining.max(steps);
        self.reentry_clamp_strength = self
            .reentry_clamp_strength
            .max((0.55 + clamp_drive * 0.35).clamp(0.55, 0.95));
        self.reentry_temp_scale = self
            .reentry_temp_scale
            .min((0.72 - clamp_drive * 0.22).clamp(0.38, 0.72));
    }

    pub(crate) fn maybe_release_reentry_clamp(&mut self) {
        if self.ablate_reentry_clamp {
            self.reentry_clamp_steps_remaining = 0;
            self.reentry_clamp_strength = 0.0;
            self.reentry_temp_scale = 1.0;
            return;
        }
        if self.reentry_clamp_steps_remaining == 0 {
            self.reentry_clamp_strength = 0.0;
            self.reentry_temp_scale = 1.0;
            return;
        }
        let max_tension = self
            .runtime_recovery_ops
            .iter()
            .map(|op| op.tension_point)
            .fold(0.0f32, f32::max);
        if self.last_trap_score >= 0.95 || self.last_absence_signal >= 1.25 || max_tension >= 1.05 {
            println!(
                " [REENTRY_CLAMP] released trap={:.2} absence={:.2} tension={:.2}",
                self.last_trap_score, self.last_absence_signal, max_tension
            );
            self.reentry_clamp_steps_remaining = 0;
            self.reentry_clamp_strength = 0.0;
            self.reentry_temp_scale = 1.0;
        }
    }

    pub(crate) fn log_motif_promotion_attempt(
        &mut self,
        motif: &RuntimeMotifField,
        promoted: bool,
        failure_reason: Option<&str>,
        crystal_signal: f32,
    ) {
        self.promotion_attempt_count += 1;
        if self.first_promotion_attempt_step.is_none() {
            self.first_promotion_attempt_step = Some(self.current_step);
        }
        if promoted {
            match motif.promotion_status.as_str() {
                "recovered_promoted" => {
                    self.first_recovered_promoted_step
                        .get_or_insert(self.current_step);
                }
                "promoted" => {
                    self.first_organic_promoted_step
                        .get_or_insert(self.current_step);
                }
                _ => {}
            }
            if self.task_anchor_similarity_hinge <= 0.0 {
                self.update_task_anchor_similarity_snapshot("hinge");
            }
        } else {
            self.promotion_failure_count += 1;
        }

        let status = if promoted { "promoted" } else { "failed" };
        let reason = failure_reason.unwrap_or("-");
        println!(
            " [MOTIF_ATTEMPT] status={} id={} kind={} promo_status={} score={:.3} crystal={:.3} structure={:.3} tightness={:.3} members={} reason={}",
            status,
            motif.motif_id,
            motif.motif_kind,
            motif.promotion_status,
            motif.promotion_score,
            crystal_signal,
            motif.structured_signal,
            motif.tightness_score,
            motif.member_count,
            reason
        );
        emit_ui_event_value(
            self.ui_events_json,
            "motif_promotion_attempt",
            serde_json::json!({
                "step": self.current_step,
                "status": status,
                "motif_id": motif.motif_id,
                "motif_kind": motif.motif_kind,
                "promotion_status": motif.promotion_status,
                "promotion_score": motif.promotion_score,
                "structured_signal": motif.structured_signal,
                "tightness_score": motif.tightness_score,
                "member_count": motif.member_count,
                "crystal_signal": crystal_signal,
                "failure_reason": failure_reason,
            }),
        );
        self.maybe_capture_hinge_window(
            if promoted {
                "hinge_flip"
            } else {
                "promotion_attempt"
            },
            failure_reason.map(|reason| reason.to_string()),
        );
    }

    pub(crate) fn compact_runtime_motif_anchor(&self) -> Result<Option<(Vec<f32>, f32)>> {
        let mut candidates: Vec<(&RuntimeMotifField, f32, bool)> = self
            .runtime_motifs
            .iter()
            .filter(|motif| {
                motif.motif_kind == "promoted" && !Self::is_restored_compact_promoted(motif)
            })
            .map(|motif| {
                let density = ((motif.member_count.saturating_sub(1)) as f32 / 5.0).clamp(0.0, 1.0);
                let weight = (0.35 + motif.promotion_score * 0.45 + density * 0.20).clamp(0.1, 1.2);
                (motif, weight, true)
            })
            .collect();

        if candidates.is_empty() {
            candidates = self
                .runtime_motifs
                .iter()
                .filter(|motif| motif.motif_kind == "promoted")
                .map(|motif| {
                    let density =
                        ((motif.member_count.saturating_sub(1)) as f32 / 5.0).clamp(0.0, 1.0);
                    let weight =
                        (0.35 + motif.promotion_score * 0.45 + density * 0.20).clamp(0.1, 1.2);
                    (motif, weight, true)
                })
                .collect();
        }

        if candidates.is_empty() {
            candidates = self
                .runtime_motifs
                .iter()
                .filter(|motif| {
                    motif.motif_kind == "live"
                        && motif.promotion_status != "seeded"
                        && motif.member_count >= 2
                })
                .map(|motif| {
                    let density =
                        ((motif.member_count.saturating_sub(1)) as f32 / 5.0).clamp(0.0, 1.0);
                    let weight = (0.22
                        + motif.promotion_score * 0.35
                        + density * 0.20
                        + motif.readiness_score * 0.15)
                        .clamp(0.08, 0.90);
                    (motif, weight, false)
                })
                .collect();
        }

        if candidates.is_empty() {
            return Ok(None);
        }

        candidates.sort_by(
            |(motif_a, weight_a, promoted_a), (motif_b, weight_b, promoted_b)| {
                promoted_b
                    .cmp(promoted_a)
                    .then_with(|| {
                        weight_b
                            .partial_cmp(weight_a)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then_with(|| motif_b.member_count.cmp(&motif_a.member_count))
                    .then_with(|| {
                        motif_b
                            .promotion_score
                            .partial_cmp(&motif_a.promotion_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            },
        );
        candidates.truncate(COMPACT_PROMOTED_MOTIF_TOP_K);

        let device = self.charge_tensor.device();
        let mut centroid = Tensor::zeros((self.hidden_dim,), DType::F32, device)?;
        let mut total_weight = 0.0f32;
        let mut signal_sum = 0.0f32;
        let mut any_promoted = false;
        for (motif, weight, is_promoted) in &candidates {
            let weight_t = Tensor::new(*weight, device)?;
            centroid = (centroid + motif.vector.broadcast_mul(&weight_t)?)?;
            total_weight += *weight;
            signal_sum += motif.promotion_score;
            any_promoted |= *is_promoted;
        }
        if total_weight <= 1e-6 {
            return Ok(None);
        }

        let centroid = centroid.broadcast_div(&Tensor::new(total_weight, device)?)?;
        let compressed = compress_tensor_to_dim(&centroid, 32)?;
        let base_signal = (signal_sum / candidates.len() as f32).clamp(0.0, 1.0);
        let compact_signal = if any_promoted {
            base_signal
        } else {
            base_signal.min(0.58)
        };
        Ok(Some((compressed, compact_signal)))
    }

    pub(crate) fn decay_restored_compact_promoted(&mut self) -> usize {
        let mut decayed = 0usize;
        let current_step = self.current_step;
        for motif in self.runtime_motifs.iter_mut() {
            if !Self::is_restored_compact_promoted(motif) {
                continue;
            }
            let age = current_step.saturating_sub(motif.last_updated_step);
            if age >= RESTORED_PROMOTED_DECAY_STEP_WINDOW
                && motif.member_count < RESTORED_PROMOTED_RECOVERY_MEMBER_FLOOR
            {
                motif.motif_kind = "live".to_string();
                motif.promotion_status = "restored_fading".to_string();
                motif.promotion_score = (motif.promotion_score * 0.65).clamp(0.0, 1.0);
                decayed += 1;
            }
        }
        decayed
    }

    pub(crate) fn prune_stale_live_motifs(&mut self) -> usize {
        let before = self.runtime_motifs.len();
        let current_step = self.current_step;
        let tuning = continuity_mode_tuning(self.runtime_mode);
        let empathy = self.empathy_spike.clamp(0.0, 2.0);
        let stale_window = if self.motif_regression_assist_steps_remaining > 0 {
            LIVE_MOTIF_STALE_STEP_WINDOW
                + (self.motif_regression_assist_strength * 64.0).round() as usize
        } else {
            LIVE_MOTIF_STALE_STEP_WINDOW
        } as f32
            * tuning.prune_window_scale
            * (1.0 + empathy * 0.25);
        let prune_promotion_max = if self.motif_regression_assist_steps_remaining > 0 {
            (LIVE_MOTIF_PRUNE_PROMOTION_MAX + self.motif_regression_assist_strength * 0.18)
                .clamp(LIVE_MOTIF_PRUNE_PROMOTION_MAX, 0.70)
        } else {
            LIVE_MOTIF_PRUNE_PROMOTION_MAX
        } * tuning.prune_threshold_scale
            - empathy * 0.05;
        let stale_window = stale_window.round() as usize;
        let prune_promotion_max =
            prune_promotion_max.clamp(LIVE_MOTIF_PRUNE_PROMOTION_MAX * 0.55, 0.75);
        self.runtime_motifs.retain(|motif| {
            if motif.motif_kind != "live" {
                return true;
            }
            let age = current_step.saturating_sub(motif.last_updated_step);
            let stale = age >= stale_window;
            let weak_seed = motif.member_count <= LIVE_MOTIF_PRUNE_MEMBER_MAX
                && motif.promotion_score <= prune_promotion_max
                && motif.promotion_status == "seeded";
            !(stale && weak_seed)
        });
        before.saturating_sub(self.runtime_motifs.len())
    }

    pub(crate) fn inject_sentence_context_motif(
        &mut self,
        anchor: &Tensor,
        context_signal: f32,
    ) -> Result<()> {
        if context_signal <= 0.05 {
            return Ok(());
        }

        let anchor_detached = anchor.detach();
        let raw_signature = compress_hidden_state_to_64d(&anchor_detached)?;
        let member_count = (2.0 + context_signal * 3.0).round().clamp(2.0, 5.0) as usize;
        let mut replaced = false;
        for motif in self.runtime_motifs.iter_mut() {
            if motif.source == "secret_sauce::sentence_context" {
                motif.vector = anchor_detached.clone();
                motif.raw_signature = raw_signature.clone();
                motif.motif_kind = "live".to_string();
                motif.promotion_status = "restored_context".to_string();
                motif.member_count = motif.member_count.max(member_count);
                motif.promotion_score = motif.promotion_score.max(context_signal * 0.75);
                motif.persistence_score = motif
                    .persistence_score
                    .max((0.25 + context_signal * 0.25).clamp(0.25, 0.70));
                motif.readiness_score = motif
                    .readiness_score
                    .max((0.30 + context_signal * 0.30).clamp(0.30, 0.80));
                motif.structured_signal = motif.structured_signal.max(0.0);
                motif.tightness_score = motif.tightness_score.max(motif_tightness(0.10, 0.03));
                motif.last_updated_step = self.current_step;
                replaced = true;
                break;
            }
        }

        if !replaced {
            self.runtime_motifs.push(RuntimeMotifField {
                motif_id: format!("live::context::{:04}", self.current_step),
                source: "secret_sauce::sentence_context".to_string(),
                motif_kind: "live".to_string(),
                promotion_status: "restored_context".to_string(),
                raw_signature,
                vector: anchor_detached,
                member_count,
                last_updated_step: self.current_step,
                persistence_score: (0.25 + context_signal * 0.25).clamp(0.25, 0.70),
                readiness_score: (0.30 + context_signal * 0.30).clamp(0.30, 0.80),
                injection_strength: (0.08 + context_signal * 0.10).clamp(0.08, 0.24),
                max_pre_energy: 0.0,
                flip_rate: 0.0,
                orbit_count: context_signal,
                radius_mean: 0.10,
                radius_std: 0.03,
                radius_m2: 0.0009,
                promotion_score: (context_signal * 0.75).clamp(0.0, 0.70),
                structured_signal: 0.0,
                tightness_score: motif_tightness(0.10, 0.03),
                conflict_ratio: 0.0,
                mixed_ratio: 0.0,
                routing_safety_score: 1.0,
                topology_density: 0.0,
                sequential_gap_rate: 0.0,
                fragmentation: 0.0,
                hole_pressure: 0.0,
                tension_anchor_strength: 0.0,
                motif_role: "neutral".to_string(),
                controller_selected_count: 0,
                controller_rejected_count: 0,
                origin_run_id: self.current_run_id.clone(),
                promotion_epoch: 0,
                parent_motif_ids: Vec::new(),
                provenance_summary: "live::secret_sauce::restored_context".to_string(),
                merge_key: format!(
                    "{}::0::live::context::{:04}",
                    self.current_run_id, self.current_step
                ),
                task_anchor_signature: None,
                live_hidden_remapped: true,
            });
        }

        sort_runtime_motifs_by_priority(&mut self.runtime_motifs);
        self.refresh_runtime_motif_metadata()?;
        Ok(())
    }

    pub(crate) fn inject_compact_runtime_motif(
        &mut self,
        anchor: &Tensor,
        compact_signal: f32,
    ) -> Result<()> {
        if compact_signal <= 0.05 {
            return Ok(());
        }

        if compact_signal < 0.60 {
            return self.inject_sentence_context_motif(anchor, compact_signal);
        }

        let anchor_detached = anchor.detach();
        let raw_signature = compress_hidden_state_to_64d(&anchor_detached)?;
        let member_count = (2.0 + compact_signal * 4.0).round().clamp(2.0, 6.0) as usize;
        let mut replaced = false;
        for motif in self.runtime_motifs.iter_mut() {
            if motif.source == "secret_sauce::motif_anchor"
                || motif.source == "secret_sauce::promoted_anchor"
            {
                motif.vector = anchor_detached.clone();
                motif.raw_signature = raw_signature.clone();
                motif.motif_kind = "promoted".to_string();
                motif.promotion_status = "restored_compact".to_string();
                motif.member_count = motif.member_count.max(member_count);
                motif.promotion_score = motif.promotion_score.max(compact_signal);
                motif.structured_signal = motif.structured_signal.max(compact_signal);
                motif.tightness_score = motif.tightness_score.max(motif_tightness(0.08, 0.02));
                motif.last_updated_step = self.current_step;
                motif.source = "secret_sauce::motif_anchor".to_string();
                replaced = true;
                break;
            }
        }

        if !replaced {
            self.runtime_motifs.push(RuntimeMotifField {
                motif_id: format!("promoted::compact::{:04}", self.current_step),
                source: "secret_sauce::motif_anchor".to_string(),
                motif_kind: "promoted".to_string(),
                promotion_status: "restored_compact".to_string(),
                raw_signature,
                vector: anchor_detached,
                member_count,
                last_updated_step: self.current_step,
                persistence_score: (0.45 + compact_signal * 0.35).clamp(0.45, 0.90),
                readiness_score: (0.40 + compact_signal * 0.40).clamp(0.40, 0.95),
                injection_strength: (0.10 + compact_signal * 0.20).clamp(0.10, 0.35),
                max_pre_energy: 0.0,
                flip_rate: 0.0,
                orbit_count: compact_signal,
                radius_mean: 0.08,
                radius_std: 0.02,
                radius_m2: 0.0004,
                promotion_score: compact_signal,
                structured_signal: compact_signal,
                tightness_score: motif_tightness(0.08, 0.02),
                conflict_ratio: 0.0,
                mixed_ratio: 0.0,
                routing_safety_score: 1.0,
                topology_density: 0.0,
                sequential_gap_rate: 0.0,
                fragmentation: 0.0,
                hole_pressure: 0.0,
                tension_anchor_strength: 0.0,
                motif_role: "neutral".to_string(),
                controller_selected_count: 0,
                controller_rejected_count: 0,
                origin_run_id: self.current_run_id.clone(),
                promotion_epoch: self.current_step,
                parent_motif_ids: vec!["secret_sauce::motif_anchor".to_string()],
                provenance_summary: "promoted::secret_sauce::restored_compact".to_string(),
                merge_key: format!(
                    "{}::{}::promoted::compact::{:04}",
                    self.current_run_id, self.current_step, self.current_step
                ),
                task_anchor_signature: self.current_task_anchor_signature.clone(),
                live_hidden_remapped: true,
            });
        }

        sort_runtime_motifs_by_priority(&mut self.runtime_motifs);
        self.refresh_runtime_motif_metadata()?;
        Ok(())
    }

    pub(crate) fn refresh_live_motif_promotion(
        motif: &mut RuntimeMotifField,
        empathy_spike: f32,
        structured_streak: usize,
        current_step: usize,
        ablate_promotion_override: bool,
        task_anchor_signature: Option<&Vec<f32>>,
    ) -> Option<(bool, Option<&'static str>, f32)> {
        let density = ((motif.member_count.saturating_sub(1)) as f32 / 5.0).clamp(0.0, 1.0);
        let tightness = motif_tightness(motif.radius_mean, motif.radius_std);
        let persistence = motif.persistence_score.clamp(0.0, 1.0);
        let empathy = (empathy_spike / 2.0).clamp(0.0, 1.0);
        let structure = motif.structured_signal.clamp(0.0, 1.0);
        let crystal_signal = (structure * tightness).clamp(0.0, 1.0);
        let topology_density = motif.topology_density.clamp(0.0, 1.0);
        let gap_penalty = motif.sequential_gap_rate.clamp(0.0, 1.0);
        let streak_bonus = if structured_streak >= STRUCTURED_RATCHET_MIN_STREAK {
            ((structured_streak - STRUCTURED_RATCHET_MIN_STREAK + 1) as f32 / 4.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let empathy_memory_bonus = empathy * (0.03 + tightness * 0.05);
        motif.promotion_score = (density * 0.28
            + tightness * 0.20
            + persistence * 0.14
            + motif.readiness_score.clamp(0.0, 1.0) * 0.12
            + structure * 0.12
            + crystal_signal * 0.10
            + topology_density * 0.08
            + streak_bonus * 0.08
            + empathy_memory_bonus
            - gap_penalty * 0.05)
            .clamp(0.0, 1.0);
        motif.tightness_score = tightness;

        if motif.motif_kind == "bridge" {
            motif.promotion_status = "imported".to_string();
            motif.promotion_score = 1.0;
            return None;
        }

        let was_restored_compact = (motif.source == "secret_sauce::motif_anchor"
            || motif.source == "secret_sauce::promoted_anchor")
            && motif.promotion_status == "restored_compact";
        let clamp_like_override = !ablate_promotion_override
            && structured_streak >= 1
            && tightness >= 0.84
            && structure >= LIVE_MOTIF_STRUCTURED_SIGNAL_THRESHOLD;
        let promotion_member_floor = if clamp_like_override {
            1
        } else if !ablate_promotion_override
            && (crystal_signal >= 0.40 || structured_streak >= STRUCTURED_PROMOTION_OVERRIDE_STREAK)
        {
            2
        } else {
            3
        };
        let should_attempt = motif.promotion_score >= LIVE_MOTIF_PROMOTION_ATTEMPT_THRESHOLD
            || (crystal_signal >= 0.32 && motif.member_count >= 2)
            || clamp_like_override;
        let mut promotion_threshold = if was_restored_compact {
            LIVE_MOTIF_CRYSTAL_PROMOTION_THRESHOLD - 0.04
        } else {
            LIVE_MOTIF_CRYSTAL_PROMOTION_THRESHOLD
        };
        if structured_streak >= STRUCTURED_RATCHET_MIN_STREAK {
            promotion_threshold -= 0.06 * streak_bonus;
        }

        if (motif.promotion_score >= promotion_threshold
            || (!ablate_promotion_override
                && structured_streak >= STRUCTURED_PROMOTION_OVERRIDE_STREAK
                && tightness >= 0.78
                && structure >= LIVE_MOTIF_STRUCTURED_SIGNAL_THRESHOLD
                && motif.member_count >= 2))
            && motif.member_count >= promotion_member_floor
            && (structure >= LIVE_MOTIF_STRUCTURED_SIGNAL_THRESHOLD
                || crystal_signal >= 0.42
                || motif.member_count >= 4)
        {
            motif.motif_kind = "promoted".to_string();
            motif.promotion_status = if was_restored_compact {
                "recovered_promoted".to_string()
            } else {
                "promoted".to_string()
            };
            if motif.promotion_epoch == 0 {
                motif.promotion_epoch = current_step.max(1);
            }
            // Phase 3: Attach the immutable task anchor payload on promotion.
            // This ensures the reasoning basin carries the 64D task signature
            // through the phase transition.
            if motif.task_anchor_signature.is_none() {
                if let Some(sig) = task_anchor_signature {
                    motif.task_anchor_signature = Some(sig.clone());
                }
            }
            return should_attempt.then_some((true, None, crystal_signal));
        } else if motif.member_count >= 2 || motif.promotion_score >= 0.40 {
            motif.motif_kind = "live".to_string();
            motif.promotion_status = if was_restored_compact {
                "restored_fading".to_string()
            } else {
                "reinforcing".to_string()
            };
        } else {
            motif.motif_kind = "live".to_string();
            motif.promotion_status = if was_restored_compact {
                "restored_fading".to_string()
            } else {
                "seeded".to_string()
            };
        }

        let failure_reason = if !should_attempt {
            None
        } else if motif.member_count < promotion_member_floor {
            Some("insufficient_recurrence")
        } else if tightness < 0.45 && structure >= LIVE_MOTIF_STRUCTURED_SIGNAL_THRESHOLD {
            Some("radius_too_wide")
        } else if motif.sequential_gap_rate > 0.55 && motif.topology_density < 0.40 {
            Some("fragmented_topology")
        } else if structure < LIVE_MOTIF_STRUCTURED_SIGNAL_THRESHOLD && motif.member_count < 4 {
            Some("insufficient_structure")
        } else {
            Some("promotion_score_below_threshold")
        };

        should_attempt.then_some((false, failure_reason, crystal_signal))
    }

    pub(crate) fn mint_or_update_live_motif_from_last_sentence(&mut self) -> Result<()> {
        let Some(last_particle) = self.sentence_history.back() else {
            return Ok(());
        };
        let particle_text = last_particle.text.clone();
        let last_particle_fitness = last_particle.fitness;
        let particle_position = last_particle.position.clone();
        let particle_step = last_particle.birth_step;
        let particle_structure_signal = structured_reasoning_signal(&particle_text)
            .max(self.current_turn_structure_bias * 0.60);
        self.update_structured_streak(particle_structure_signal);
        let structured_override = particle_structure_signal >= STRUCTURED_STREAK_SIGNAL_THRESHOLD
            || self.current_turn_structure_bias >= STRUCTURED_REENTRY_PROMPT_THRESHOLD;
        if last_particle_fitness < LIVE_MOTIF_SEED_FITNESS_FLOOR && !structured_override {
            if self.stdout_debug() {
                println!(
                    " [LIVE_MOTIF] skipped fitness={:.3} structure={:.2} text=\"{}\"",
                    last_particle_fitness,
                    particle_structure_signal,
                    particle_text
                        .chars()
                        .take(72)
                        .collect::<String>()
                        .replace('\n', " ")
                );
            }
            return Ok(());
        }

        let particle_fitness = if structured_override {
            last_particle_fitness
                .max(LIVE_MOTIF_SEED_FITNESS_FLOOR + particle_structure_signal * 0.12)
        } else {
            last_particle_fitness
        };

        let mut best_match: Option<(usize, f32)> = None;
        for (idx, motif) in self.runtime_motifs.iter().enumerate() {
            if !Self::is_live_runtime_motif(motif) {
                continue;
            }
            let sim = self.compute_similarity(&particle_position, &motif.vector)?;
            let distance = (1.0 - sim).max(0.0);
            let threshold = Self::live_motif_merge_threshold(motif);
            if distance <= threshold {
                match best_match {
                    Some((_, best_distance)) if distance >= best_distance => {}
                    _ => best_match = Some((idx, distance)),
                }
            }
        }

        if let Some((idx, distance)) = best_match {
            let (motif_clone, log_line) = {
                let motif = &mut self.runtime_motifs[idx];
                let old_count = motif.member_count.max(1) as f32;
                let new_count = old_count + 1.0;

                let old_weight = Tensor::new(old_count / new_count, particle_position.device())?;
                let new_weight = Tensor::new(1.0 / new_count, particle_position.device())?;
                let blended = (motif.vector.broadcast_mul(&old_weight)?
                    + particle_position.broadcast_mul(&new_weight)?)?;
                let norm = blended.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
                motif.vector = if norm > 1e-6 {
                    blended
                        .broadcast_div(&Tensor::new(norm, particle_position.device())?)?
                        .detach()
                } else {
                    blended.detach()
                };

                motif.member_count += 1;
                motif.last_updated_step = particle_step;
                motif.structured_signal =
                    ((motif.structured_signal * old_count) + particle_structure_signal) / new_count;
                let delta = distance - motif.radius_mean;
                motif.radius_mean += delta / motif.member_count as f32;
                let delta2 = distance - motif.radius_mean;
                motif.radius_m2 += delta * delta2;
                motif.radius_std = if motif.member_count > 1 {
                    (motif.radius_m2 / (motif.member_count as f32 - 1.0))
                        .max(0.0)
                        .sqrt()
                } else {
                    0.0
                };
                motif.tightness_score = motif_tightness(motif.radius_mean, motif.radius_std);
                motif.raw_signature = compress_hidden_state_to_64d(&motif.vector)?;
                Self::refresh_live_motif_scores(
                    motif,
                    particle_fitness,
                    self.empathy_spike,
                    self.current_turn_structure_bias,
                );
                if !self.ablate_crystal_ratchet {
                    Self::apply_crystal_ratchet(motif, self.structured_streak);
                }
                let promotion_outcome = Self::refresh_live_motif_promotion(
                    motif,
                    self.empathy_spike,
                    self.structured_streak,
                    self.current_step,
                    self.ablate_promotion_override,
                    self.current_task_anchor_signature.as_ref(),
                );
                let motif_clone = motif.clone();
                let log_line = (
                    motif.motif_id.clone(),
                    motif.motif_kind.clone(),
                    motif.promotion_status.clone(),
                    motif.promotion_score,
                    motif.member_count,
                    motif.radius_mean,
                    motif.radius_std,
                    motif.structured_signal,
                    motif.tightness_score,
                    promotion_outcome,
                );
                (motif_clone, log_line)
            };

            if let Some((promoted, failure_reason, crystal_signal)) = log_line.9 {
                self.log_motif_promotion_attempt(
                    &motif_clone,
                    promoted,
                    failure_reason,
                    crystal_signal,
                );
            }
            if self.stdout_debug() {
                println!(
                    " [LIVE_MOTIF] merged id={} kind={} status={} score={:.2} members={} dist={:.4} radius={:.4}±{:.4} structure={:.2} tightness={:.2} empathy={:.2}",
                    log_line.0,
                    log_line.1,
                    log_line.2,
                    log_line.3,
                    log_line.4,
                    distance,
                    log_line.5,
                    log_line.6,
                    log_line.7,
                    log_line.8,
                    self.empathy_spike
                );
            }
        } else {
            let motif_id = format!(
                "live::sentence::{:04}::{}",
                particle_step,
                self.runtime_motifs.len()
            );
            let vector = particle_position.detach();
            let raw_signature = compress_hidden_state_to_64d(&vector)?;
            let mut motif = RuntimeMotifField {
                motif_id: motif_id.clone(),
                source: "live::sentence_history".to_string(),
                motif_kind: "live".to_string(),
                promotion_status: "seeded".to_string(),
                raw_signature,
                vector,
                member_count: 1,
                last_updated_step: particle_step,
                persistence_score: 0.05,
                readiness_score: 0.10,
                injection_strength: 0.05,
                max_pre_energy: 0.0,
                flip_rate: 0.0,
                orbit_count: 0.0,
                radius_mean: 0.0,
                radius_std: 0.0,
                radius_m2: 0.0,
                promotion_score: 0.0,
                structured_signal: particle_structure_signal,
                tightness_score: 1.0,
                conflict_ratio: 0.0,
                mixed_ratio: 0.0,
                routing_safety_score: 1.0,
                topology_density: 0.0,
                sequential_gap_rate: 0.0,
                fragmentation: 0.0,
                hole_pressure: 0.0,
                tension_anchor_strength: 0.0,
                motif_role: "neutral".to_string(),
                controller_selected_count: 0,
                controller_rejected_count: 0,
                origin_run_id: self.current_run_id.clone(),
                promotion_epoch: 0,
                parent_motif_ids: Vec::new(),
                provenance_summary: "live::sentence_history::seeded".to_string(),
                merge_key: format!("{}::0::{}", self.current_run_id, motif_id),
                task_anchor_signature: None,
                live_hidden_remapped: true,
            };
            Self::refresh_live_motif_scores(
                &mut motif,
                particle_fitness,
                self.empathy_spike,
                self.current_turn_structure_bias,
            );
            if !self.ablate_crystal_ratchet {
                Self::apply_crystal_ratchet(&mut motif, self.structured_streak);
            }
            if let Some((promoted, failure_reason, crystal_signal)) =
                Self::refresh_live_motif_promotion(
                    &mut motif,
                    self.empathy_spike,
                    self.structured_streak,
                    self.current_step,
                    self.ablate_promotion_override,
                    self.current_task_anchor_signature.as_ref(),
                )
            {
                self.log_motif_promotion_attempt(&motif, promoted, failure_reason, crystal_signal);
            }
            if self.stdout_debug() {
                println!(
                    " [LIVE_MOTIF] seeded id={} kind={} status={} score={:.2} fitness={:.3} structure={:.2} tightness={:.2} empathy={:.2} text=\"{}\"",
                    motif_id,
                    motif.motif_kind,
                    motif.promotion_status,
                    motif.promotion_score,
                    particle_fitness,
                    motif.structured_signal,
                    motif.tightness_score,
                    self.empathy_spike,
                    particle_text.chars().take(72).collect::<String>().replace('\n', " ")
                );
            }
            self.runtime_motifs.push(motif);
        }

        let pruned = self.prune_stale_live_motifs();
        if pruned > 0 && self.stdout_debug() {
            println!(" [LIVE_MOTIF] pruned_stale_live={}", pruned);
        }
        let decayed = self.decay_restored_compact_promoted();
        if decayed > 0 && self.stdout_debug() {
            println!(" [LIVE_MOTIF] decayed_restored_promoted={}", decayed);
        }
        sort_runtime_motifs_by_priority(&mut self.runtime_motifs);
        self.refresh_runtime_motif_metadata()?;

        if let Some(last_mut) = self.sentence_history.back_mut() {
            if self.runtime_motifs.iter().any(|motif| {
                Self::is_live_runtime_motif(motif)
                    && motif.last_updated_step == particle_step
                    && motif.member_count >= 2
            }) {
                last_mut.is_attractor = true;
            }
        }

        Ok(())
    }

    pub(crate) fn compute_quantum_coherence(&self, emb: &Tensor) -> Result<f32> {
        // Simplified: Sample sub-vectors and compute interference
        let samples = 10;
        let mut coh = 0.0;
        let dim = emb.dim(0)?;
        // Deterministic Sampling: Strided
        for i in 0..samples {
            let idx = (i * (dim / samples)) % dim;
            let sub = emb.narrow(0, idx, 1)?;
            let atomic_inf = self.influence_atomic_sub(&sub)?;
            coh += sigmoid(&atomic_inf)?.sum_all()?.to_scalar::<f32>()?;
        }
        Ok(coh / samples as f32)
    }

    pub(crate) fn compute_total_quantum(&self) -> Result<f32> {
        if self.sentence_history.is_empty() {
            return Ok(0.0);
        }
        Ok(self
            .sentence_history
            .iter()
            .map(|p| p.m_quantum)
            .sum::<f32>()
            / self.sentence_history.len() as f32)
    }

    pub(crate) fn compute_geometric_score(&self, emb: &Tensor) -> Result<f32> {
        if let Some(gdl) = &self.geometric_dl {
            let mesh = self.embedding_to_mesh(emb)?;
            let perf = gdl.process_mesh(&mesh)?.mean_all()?.to_scalar::<f32>()?;
            Ok(perf)
        } else {
            Ok(0.5)
        }
    }

    pub(crate) fn compute_total_geometric(&self) -> Result<f32> {
        if self.sentence_history.is_empty() {
            return Ok(0.0);
        }
        Ok(self
            .sentence_history
            .iter()
            .map(|p| p.m_geometric)
            .sum::<f32>()
            / self.sentence_history.len() as f32)
    }

    pub(crate) fn adjust_pinn_with_lpm(&self, pinn: &Tensor) -> Result<Tensor> {
        if let Some(lpm) = &self.lpm_collaborator {
            lpm.adjust_loss(pinn)
        } else {
            Ok(pinn.clone())
        }
    }

    pub(crate) fn process_photon_subsamples(&self, _state: &Tensor, _n: usize) -> Result<()> {
        Ok(())
    }

    pub(crate) fn influence_atomic_sub(&self, sub: &Tensor) -> Result<Tensor> {
        if let Some(deepmd) = &self.deepmd_kit {
            deepmd.influence(sub)
        } else {
            Ok(sub.clone())
        }
    }

    pub(crate) fn embedding_to_mesh(&self, emb: &Tensor) -> Result<Tensor> {
        let dim = emb.dim(0)?;
        let d2 = dim / 2;
        Ok(emb.narrow(0, 0, d2 * 2)?.reshape((d2, 2))?)
    }

    pub(crate) fn adjust_geometric(&self, emb: &Tensor) -> Result<Tensor> {
        if let Some(gdl) = &self.geometric_dl {
            gdl.adjust(emb)
        } else {
            Ok(emb.clone())
        }
    }

    #[allow(dead_code)]
    pub(crate) fn quantum_epsilon(&self, layer: usize) -> Result<f64> {
        Ok(1e-10 + (layer as f64 * 1e-12))
    }

    #[allow(dead_code)]
    pub(crate) fn boson_bunch(&self, prs: &Tensor) -> Result<Tensor> {
        let soft = softmax(prs, D::Minus1)?;
        let bunch = soft.powf(2.0)?;
        let sum = bunch.sum_all()?;
        let sum_scalar = sum.to_scalar::<f32>()?;
        if sum_scalar == 0.0 {
            return Ok(bunch);
        }
        let ss_t = Tensor::new(sum_scalar, bunch.device())?;
        Ok(bunch.broadcast_div(&ss_t)?)
    }

    pub(crate) fn generate_sub_particles(&self, count: usize) -> Result<Vec<SubParticle>> {
        (0..count.min(100))
            .map(|_| SubParticle::new(self.hidden_dim))
            .collect()
    }

    pub(crate) fn print_mind_state(
        &self,
        step: usize,
        current_hidden: &Tensor,
        top_k: usize,
    ) -> Result<()> {
        println!("\n=== MIND STATE @ STEP {} ===", step);
        let device = Device::Cpu;

        let last_hidden = if current_hidden.rank() == 3 {
            let seq_len = current_hidden.dim(1)?;
            current_hidden.i((.., seq_len - 1, ..))?
        } else {
            current_hidden.clone()
        };
        let hidden_cpu = last_hidden.to_device(&device)?.to_dtype(DType::F32)?;

        let hidden_norm = hidden_cpu.broadcast_div(&hidden_cpu.sqr()?.sum_all()?.sqrt()?)?;

        let charge_cpu = self
            .charge_tensor
            .to_device(&device)?
            .to_dtype(DType::F32)?;

        let hidden_proj = if self.hidden_dim != self.emb_dim {
            if let Some(proj) = &self.proj_matrix {
                hidden_norm.matmul(proj)?
            } else {
                let proj = Tensor::randn(0.0f32, 0.02, (self.hidden_dim, self.emb_dim), &device)?;
                hidden_norm.matmul(&proj)?
            }
        } else {
            hidden_norm
        };

        let scores = hidden_proj.matmul(&charge_cpu.t()?)?.squeeze(0)?;

        let scores_vec: Vec<f32> = scores.to_vec1()?;
        let mut indexed: Vec<(usize, f32)> = scores_vec
            .iter()
            .enumerate()
            .filter(|&(_, &s)| !s.is_nan())
            .map(|(i, &s)| (i, s))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Handle all-NaN case gracefully
        if indexed.is_empty() {
            println!("SOUL ORBITING: (No valid scores - all NaN)");
            println!("=================================\n");
            return Ok(());
        }

        println!("SOUL ORBITING:");
        for (rank, (idx, score)) in indexed.iter().take(top_k).enumerate() {
            if *idx < self.particle_words.len() {
                println!(
                    "  {}. {} ({:.3})",
                    rank + 1,
                    self.particle_words[*idx],
                    score
                );
            }
        }
        println!("=================================\n");
        Ok(())
    }

    pub(crate) fn get_positions(&self) -> Result<Tensor> {
        if self.sentence_history.is_empty() {
            let dev = self.charge_tensor.device();
            return Ok(Tensor::zeros((1, self.emb_dim), DType::F32, dev)?);
        }
        let tensors: Vec<Tensor> = self
            .sentence_history
            .iter()
            .filter(|p| p.fitness > 0.1 && p.m_quantum > 0.4 && p.m_geometric > 0.6)
            .map(|p| p.position.clone())
            .collect();
        if tensors.is_empty() {
            let dev = self.charge_tensor.device();
            return Ok(Tensor::zeros((1, self.emb_dim), DType::F32, dev)?);
        }
        Ok(Tensor::cat(&tensors, 0)?)
    }
}
