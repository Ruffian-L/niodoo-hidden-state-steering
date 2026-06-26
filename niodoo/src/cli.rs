//! CLI argument struct (Args) and all clap value_enum companions.
//! Extracted from main.rs as part of the comprehensive refactor
//! (pre-refactor-main-split-20260508 backup).

use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use crate::runtime::activation::{GRAVITY_WELL, ORBIT_SPEED};
use crate::runtime::finalization::LockStopPolicy;
use crate::runtime::secret_sauce_codec::SecretSauceInputVersion;
use crate::runtime::telemetry::TelemetryProfile;
use crate::ModelArchetype;

// =============================================================================
// CLI ARGUMENTS
// =============================================================================
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum ModelArchArg {
    Auto,
    Llama,
    Qwen35,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum ChatTemplateArg {
    Auto,
    Llama3,
    Qwen35,
    Raw,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum QwenThinkingMode {
    On,
    Off,
    Closed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum RuntimeSpeedProfile {
    Default,
    EvalFast,
}

impl RuntimeSpeedProfile {
    pub(crate) fn is_eval_fast(self) -> bool {
        matches!(self, Self::EvalFast)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum StdoutProfile {
    Debug,
    Telemetry,
    Quiet,
    Chat,
}

impl StdoutProfile {
    pub(crate) fn debug_enabled(self) -> bool {
        matches!(self, Self::Debug)
    }

    pub(crate) fn telemetry_enabled(self) -> bool {
        matches!(self, Self::Debug | Self::Telemetry)
    }

    pub(crate) fn chat_enabled(self) -> bool {
        matches!(self, Self::Chat)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum MistakeReflexMode {
    Off,
    Shadow,
    Influence,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum MistakeReflexActionMode {
    TextHint,
    TextHintHiddenPacket,
    SummaryHint,
    EvidenceGate,
    HiddenControl,
}

impl MistakeReflexActionMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::TextHint => "text-hint",
            Self::TextHintHiddenPacket => "text-hint-hidden-packet",
            Self::SummaryHint => "summary-hint",
            Self::EvidenceGate => "evidence-gate",
            Self::HiddenControl => "hidden-control",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum BridgeForceLayerPolicy {
    All,
    Single,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum BridgeForceSelection {
    All,
    Routed,
}

impl BridgeForceSelection {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Routed => "routed",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum BridgeForceTrajectorySchedule {
    Off,
    NearestStep,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum BridgeForceRoleFilter {
    Any,
    Structured,
    NonNeutral,
}

impl BridgeForceRoleFilter {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::Structured => "structured",
            Self::NonNeutral => "non-neutral",
        }
    }

    pub(crate) fn accepts(self, role: &str) -> bool {
        match self {
            Self::Any => true,
            Self::Structured => role == "structured" || role == "structured_candidate",
            Self::NonNeutral => role != "neutral",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum SecretSauceCapturePolicy {
    PerToken,
    Final,
    Off,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum CorrectionPacketAuthorityMode {
    Off,
    Shadow,
    Enforce,
}

impl CorrectionPacketAuthorityMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Shadow => "shadow",
            Self::Enforce => "enforce",
        }
    }
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub(crate) struct Args {
    #[arg(long, required = true)]
    pub(crate) model_path: String,

    /// Native GGUF architecture selector. Qwen35 uses the native hybrid runtime.
    #[arg(long, value_enum, default_value_t = ModelArchArg::Auto)]
    pub(crate) model_arch: ModelArchArg,

    /// Explicit tokenizer path, useful when a GGUF lives outside its HF tokenizer snapshot.
    #[arg(long)]
    pub(crate) tokenizer_path: Option<PathBuf>,

    /// Chat template used for prompt construction.
    #[arg(long, value_enum, default_value_t = ChatTemplateArg::Auto)]
    pub(crate) chat_template: ChatTemplateArg,

    /// Qwen thinking prelude mode for Qwen chat formatting. Default keeps smoke prompts concise.
    #[arg(long, value_enum, default_value_t = QwenThinkingMode::Off)]
    pub(crate) qwen_thinking: QwenThinkingMode,

    /// Requested context length. For Llama-3.1 70B use 131072 for full 128k. Wired to naked_llama rope tables + KV planning.
    #[arg(long, default_value_t = 131072)]
    pub(crate) context_length: usize,

    /// Load and report GGUF metadata, then exit before universe/bootstrap/generation.
    #[arg(long, default_value_t = false)]
    pub(crate) metadata_only: bool,

    /// Format, encode, and decode the selected chat prompt, then exit before generation.
    #[arg(long, default_value_t = false)]
    pub(crate) tokenizer_smoke: bool,

    /// Optional external universe tensor. Leave blank to synthesize from live model token embeddings.
    #[arg(long, default_value = "")]
    pub(crate) particles_path: String,

    #[arg(long, default_value = "7b")]
    pub(crate) model_size: String,

    /// Automatically scale steering parameters from model size using the square-root Lumina rule.
    /// GOLDEN CONFIG: Set to false to use fixed golden parameters (σ=0.15, θ=2.0, β=100)
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    pub(crate) model_auto_scale: bool,

    /// Override sigma (OU noise) - the "jiggle" parameter
    /// GOLDEN: 0.15 (optimal for 3B models)
    #[arg(long, default_value_t = 0.15)]
    pub(crate) sigma_override: f64,

    /// Override theta (physics blend/mean reversion)
    /// GOLDEN: 2.0 (optimal for 3B models)
    #[arg(long, default_value_t = 2.0)]
    pub(crate) theta_override: f32,

    /// Let accepted visible control requests open a short force gate even when
    /// raw pressure is below the background activation threshold.
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    pub(crate) visible_request_gate: bool,

    /// Model behavior class used by the auto-scaler.
    #[arg(long, value_enum, default_value_t = ModelArchetype::Auto)]
    pub(crate) model_archetype: ModelArchetype,

    #[arg(long, default_value = "Make me a sandwich.")]
    pub(crate) prompt: String,

    #[arg(long, default_value = "fixed-001")]
    pub(crate) req_id: String,

    #[arg(long, default_value_t = false)]
    pub(crate) bridge_off: bool,

    /// Require a working CUDA device; if true, abort instead of falling back to
    /// CPU. Defaults to false so the runtime is portable (runs on any machine,
    /// CPU-only included). The canonical GPU reproduction (reproduce.sh) passes
    /// `--require-cuda true` so a missing/misconfigured GPU fails loudly there.
    #[arg(long, action = clap::ArgAction::Set, default_value_t = false)]
    pub(crate) require_cuda: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) bridge_influence_smoke: bool,

    #[arg(long, default_value_t = 0.03)]
    pub(crate) bridge_influence_smoke_clamp: f32,

    #[arg(long, default_value_t = false)]
    pub(crate) bridge_influence_selective: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) bridge_gate34_latch: bool,

    #[arg(long, default_value_t = 8)]
    pub(crate) gate34_warmup_steps: u32,

    #[arg(long, default_value_t = 96)]
    pub(crate) gate34_hold_steps: u32,

    #[arg(long, default_value_t = 0.00005)]
    pub(crate) gate34_release_margin_floor: f32,

    #[arg(long, default_value_t = 6)]
    pub(crate) gate34_release_patience: u32,

    #[arg(long, default_value_t = 1.35)]
    pub(crate) gate34_release_distance_mult: f32,

    #[arg(long, default_value_t = 3)]
    pub(crate) gate34_acquire_top_k: usize,

    #[arg(long, default_value_t = 1.0)]
    pub(crate) bridge_prompt_weight: f32,

    #[arg(long, default_value = "motifs")]
    pub(crate) gate34_target_source: String,

    #[arg(long, default_value_t = 0.5)]
    pub(crate) gate34_motif_routing_safety_floor: f32,

    #[arg(long, default_value_t = 0.0006)]
    pub(crate) bridge_margin_threshold: f32,

    #[arg(long, default_value_t = 3)]
    pub(crate) bridge_stability_k: u32,

    #[arg(long, default_value_t = 0)]
    pub(crate) bridge_cooldown_after_switch: u32,

    #[arg(long, default_value_t = false)]
    pub(crate) bridge_scale_by_margin: bool,

    /// Path to codebook_256.json for VQ codec specialist integration.
    #[arg(long)]
    pub(crate) codebook_path: Option<PathBuf>,

    /// Path to specialist_phase2_params.json for rule-based specialist integration.
    #[arg(long)]
    pub(crate) specialist_params_path: Option<PathBuf>,

    /// When true and both rave codec and rule-based specialist are loaded, the specialist's
    /// 2D correction delta is converted into a real 4096D hidden-state force via the trained
    /// codec (z = encode(probe); z' = z + [delta_x, delta_y, 0...]; force = decode(z') - decode(z))
    /// and added to apply_forces. Default false preserves observational-only behavior.
    /// Closes the hollow-flag class where intervention_applied=true had no effect on the tensor.
    #[arg(long, default_value_t = false)]
    pub(crate) specialist_correction_apply: bool,

    /// Maximum L2 norm of the codec-mediated specialist force in 4096D space. Forces above
    /// this magnitude are clamped to it. Used only when --specialist-correction-apply is true.
    #[arg(long, default_value_t = 0.03)]
    pub(crate) specialist_correction_clamp: f32,

    /// Path to a JSONL of correction packets indexed by codebook bucket (`vq_code`).
    /// When present and the rave codec + codebook are loaded, every step the runtime
    /// encodes the probe, looks up packets matching the current vq_code, and for each
    /// firing packet adds a codec-mediated 4096D pull-toward-target force to probe_force.
    /// Independent of --specialist-correction-apply (which is the rule-based phase2 path).
    #[arg(long)]
    pub(crate) correction_packets_path: Option<PathBuf>,

    /// Per-fire L2 clamp for codec-mediated correction-packet forces in 4096D space.
    /// Each firing packet's individual contribution is clamped to this magnitude; the
    /// total per-step force is the sum of clamped per-packet contributions.
    #[arg(long, default_value_t = 0.03)]
    pub(crate) correction_packet_clamp: f32,

    /// Optional factual payload blend for correction-packet force. `0.0` preserves
    /// legacy target-only pull. When >0 and a packet carries `payload_z_64d`, the
    /// runtime orthogonalizes that payload against the target-pull direction and
    /// mixes it into the same per-packet force budget. This is the opt-in Anti-Splat
    /// payload path for packets that have a bound factual vector.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) correction_packet_payload_blend: f32,

    /// When set, the runtime appends one CorrectionPacket JSONL record at end-of-run with
    /// `target_z_64d` = bucket-mean of the final-step hidden probe and
    /// `vq_code = codebook.encode(target)`. The output file is opened in append mode, so
    /// repeated runs accumulate packets. Pair with `--correction-packets-path <same-file>`
    /// on a subsequent run to load and fire them. The "preserve correction across fresh
    /// process boundaries" North Star primitive.
    #[arg(long)]
    pub(crate) correction_packets_out: Option<PathBuf>,

    /// When true, packets minted via `--correction-packets-out` store the 64D target as a
    /// secret_sauce V3 Unicode string (`target_z_unicode_v3`) and omit the numeric
    /// `target_z_64d` array. This forces the replay process to cross the Unicode codec
    /// boundary on load (decoder reconstructs 64D target before the RAVE-codec → 4096D
    /// force chain). Default false preserves the legacy numeric JSONL format.
    #[arg(long, default_value_t = false)]
    pub(crate) correction_packet_out_unicode_v3: bool,

    /// `pull_strength` field stamped on each minted CorrectionPacket. Used at runtime by the
    /// receiver: `delta_64 = pull_strength * direction(target - probe)`.
    #[arg(long, default_value_t = 0.1)]
    pub(crate) correction_packet_out_pull_strength: f32,

    /// `distance_threshold` field stamped on each minted CorrectionPacket. The receiver does
    /// not fire the packet when probe-to-target distance is below this value.
    #[arg(long, default_value_t = 0.05)]
    pub(crate) correction_packet_out_distance_threshold: f32,

    /// `pull_strength` written on packets minted from `[REQUEST: LOCK]` ("earned answer")
    /// emissions. Defaults higher than `--correction-packet-out-pull-strength` (which
    /// applies to REMEMBER and end-of-run captures) because LOCK signals a final answer
    /// the model is committing to — the goal is to anchor future generation back toward
    /// that earned probe state. The "preserve earned answers before drift" North Star
    /// primitive.
    #[arg(long, default_value_t = 0.3)]
    pub(crate) correction_packet_lock_pull_strength: f32,

    /// Multiplier applied to `correction_packet_lock_pull_strength` when the user
    /// contradicts a prior LOCK in this turn — i.e. the agency-hands learning-event
    /// signal fires. The corrected answer's earned packet mints with
    /// `pull = lock_pull * multiplier` so the new basin dominates the contradicted
    /// prior earned basin. Defaults to 2.0 (corrected packets pull twice as hard).
    /// Surfaces the user's "I changed my mind" signal as a stronger steering reflex.
    #[arg(long, default_value_t = 2.0)]
    pub(crate) correction_packet_lock_contradiction_multiplier: f32,

    /// On LOCK contradiction, also explicitly INVALIDATE prior correction packets
    /// that were minted from the contradicted payload (matched via their
    /// `lh_<lhash>` segment in `packet_id`). Invalidated packets stay in the store
    /// (and the JSONL) but `forward_with_pull` returns None for them, so they no
    /// longer fire. Pairs with the multiplier amplification: the corrected basin
    /// pulls harder AND the contradicted basin gets switched off. Defaults to true.
    #[arg(long, default_value_t = true)]
    pub(crate) correction_packet_invalidate_on_contradiction: bool,

    /// On every LOCK emission (contradiction or not), revalidate any previously
    /// invalidated packets matching this LOCK's exact payload hash. Inverse of
    /// `--correction-packet-invalidate-on-contradiction`: when the user re-affirms
    /// a payload they had previously contradicted, the prior invalidation rolls
    /// back so the original earned packet fires again. Only the exact `lh_<hash>`
    /// form is revalidated (not the semantic-key form, to avoid ping-pong on
    /// oscillation between values for the same key). Defaults to true.
    #[arg(long, default_value_t = true)]
    pub(crate) correction_packet_revalidate_on_affirmation: bool,

    /// Cap on the per-payload-key contradiction count used for adaptive
    /// multiplier scaling. Each contradiction event for the same key
    /// increments a counter; the effective multiplier becomes
    /// `lock_contradiction_multiplier × min(count, cap)`. With base 2.0 and
    /// cap 5, the corrected packet's pull tops out at `lock_pull × 10`.
    /// Surfaces escalating user frustration ("no, REALLY, X") as a
    /// progressively stronger steering reflex without unbounded growth.
    #[arg(long, default_value_t = 5)]
    pub(crate) correction_packet_adaptive_contradiction_cap: u64,

    /// Path to a JSONL file persisting per-payload-key contradiction counts
    /// across process restarts. Loaded at startup (if file exists), atomically
    /// rewritten at end-of-run with current counts. Without this, the
    /// escalation in §10ac resets to zero each restart; with it, the user's
    /// correction history accumulates across sessions just like the packet
    /// fire counters do (§10t).
    #[arg(long)]
    pub(crate) correction_contradiction_counts_path: Option<PathBuf>,

    /// Threshold on `mistake_reflex_retry_count` that adds a SECOND relapse trigger
    /// alongside the existing vq_encode_error trigger (§10s). When the count
    /// reaches this value at generation start, firing packets get unfold-boosted.
    /// `0` disables the trigger entirely (legacy single-source behavior). Default
    /// `1` — any retry counts as relapse, since the model just had to be re-prompted
    /// for the same answer.
    #[arg(long, default_value_t = 1)]
    pub(crate) correction_packet_unfold_on_retry_count: usize,

    /// Per-source unfold factor for the retry-relapse trigger (§10ae). When
    /// `> 1.0`, retry-relapse uses this factor instead of the engine global
    /// `--correction-packet-unfold-factor`. When 0 (default) or `≤ unfold_factor`,
    /// retry-relapse uses the engine global. Allows retry-relapse to boost more
    /// strongly than OOD-relapse since "model just had to retry" is more direct
    /// evidence of failure than "probe drifted out of codebook training distribution".
    /// The applied factor on a step is max(encode_error_factor, retry_factor).
    #[arg(long, default_value_t = 0.0)]
    pub(crate) correction_packet_unfold_retry_factor: f32,

    /// Eviction floor for long-decayed packets (§10ai). At end-of-run, before
    /// state-out persistence, packets whose `effective_pull / pull_strength`
    /// falls below this ratio are pruned from the store. Earned packets
    /// (decay_rate=Some(1.0)) never qualify since their ratio stays at 1.0
    /// regardless of fire_count. Default `0.0` disables eviction entirely.
    /// Reasonable values are 0.01–0.1 — keeps the JSONL bounded over very long
    /// sessions without affecting earned-answer preservation.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) correction_packet_eviction_floor: f32,

    /// Per-fire decay rate for correction packets: `effective_pull = pull_strength *
    /// decay_rate.powi(fire_count)`. Values in `(0.0, 1.0)` enable decay (closer to 0 =
    /// faster decay). Default `1.0` disables decay (legacy behavior). The "decay scaffolding
    /// as competence improves" North Star primitive — packets that fire often gradually
    /// weaken as the model "learns" their bucket.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) correction_packet_decay_rate: f32,

    /// Relapse trigger: when `vq_encode_error > threshold` for the current step's probe,
    /// firing packets in this bucket get their effective pull multiplied by
    /// `--correction-packet-unfold-factor`. The "unfold stronger guidance when relapse
    /// returns" North Star primitive. A high quantization error means the probe has
    /// drifted out of the codebook's training distribution — i.e. the model is back in
    /// territory it once knew (we have packets here) but is now off-distribution
    /// (relapse). Default `0.0` disables the unfold path (raw decayed pull is used).
    #[arg(long, default_value_t = 0.0)]
    pub(crate) correction_packet_unfold_encode_error_threshold: f32,

    /// Multiplicative boost applied to each firing packet's effective pull when the
    /// relapse trigger fires (see `--correction-packet-unfold-encode-error-threshold`).
    /// Values `> 1.0` re-strengthen decayed packets; `1.0` leaves the decayed pull
    /// unchanged. Pairs with the decay path so a packet that has decayed under
    /// `--correction-packet-decay-rate` can be temporarily un-decayed when the model
    /// hits trouble in its bucket again.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) correction_packet_unfold_factor: f32,

    /// Competence-aware suppression factor (North Star bullet 7: "preserve earned
    /// answers before drift"). When the previous step's normalized sampling entropy
    /// is BELOW `correction_packet_competence_entropy_threshold` (model is confident
    /// about the next token, i.e., trajectory is competent), each firing packet's
    /// effective pull is multiplied by this factor. Values < 1.0 reduce pull on
    /// confident steps to avoid overriding correct trajectories. Default 1.0 = no
    /// suppression (legacy behavior). 0.0 = full suppression on competent steps.
    /// Backed by §10aw evidence: global decay/clamp tuning cannot unify per-seed
    /// lift; per-step competence-aware modulation is the unblock.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) correction_packet_competence_suppress_factor: f32,

    /// Entropy threshold below which the previous-step sampling is considered
    /// "competent" and `correction_packet_competence_suppress_factor` engages.
    /// `entropy_norm` is Shannon entropy / ln(vocab_size) ∈ [0, 1]. Typical
    /// confident-step values are 0.05-0.25; uncertain values are 0.4+. Default
    /// 0.0 = never trigger (suppression is disabled). Increase to 0.25-0.35 to
    /// engage suppression on confident-token steps.
    ///
    /// NOTE: at temp=0.0, entropy is consistently low (sharply-peaked distribution
    /// even at T=1.0 for many tasks), so this trigger fires on every step. The
    /// density-based trigger below is more discriminating.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) correction_packet_competence_entropy_threshold: f32,

    /// Density threshold for the competence-aware suppression trigger. When the
    /// number of firing packets at the current step is at-or-above this value,
    /// the probe is in a high-density "correct geometry" region and the
    /// `correction_packet_competence_suppress_factor` engages. Default 0
    /// disables the density trigger. Typical values: 50-150 (per §10av's
    /// "competence preservation threshold around 150 force-weighted density").
    /// OR-combined with the entropy trigger: either can engage suppression.
    #[arg(long, default_value_t = 0)]
    pub(crate) correction_packet_competence_density_threshold: usize,

    /// Distance threshold for the per-trajectory competence trigger (§10ay).
    /// When the minimum 64D distance from the probe to any matched-bucket
    /// packet's target_z is BELOW this value, trajectory is "near correct
    /// geometry" and `correction_packet_competence_suppress_factor` engages.
    /// Finer-grained than density: distance grades closeness, while density
    /// is binary on bucket population. Default 0.0 disables. Typical values:
    /// 0.05-0.20 in normalized 64D space. OR-combined with entropy/density.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) correction_packet_competence_distance_threshold: f32,

    /// Combine mode for the competence triggers (§10az). When `or` (default,
    /// legacy behavior), ANY enabled trigger engages suppression. When `and`,
    /// ALL enabled triggers must agree to engage suppression — more
    /// restrictive, useful when distance and density signals fight on certain
    /// trajectories. Disabled triggers are excluded from the AND check.
    /// When `continuous`, suppression factor varies smoothly with the
    /// max competence strength across enabled triggers — no binary threshold,
    /// each step gets a per-trajectory-adaptive factor (§10ba).
    #[arg(long, default_value_t = String::from("or"))]
    pub(crate) correction_packet_competence_combine_mode: String,

    /// Total per-step force budget across ALL firing packets (§10bb). The
    /// existing `--correction-packet-clamp` is per-packet; total per-step
    /// force scales linearly with the number of firings. With store sizes
    /// > ~5-9 packets, cumulative force can override correct OFF trajectories
    /// even with competence-aware suppression (iter-46 evidence).
    /// This flag bounds the L2 norm of the cumulative new_probe_force after
    /// all packets have been added. Default 0.0 disables (legacy linear-scale
    /// behavior). Typical values: 5-10× per-packet clamp (e.g. 0.025-0.05 with
    /// per-packet clamp 0.005). Bounds total force regardless of store size,
    /// fixing the "more scar tissue = more interference" failure mode.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) correction_packet_total_clamp: f32,

    /// Direction-aware firing: maximum 64D distance from probe to packet's
    /// target_z BELOW which the packet is still considered "aligned" enough
    /// to fire (§10bc). Per iter-47 diagnosis, holdout probes match packets
    /// by bucket but the target_z direction is wrong for the holdout's
    /// correct answer. This upper-bound distance gate filters out packets
    /// whose target is too far from the current probe, preventing
    /// wrong-direction pull on cross-prompt holdouts.
    /// Default 0.0 = disabled (fire all bucket-matched packets, legacy).
    /// Typical values: 0.30-0.50. Combines with the per-packet
    /// distance_threshold (lower bound — probe must be far enough to need
    /// correction) to define a "ring of relevance" around each target.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) correction_packet_fire_max_distance: f32,

    /// Per-trajectory top-K filtering (§10be): when > 0, only fire the
    /// K packets whose target_z is CLOSEST to the current probe out of
    /// all bucket-matched + distance-passed packets. Direction-aware
    /// retrieval that scales gracefully with packet store size — at
    /// 15-teach with K=5, only the 5 most-aligned packets fire,
    /// recovering the alignment quality of 9-teach without losing
    /// the broader scar tissue. Default 0 = disabled (fire all matched,
    /// legacy). Addresses iter-57b/58/59's U-shape: direction-mediated
    /// interference from many packets even when force is bounded.
    #[arg(long, default_value_t = 0)]
    pub(crate) correction_packet_fire_top_k: usize,

    /// Mint-time per-bucket cap (§10bf). When > 0, mint operations skip
    /// writing new packets if the bucket already holds K packets. Forces
    /// store-level bucket diversity. Iter-62 identified bucket concentration
    /// as the U-shape root cause: 15-teach naively converges to 2 buckets
    /// while 9-teach has 3, which is why 9-teach lifts and 15-teach doesn't.
    /// This flag prevents that concentration at mint time. Default 0
    /// disables. Typical: 1-3.
    #[arg(long, default_value_t = 0)]
    pub(crate) correction_packet_mint_bucket_cap: usize,

    /// Per-trajectory step-window fire-gate (§10bh). Format "start:end".
    /// When set, packets are only allowed to fire when current_step is in
    /// [start, end] inclusive. Steps outside the window emit zero force.
    /// Iter-228 found cross-word lift mechanism is post-enumeration recovery
    /// shaping: saturations during enumeration phase + drift firings during
    /// recovery phase. This flag lets you isolate which phase actually
    /// matters by gating firing to specific step ranges. Default empty
    /// disables (fires at all steps).
    #[arg(long, default_value_t = String::new())]
    pub(crate) correction_packet_fire_step_window: String,

    /// §10bd Track 2 v11: per-trajectory adaptive gate routing. Addresses
    /// §10az structural unification gap (no static combine_mode unifies
    /// seeds 143/211/377). When enabled, the engine accumulates the
    /// per-step `last_correction_packet_fire_count` over the first
    /// `correction_packet_trajectory_classify_step` steps of each turn
    /// (default 10), then once-classifies the trajectory: if mean fire
    /// count is ABOVE
    /// `correction_packet_trajectory_fire_count_threshold` (default 55)
    /// the trajectory is "competent" (in a high-density basin) and the
    /// gate switches to AND-combined density-only suppression (§10az v9,
    /// best for s377-like). Otherwise "drifting" — uses the user-
    /// configured combine mode (default OR, §10az v8 best for s143/211-
    /// like). Threshold tuned from §10az v8 telemetry: s377 first-10
    /// mean ≈ 58.5, s143/211 first-10 mean ≈ 51.4 — 55 cleanly separates.
    /// Entropy-based design (v11.0) was degenerate at temp=0 (entropy ≡ 0).
    /// Operationalizes North Star bullet 5 ("recognize related future
    /// situations") as a runtime classifier.
    #[arg(long, default_value_t = false)]
    pub(crate) correction_packet_trajectory_routing: bool,

    /// Step (per-turn, not session-monotonic) at which the trajectory
    /// classifier runs. Below this step the user-configured combine mode
    /// is used. Only relevant when
    /// `--correction-packet-trajectory-routing` is set.
    #[arg(long, default_value_t = 10)]
    pub(crate) correction_packet_trajectory_classify_step: usize,

    /// Mean-fire-count threshold for the trajectory classifier. Mean of
    /// `last_correction_packet_fire_count` across steps [0, classify_step)
    /// is compared to this; ABOVE = "competent" (dense basin → AND mode),
    /// BELOW = "drifting" (sparse → user-configured mode). Only relevant
    /// when `--correction-packet-trajectory-routing` is set. Default 55
    /// tuned from §10az v8 telemetry.
    #[arg(long, default_value_t = 55.0)]
    pub(crate) correction_packet_trajectory_fire_count_threshold: f32,

    /// §10cq trajectory-routed top-K for "competent" classification.
    /// When the §10bd v11.5 trajectory router is enabled and the
    /// classifier emits "competent", `fire_top_k` is overridden by
    /// this value (UNLESS the §10ck prompt-K map already overrode it).
    /// Default 0 means no override (legacy §10cp behavior — router
    /// only flips combine_mode). Set to the K derived for the
    /// "competent" trajectory class to make the router actually
    /// change firing density per class.
    #[arg(long, default_value_t = 0)]
    pub(crate) correction_packet_trajectory_top_k_competent: usize,

    /// §10cq trajectory-routed top-K for "drifting" classification.
    /// Mirror of `--correction-packet-trajectory-top-k-competent`.
    /// Default 0 = no override.
    #[arg(long, default_value_t = 0)]
    pub(crate) correction_packet_trajectory_top_k_drifting: usize,

    /// §10bf prompt-level codec activation gate (the *coarse* layer
    /// complementing §10bd v11's per-trajectory *fine* layer). When this
    /// flag is non-empty, the runtime activates correction-packet
    /// AND bridge-influence forces only when the current prompt
    /// contains AT LEAST ONE of the listed substrings. Comma-separated.
    /// Default empty disables the gate (legacy: always active).
    /// Operationalizes the §10bc family-routing simulation as a runtime
    /// classifier — bullet 5 ("recognize related future situations") at
    /// the prompt-text layer, before any token has been decoded.
    /// Example: `--codec-active-prompt-substrings reverse,what is the,explain`
    /// activates the codec/bridge path on presentation/recall families
    /// and suppresses on counting/arithmetic families.
    #[arg(long, default_value_t = String::new())]
    pub(crate) codec_active_prompt_substrings: String,

    /// §10ck per-prompt-substring → top-K override map. Format:
    /// `<substring>:<K>,<substring>:<K>,...`. Substring match is
    /// case-insensitive against the user prompt. First match wins.
    /// When set and the current prompt matches a substring, the
    /// corresponding K overrides `--correction-packet-fire-top-k`
    /// for that turn. Empty = no override (legacy behavior).
    /// Operationalizes the §10cj static prompt→K routing table:
    /// `--correction-packet-prompt-top-k-map "letters that match the letter s:3,letter s:5,letter i:0,letter p:0,letter m:3"`
    /// achieves 14/15 = 0.933 on Mississippi mini-AB cross-seed.
    #[arg(long, default_value_t = String::new())]
    pub(crate) correction_packet_prompt_top_k_map: String,

    /// §10cn task-neutrality gate. When the §10ck prompt-K map is
    /// non-empty AND no substring matched the current prompt, skip
    /// correction-packet firing entirely for that turn (treat as
    /// out-of-distribution for the trained reflex). Default false
    /// (legacy: unmatched prompts use the global fire_top_k and
    /// packets fire normally — which can hurt unrelated tasks).
    /// Validated on §10.m presentation/recall families: v2_090
    /// (Mississippi-derived) hurts -0.17 without the gate.
    #[arg(long, default_value_t = false)]
    pub(crate) correction_packet_suppress_when_no_prompt_match: bool,

    /// Prompt-level source-target filter for correction packets. Format:
    /// `<substring>:<target_id>,<substring>:<target_id>,...`. Substring
    /// match is case-insensitive against the user prompt. First match wins.
    /// When set and the current prompt matches a substring, only packets
    /// whose `source_label` contains `target_id=<target_id>` are allowed to
    /// fire for that turn. Empty = no filter (legacy behavior).
    /// This is the target/source-aware selector seeded by the repaired
    /// Tennessee/Hippopotamus holdout audit: exact source packets can be
    /// preferred without broadly increasing top-K.
    #[arg(long, default_value_t = String::new())]
    pub(crate) correction_packet_prompt_source_target_map: String,

    /// §10bs ghost-force-aware packet suppression. When enabled, skip
    /// correction-packet force application on any step where the prior
    /// step's `applied_ghost_mag` exceeded
    /// `correction_packet_suppress_when_bridge_force_above`. Validated by
    /// §10bs empirical sweep: at default packet clamp 0.005 with bridge
    /// active, packets cause probe relocation that drops a +0.20 lift
    /// to 0.00 on Mississippi. Disabling packets when bridge is firing
    /// recovers the bridge-alone lift. Default `0.0` disables this
    /// gate (legacy behavior).
    #[arg(long, default_value_t = 0.0)]
    pub(crate) correction_packet_suppress_when_bridge_force_above: f32,

    /// §10bx post-bridge re-encoding mode. When TRUE and bridge is
    /// active (prev_step_max_ghost_mag > 0), block packet firing
    /// ONLY at layer 0 of each decode step (where the probe state
    /// hasn't yet been shifted by this step's bridge force, so packet
    /// matches would target pre-bridge buckets). At layer 1+, allow
    /// packets to fire — at that point the probe HAS been bridge-
    /// shifted by layer 0, so packet matches target post-bridge
    /// buckets. This is the algorithm doc's "probe-tracking bridge
    /// selection" prescription mapped to packets: packets correct
    /// on the bridge-influenced probe rather than the pre-bridge probe.
    /// Default false = legacy. When set to true, takes precedence over
    /// `--correction-packet-suppress-when-bridge-force-above` on layer 0
    /// only.
    #[arg(long, default_value_t = false)]
    pub(crate) correction_packet_post_bridge_mode: bool,

    /// Mint-readiness lock gate (DEEP_DIVE_ROADMAP P1-B). When the active
    /// routing-cache motif's `readiness_score` exceeds this threshold, skip
    /// correction-packet firing for the rest of the step — the answer is
    /// "earned" and further correction would overwrite it. Deep dive evidence:
    /// baseline_60 reached 80% mint-ready with zero post-readiness corrections,
    /// 2+ corrections degraded to 20%. Recommended threshold 0.55. Default
    /// `0.0` disables the gate (legacy behavior).
    #[arg(long, default_value_t = 0.0)]
    pub(crate) correction_packet_readiness_lock_threshold: f32,

    /// DEEP_DIVE_ROADMAP P2-A — per-layer physics blend mask. Layer index
    /// at which deep-layer "whisper" protection begins. When `layer_idx >=
    /// physics_blend_deep_layer_from`, the motif_force and recovery_force
    /// returned by `compute_bridge_forces` are scaled by
    /// `physics_blend_deep_layer_multiplier` before being added to probe_force.
    /// For Llama-3.1-8B (32 layers): set 28 to protect the last 4 layers.
    /// For Qwen 3.5 (28 layers): set 24. Default `0` disables (legacy uniform
    /// blending). Deep dive: deep layers carry route-specific information
    /// that uniform aggressive blending overwrites; protecting them retains
    /// task geometry through the bridge.
    #[arg(long, default_value_t = 0)]
    pub(crate) physics_blend_deep_layer_from: usize,

    /// DEEP_DIVE_ROADMAP P2-A — per-layer multiplier applied to bridge
    /// forces (motif + recovery) when `layer_idx >=
    /// physics_blend_deep_layer_from`. Deep dive recommendation: `0.1`.
    /// Default `1.0` disables (legacy behavior). Has no effect unless
    /// `--physics-blend-deep-layer-from` is also set non-zero.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) physics_blend_deep_layer_multiplier: f32,

    /// DEEP_DIVE_ROADMAP P2-B — consensus weight formula for motif routing.
    /// When set, replaces the additive conflict/mixed penalty
    /// (`+ 0.08*conflict_ratio + 0.03*mixed_ratio`) with the softmax-shape
    /// weight `w_i = exp(persistence_score - 0.08*conflict_ratio - 0.03*mixed_ratio)`,
    /// applied as `score -= w_i` so high w_i lowers the score (preferred).
    /// Effect path: `routing_score_for_motif` inside `run_periodic_controller`,
    /// which fires only on structured prompts. Default false keeps legacy
    /// P1-A additive form.
    #[arg(long, default_value_t = false)]
    pub(crate) motif_routing_consensus_weight: bool,

    /// DEEP_DIVE_ROADMAP P2-C — autonomic physics adaptation threshold.
    /// When the rolling-window average bridge force magnitude (motif +
    /// recovery, taken from `last_motif_mag` + `last_recovery_mag` per
    /// step) exceeds this value, scale motif_force_scale and
    /// recovery_force_scale by 0.9×; when it falls below 0.5× the
    /// threshold, restore by 1.05× (capped at original value). `0.0`
    /// disables. Deep dive's `physics_blend` knob isn't a force
    /// multiplier in this codebase, so adaptation targets the actual
    /// scale knobs instead.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) autonomic_physics_force_threshold: f32,

    /// DEEP_DIVE_ROADMAP P2-C — rolling window size for autonomic
    /// adaptation (number of bridge-force samples). Default 50 matches
    /// the deep dive. Effective only when `--autonomic-physics-force-threshold`
    /// is set non-zero.
    #[arg(long, default_value_t = 50)]
    pub(crate) autonomic_physics_window: usize,

    /// Feature-gated packet steering arbitration. `disabled` preserves the
    /// existing force path. `auto` can stand packet force down to shadow/no_packet
    /// when packet authority is weak, the current route looks healthy, or candidate
    /// packet targets are too distant. Explicit modes are test/smoke surfaces.
    #[cfg(feature = "niodv4_bridge")]
    #[arg(long, default_value_t = String::from("disabled"))]
    pub(crate) correction_packet_arbitration: String,

    /// Auto arbitration healthy-route proxy. When auto arbitration is enabled and
    /// the existing competence gate has already reduced packet force below this
    /// factor, candidates stay in packet_shadow instead of applying force.
    #[cfg(feature = "niodv4_bridge")]
    #[arg(long, default_value_t = 0.999)]
    pub(crate) correction_packet_arbitration_healthy_factor_threshold: f32,

    /// Auto arbitration stale/wrong-basin proxy. When >0 and the nearest candidate
    /// packet target is farther than this 64D distance, candidates stay in
    /// packet_shadow instead of applying force.
    #[cfg(feature = "niodv4_bridge")]
    #[arg(long, default_value_t = 0.0)]
    pub(crate) correction_packet_arbitration_stale_distance_threshold: f32,

    /// Mint a live-capture packet from each turn's final probe state, not
    /// just on echoed LOCK/REMEMBER tags (§10bg). Iter-66 identified that
    /// agency-hands minting is gated by model behavior — non-counting
    /// prompts didn't produce LOCK echoes so they minted no packets,
    /// preventing bucket diversification. This flag bypasses that gate:
    /// every turn's probe-bucket-mean produces a live_capture packet.
    /// Pairs naturally with --correction-packet-mint-bucket-cap to enforce
    /// diversity even with this firehose minting. Default false (legacy).
    #[arg(long, default_value_t = false)]
    pub(crate) correction_packet_capture_every_turn: bool,

    /// Source-aware firing: only fire packets whose embedded source prompt
    /// hash matches the current prompt hash (§10bd). Each minted packet has
    /// `ph_<hash>` in its packet_id from the prompt it was minted on. When
    /// this flag is true, packets only fire on probes from the EXACT same
    /// prompt — full per-prompt isolation. Solves iter-46/47/48 cross-word
    /// interference at the cost of cross-prompt generalization (which
    /// already wasn't working anyway). Default false = legacy.
    #[arg(long, default_value_t = false)]
    pub(crate) correction_packet_fire_match_prompt_hash: bool,

    /// Packet authority gate for latent steering stabilization. `shadow`
    /// records allow/block telemetry without changing legacy force behavior.
    /// `enforce` drops weak/unknown authority packets before hidden force is
    /// applied. `off` disables this diagnostic gate.
    #[arg(long, value_enum, default_value_t = CorrectionPacketAuthorityMode::Shadow)]
    pub(crate) correction_packet_authority_mode: CorrectionPacketAuthorityMode,

    /// At end-of-run, atomically rewrite the entire correction-packet store to this path
    /// with current `fire_count` and `last_fire_step` counters preserved. Pair with
    /// `--correction-packets-path <same-file>` on a subsequent run so decay/unfold state
    /// survives process boundaries — true cross-session "scar tissue" memory. Distinct
    /// from `--correction-packets-out`, which only appends *new* live-capture packets
    /// minted from the final probe.
    #[arg(long)]
    pub(crate) correction_packets_state_out: Option<PathBuf>,

    #[arg(long)]
    pub(crate) telemetry_out: Option<PathBuf>,

    /// Read-only Active Context adapter decision JSONL. Loads and logs control metadata only.
    #[arg(long)]
    pub(crate) active_context_adapter_decisions: Option<PathBuf>,

    /// Optional read-only Active Context startup summary JSON. Requires --active-context-adapter-decisions.
    #[arg(long)]
    pub(crate) active_context_startup_summary_out: Option<PathBuf>,

    /// Add a read-only Active Context startup metadata row to route telemetry. Requires --active-context-adapter-decisions.
    #[arg(long, default_value_t = false)]
    pub(crate) active_context_startup_telemetry: bool,

    /// JSONL telemetry detail level. full preserves existing artifacts.
    #[arg(long, value_enum, default_value = "full")]
    pub(crate) telemetry_profile: TelemetryProfile,

    /// Enable the live TDA shadow monitor over per-token telemetry windows.
    #[arg(long, default_value_t = false)]
    pub(crate) tda_shadow_monitor: bool,

    /// Rolling token window for --tda-shadow-monitor. Runtime clamps this to 3..64.
    #[arg(long, default_value_t = 32)]
    pub(crate) tda_shadow_window: usize,

    /// Recompute the TDA monitor every N tokens after the window is full.
    #[arg(long, default_value_t = 8)]
    pub(crate) tda_shadow_stride: usize,

    /// Let fresh TDA breath recommendations trigger the existing focus-lock heartbeat.
    /// Default false keeps the monitor observational.
    #[arg(long, default_value_t = false)]
    pub(crate) tda_shadow_breath_apply: bool,

    /// Runtime speed profile. eval-fast keeps artifacts but disables expensive exploratory extras.
    #[arg(long, value_enum, default_value = "default")]
    pub(crate) runtime_speed_profile: RuntimeSpeedProfile,

    /// Stdout volume. Artifact JSONL still writes unless disabled elsewhere.
    #[arg(long, value_enum, default_value = "debug")]
    pub(crate) stdout_profile: StdoutProfile,

    /// Apply bridge/memory force on all physics layers or a single selected layer.
    #[arg(long, value_enum, default_value = "all")]
    pub(crate) bridge_force_layer_policy: BridgeForceLayerPolicy,

    /// Selected layer when --bridge-force-layer-policy=single.
    #[arg(long, default_value_t = 24)]
    pub(crate) bridge_force_layer: usize,

    /// Select which bridge motifs contribute force. all preserves legacy broad softmax.
    #[arg(long, value_enum, default_value = "all")]
    pub(crate) bridge_force_selection: BridgeForceSelection,

    /// Optional decode-step scheduler for hidden_trajectory bridge motifs.
    #[arg(long, value_enum, default_value = "off")]
    pub(crate) bridge_force_trajectory_schedule: BridgeForceTrajectorySchedule,

    /// Optional role filter for bridge motif force candidates.
    #[arg(long, value_enum, default_value = "any")]
    pub(crate) bridge_force_role_filter: BridgeForceRoleFilter,

    /// Minimum selected-vs-runner-up score margin required before bridge force applies.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) bridge_force_min_margin: f32,

    /// Per-token compact Unicode/state capture policy.
    #[arg(long, value_enum, default_value = "per-token")]
    pub(crate) secret_sauce_capture_policy: SecretSauceCapturePolicy,

    /// Shadow-mode specialist memory packet selector built from niodv4 4096D/64D captures.
    #[arg(long)]
    pub(crate) specialist_memory_workers_path: Option<PathBuf>,

    /// Path to a trained Rave hidden-state codec safetensors file. When present and
    /// --features=niodv4_bridge is enabled, project_bridge_vector_to_hidden uses the codec
    /// decoder instead of the bucket-expansion fallback.
    #[arg(long)]
    pub(crate) rave_codec_path: Option<PathBuf>,

    /// Specialist memory worker runtime mode. off preserves old behavior; shadow logs selection only.
    #[arg(long, value_enum, default_value = "off")]
    pub(crate) specialist_memory_workers_mode: SpecialistMemoryWorkerMode,

    /// Number of nearest worker packets to retain for shadow routing diagnostics.
    #[arg(long, default_value_t = 5)]
    pub(crate) specialist_memory_worker_top_k: usize,

    /// Max norm for explicit route-memory worker influence mode.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) specialist_memory_worker_influence_clamp: f32,

    /// Signed multiplier for route-memory worker influence smokes; default preserves pull-to-packet.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) specialist_memory_worker_influence_sign: f32,

    /// Optional JSONL path for narrow answer-slot logit diagnostics.
    #[arg(long)]
    pub(crate) answer_logit_probe_out: Option<PathBuf>,

    /// Comma-separated answer surfaces to rank when --answer-logit-probe-out is set.
    #[arg(long, default_value = "56,60,64")]
    pub(crate) answer_logit_probe_surfaces: String,

    /// Number of top adjusted logits to include in answer logit diagnostics.
    #[arg(long, default_value_t = 20)]
    pub(crate) answer_logit_probe_top_k: usize,

    /// Shadow telemetry for parser-derived count answer finalization candidates.
    #[arg(long, default_value_t = false)]
    pub(crate) count_route_memory_finalization_candidate_telemetry: bool,

    /// Default-off count lane action: replace a same-run wrong answer window with the parser-derived count answer and stop.
    #[arg(long, default_value_t = false)]
    pub(crate) count_route_memory_finalization_replacement_action: bool,

    /// Default-off count lane action widening: allow natural parser-v2 rows to replace same-run wrong answer windows and stop.
    #[arg(long, default_value_t = false)]
    pub(crate) count_route_memory_finalization_natural_v2_replacement_action: bool,

    /// Default-off count lane action: when the assistant emits an explicit separated spelling of the parsed word, aggregate that generated evidence into the replacement answer.
    #[arg(long, default_value_t = false)]
    pub(crate) count_route_memory_finalization_enumeration_aggregation_action: bool,

    /// Default-off count lane action: preserve generated enumeration evidence before a later answer-window drift by appending the aggregate answer and stopping.
    #[arg(long, default_value_t = false)]
    pub(crate) count_route_memory_finalization_enumeration_preserve_stop: bool,

    /// Default-off count lane action: if a correct same-run answer is already protected but a later LOCK line hides the numeric surface, append the protected answer and stop.
    #[arg(long, default_value_t = false)]
    pub(crate) count_route_memory_finalization_protected_lock_surface: bool,

    /// Scope for route-memory worker influence: `full`, `answer-window`, `pre-answer`, `pre-earned`, or `token_range:<start>:<end>` (endpoints inclusive **1-based** assistant decode tokens — token 1 is the first sampled answer token).
    #[arg(long, default_value = "full")]
    pub(crate) specialist_memory_worker_influence_scope: String,

    /// Apply route-memory worker influence only on transformer layers START-END inclusive (e.g., `0-8`). When unset, influence follows the physics bridge-layer policy (eval-fast singleton layer behavior).
    #[arg(long)]
    pub(crate) specialist_memory_worker_influence_layers: Option<String>,

    /// Direction mode for route-memory worker influence. target preserves existing pull-to-packet behavior; delta64 applies packet 64D as a direct transition direction.
    #[arg(long, value_enum, default_value = "target")]
    pub(crate) specialist_memory_worker_influence_direction:
        SpecialistMemoryWorkerInfluenceDirection,

    /// Force route-memory worker selection to one exact packet id for matched causal smokes.
    #[arg(long)]
    pub(crate) specialist_memory_worker_fixed_packet_id: Option<String>,

    #[arg(long, default_value_t = 0.1)]
    pub(crate) dt: f32,

    #[arg(long, default_value_t = 2.5)]
    pub(crate) gravity: f32,

    #[arg(long, default_value_t = 256)]
    pub(crate) max_steps: usize,

    /// Optional max-step override for the first turn in a scripted/session run.
    #[arg(long)]
    pub(crate) turn1_max_steps: Option<usize>,

    /// Optional max-step override for turns after the first turn.
    #[arg(long)]
    pub(crate) turn2_max_steps: Option<usize>,

    /// Runtime finalization behavior after visible LOCK-like control surfaces.
    #[arg(long, value_enum, default_value = "off")]
    pub(crate) lock_stop_policy: LockStopPolicy,

    /// Number of decode tokens allowed after LOCK when --lock-stop-policy=taper.
    #[arg(long, default_value_t = 8)]
    pub(crate) lock_taper_tokens: usize,

    /// Allow final-answer markers to count as taper boundaries after LOCK is detected.
    #[arg(long, default_value_t = false)]
    pub(crate) lock_stop_on_final_answer: bool,

    /// Enable runtime answer-boundary finalizer (literal/word_reverse/arithmetic).
    /// When the prompt matches one of those shapes and the expected answer
    /// appears in generated text, decoding stops with stop_reason
    /// answer_boundary_<kind>_seen. Replaces the wrapper-only Python finalizers
    /// in scripts/run_v2_production.sh with an in-runtime peer of FinalizationController.
    #[arg(long, default_value_t = false)]
    pub(crate) answer_boundary_finalization: bool,

    /// Keep simple arithmetic turns out of the over-governed guardrail/physics lane.
    /// This does not inject an answer; it brakes runtime steering and guardrail
    /// resampling for prompts whose answer-boundary parser recognizes integer
    /// arithmetic, so the model is not bounced away from a numeric answer it can emit.
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    pub(crate) math_governor_relief: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) naked: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) visualized: bool,

    #[arg(long, default_value_t = 32)]
    pub(crate) batch_size: usize,

    #[arg(long, default_value_t = 1000)]
    pub(crate) n: usize,

    #[arg(long, default_value_t = 0.7)]
    pub(crate) temperature: f32,

    #[arg(long, default_value_t = 1.0)]
    pub(crate) mu: f64,

    #[arg(long, default_value_t = 0.05)]
    pub(crate) sigma: f64,

    #[arg(long, default_value = "")]
    pub(crate) goal: String,

    #[arg(long, default_value_t = true)]
    pub(crate) pinn_enabled: bool,

    #[arg(long, default_value_t = 0.1)]
    pub(crate) pinn_stiffness: f64,

    #[arg(long, default_value_t = 10.0)]
    pub(crate) ghost_gravity: f64,

    /// Physics blend factor - how much physics force to apply
    #[arg(long, default_value_t = 1.5)]
    pub(crate) physics_blend: f32,

    /// Start layer for physics application (0-31 for Llama 8B)
    /// Layer Banding: Skip syntax layers (0-15), apply to semantic layers only
    #[arg(long, default_value_t = 16)]
    pub(crate) physics_start_layer: usize,

    /// End layer for physics application (0-31 for Llama 8B)
    /// Llama-3 has 32 layers (0-31)
    #[arg(long, default_value_t = 31)]
    pub(crate) physics_end_layer: usize,

    /// Use multiplicative blending (more stable) instead of additive
    #[arg(long, default_value_t = true)]
    pub(crate) multiplicative_blend: bool,

    /// Repulsion strength for black holes
    #[arg(long, default_value_t = -0.5)]
    pub(crate) repulsion_strength: f64,

    /// Comma-separated list of words to act as black holes (repulsors)
    #[arg(long, default_value = "swift,very,really,basically")]
    pub(crate) black_holes: String,

    /// Run rainbow parameter sweep test
    #[arg(long, default_value_t = false)]
    pub(crate) rainbow_test: bool,

    /// ORBITAL MODE (Phase 2)
    #[arg(long, default_value_t = false)]
    pub(crate) mode_orbital: bool,

    /// Phase 2: How fast the thought orbits the concept (0.05 - 0.5)
    #[arg(long, default_value_t = ORBIT_SPEED)]
    pub(crate) orbit_speed: f32,

    /// Phase 2: How hard the prompt anchors the orbit
    /// 🌟 GENIUS CONFIG: 0.2 = High Elasticity (Allows the "Thinking" Phase)
    #[arg(long, default_value_t = GRAVITY_WELL)]
    pub(crate) gravity_well: f32,

    /// Random seed for reproducibility
    #[arg(long, default_value_t = 42)]
    pub(crate) seed: u64,

    /// Enable the legacy physics websocket broadcast server.
    #[arg(long, default_value_t = false)]
    pub(crate) physics_ws: bool,

    /// Port for the optional physics websocket server.
    #[arg(long, default_value_t = 3002)]
    pub(crate) physics_ws_port: u16,

    /// Print the expensive full-universe "mind state" scan every N steps.
    /// Set to 0 to disable it entirely.
    #[arg(long, default_value_t = 0)]
    pub(crate) mind_state_every: usize,

    /// Emit machine-readable JSON events prefixed with [UI_EVENT] for TUI wrappers.
    #[arg(long, default_value_t = false)]
    pub(crate) ui_events_json: bool,

    /// Print the resolved model auto-scaling profile as JSON and exit.
    #[arg(long, default_value_t = false)]
    pub(crate) print_scaling_profile_json: bool,

    /// When set, use the --prompt text as-is without wrapping it in the chat template.
    /// Useful for TUI wrappers that build the full multi-turn prompt with special tokens.
    #[arg(long, default_value_t = false)]
    pub(crate) raw_prompt: bool,

    /// Path to an optional runtime bridge manifest.
    ///
    /// The niodv4-derived bridge is intentionally opt-in because niodv4 is a
    /// deprecated research source, while niodoo runtime should be clean by
    /// default. Pass a concrete path to enable bridge steering.
    #[arg(long, default_value = "off")]
    pub(crate) runtime_bridge_path: String,

    /// Runtime output mode. `research` keeps raw agency/control surfaces visible.
    /// `agency` is the primary cybernetic path: it teaches the control language,
    /// preserves pink-elephant pressure, allows visible request surfaces, and uses
    /// hidden-request inference as fallback when pressure is present but not fully surfaced.
    /// `clean` is the later ablation/demo path for suppressing visible control surfaces.
    #[arg(long, value_enum, default_value_t = RuntimeMode::Agency)]
    pub(crate) runtime_mode: RuntimeMode,

    /// Infer control requests from logits even when the model did not emit a visible tag.
    /// Disabled by default: the model should steer itself through explicit request surfaces.
    #[arg(long, default_value_t = false)]
    pub(crate) hidden_request_inference: bool,

    /// Product-level output contract. This does not suppress agency surfaces globally;
    /// it only adds final-answer boundaries when a task needs exact-form delivery.
    #[arg(long, value_enum, default_value_t = OutputContractMode::Auto)]
    pub(crate) output_contract_mode: OutputContractMode,

    /// Dev-only scripted multi-turn session file. Uses one prompt per non-empty line
    /// and runs all turns inside one live runtime instance.
    #[arg(long)]
    pub(crate) session_script: Option<PathBuf>,

    /// Interactive chat loop over one live runtime instance. Use /quit or /exit to stop.
    #[arg(long, default_value_t = false)]
    pub(crate) chat_repl: bool,

    /// Optional override for the initial system prompt.
    #[arg(long)]
    pub(crate) system_prompt_file: Option<PathBuf>,

    /// Optional JSON artifact path for writing the final turn's compressed hidden state.
    #[arg(long)]
    pub(crate) state_capture_file: Option<PathBuf>,

    /// Optional directory for writing per-turn capture artifacts during scripted sessions.
    #[arg(long)]
    pub(crate) turn_capture_dir: Option<PathBuf>,

    /// Optional directory for live post-steering hidden-state captures.
    #[arg(long)]
    pub(crate) runtime_hidden_capture_dir: Option<PathBuf>,

    /// Capture every N decode steps when --runtime-hidden-capture-dir is active. Set to 0 to disable.
    #[arg(long, default_value_t = 0)]
    pub(crate) runtime_hidden_capture_every: usize,

    /// Write a KV snapshot every N turns when --turn-capture-dir is active. Set to 0 to disable.
    #[arg(long, default_value_t = 0)]
    pub(crate) turn_kv_every: usize,

    /// Optional JSON artifact path for writing the live transformer KV cache snapshot.
    #[arg(long)]
    pub(crate) kv_state_save_file: Option<PathBuf>,

    /// Optional JSON artifact path for writing a compact human-review summary.
    #[arg(long)]
    pub(crate) human_eval_summary_file: Option<PathBuf>,

    /// Optional JSON artifact path for loading a prior transformer KV cache snapshot.
    #[arg(long)]
    pub(crate) kv_state_load_file: Option<PathBuf>,

    /// Reset KV cache between session-script turns. Prevents context-window overflow
    /// on long multi-prompt eval sessions where each turn's cumulative context fills
    /// the model's window (~prompt 27-30 in agency mode at 4096 tokens) and silently
    /// emits empty strings thereafter. Defaults OFF — agency multi-turn features
    /// (compact resume, REMEMBER reinjection, mistake_reflex) depend on cumulative KV.
    #[arg(long, default_value_t = false)]
    pub(crate) reset_kv_cache_per_turn: bool,

    /// Optional JSON artifact path for loading compact, human-readable resume anchors.
    #[arg(long)]
    pub(crate) compact_resume_state_load_file: Option<PathBuf>,

    /// Optional JSON artifact path for saving compact, human-readable resume anchors.
    #[arg(long)]
    pub(crate) compact_resume_state_save_file: Option<PathBuf>,

    /// Inject loaded compact resume anchors into resume/continue turns.
    #[arg(long, default_value_t = true)]
    pub(crate) compact_resume_state_injection: bool,

    /// Optional JSONL ledger for user-confirmed mistake/correction memory.
    #[arg(long)]
    pub(crate) mistake_memory_path: Option<PathBuf>,

    /// Capture user-confirmed corrections into --mistake-memory-path.
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    pub(crate) mistake_memory_learning: bool,

    /// Inject matching mistake memory before generation.
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    pub(crate) mistake_memory_injection: bool,

    /// Stop a mistake-memory turn once the accepted correction answer has been emitted.
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    pub(crate) mistake_memory_stop_on_accepted: bool,

    /// Optional JSONL ledger for skill-correction reflexes that influence process without answers.
    #[arg(long)]
    pub(crate) mistake_reflex_path: Option<PathBuf>,

    /// Observe-only GMMS applicability prompt dump; exits before model load/generation.
    #[arg(long)]
    pub(crate) gmms_observe_dump_prompt: Option<String>,

    /// Optional JSON output path for --gmms-observe-dump-prompt.
    #[arg(long)]
    pub(crate) gmms_observe_dump_out: Option<PathBuf>,

    /// Maximum observe-only GMMS applicability rows to dump.
    #[arg(long, default_value_t = 3)]
    pub(crate) gmms_observe_dump_limit: usize,

    /// No-generation dump of the turn-start GMMS observe UI event record; exits before model load/generation.
    #[arg(long)]
    pub(crate) gmms_observe_turn_start_event_dump_prompt: Option<String>,

    /// Optional JSON output path for --gmms-observe-turn-start-event-dump-prompt.
    #[arg(long)]
    pub(crate) gmms_observe_turn_start_event_dump_out: Option<PathBuf>,

    /// Negative no-generation fixture for --gmms-observe-turn-start-event-dump-prompt; writes a sanitized rejected event.
    #[arg(long, default_value_t = false)]
    pub(crate) gmms_observe_turn_start_event_dump_unsafe_fixture: bool,

    /// Emit observe-only GMMS applicability metadata at turn start. Does not inject prompts or call the runtime matcher.
    #[arg(long, default_value_t = false)]
    pub(crate) gmms_observe_turn_start: bool,

    /// Maximum observe-only GMMS applicability rows to emit at turn start.
    #[arg(long, default_value_t = 1)]
    pub(crate) gmms_observe_turn_start_limit: usize,

    /// Mistake reflex runtime mode. shadow logs matches; influence adds process hints and gates LOCK.
    #[arg(long, value_enum, default_value = "off")]
    pub(crate) mistake_reflex_mode: MistakeReflexMode,

    /// Mistake reflex action surface. text-hint is v0 scaffolding; summary-hint ablates the prompt surface; evidence-gate and hidden-control reduce prompt text.
    #[arg(long, value_enum, default_value = "text-hint")]
    pub(crate) mistake_reflex_action_mode: MistakeReflexActionMode,

    /// Capture user-confirmed skill corrections into --mistake-reflex-path.
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    pub(crate) mistake_reflex_learning: bool,

    /// Stop once a mistake reflex has earned a corrected answer with required evidence.
    #[arg(long, action = clap::ArgAction::Set, default_value_t = false)]
    pub(crate) mistake_reflex_stop_on_earned_answer: bool,

    /// Trigger a bounded hidden-control retry pressure when a skill reflex detects the old path.
    #[arg(long, action = clap::ArgAction::Set, default_value_t = false)]
    pub(crate) mistake_reflex_retry_on_old_mistake: bool,

    /// Maximum retry actuator firings per turn for mistake reflex hidden-control.
    #[arg(long, default_value_t = 1)]
    pub(crate) mistake_reflex_retry_max: usize,

    /// Number of following decode steps to keep the retry token shield active.
    #[arg(long, default_value_t = 32)]
    pub(crate) mistake_reflex_retry_shield_tokens: usize,

    /// On retry, force only a visible control-surface tag back into the stream; never force the answer.
    #[arg(long, action = clap::ArgAction::Set, default_value_t = false)]
    pub(crate) mistake_reflex_retry_inject_control_surface: bool,

    /// Tokens to watch after a model-authored earned reflex answer before preserving it.
    #[arg(long, default_value_t = 24)]
    pub(crate) mistake_reflex_earned_taper_tokens: usize,

    /// Optional Niodv4 Unicode memory packet JSONL used to attach route slices to mistake reflexes.
    #[arg(long)]
    pub(crate) mistake_reflex_packet_index: Option<PathBuf>,

    /// Optional hyper-dense Unicode state string used to seed a new session.
    #[arg(long)]
    pub(crate) secret_sauce: Option<String>,

    /// Optional explicit version selector for restoring 64-glyph secret sauce payloads.
    #[arg(long, value_enum, default_value_t = SecretSauceInputVersion::Auto)]
    pub(crate) secret_sauce_version: SecretSauceInputVersion,

    /// Warm-start the KV cache with a small synthetic ghost prefix derived from the secret sauce.
    #[arg(long, default_value_t = false)]
    pub(crate) secret_sauce_kv_prefix: bool,

    /// Number of warm-start ghost tokens to seed before the first prompt token.
    #[arg(long, default_value_t = 4)]
    pub(crate) secret_sauce_kv_prefix_len: usize,

    /// Ablation: disable the periodic routing/controller cadence during decoding.
    #[arg(long, default_value_t = false)]
    pub(crate) ablate_periodic_controller: bool,

    /// Ablation: disable live motif minting so bridge/runtime memory force can be isolated.
    #[arg(long, default_value_t = false)]
    pub(crate) ablate_live_motifs: bool,

    /// Experimental minimum activation gate for imported bridge motifs. Default preserves pressure-only gating.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) bridge_motif_gate_floor: f32,

    /// Ablation: disable conflict-aware routing and fall back to nearest-only motif choice.
    #[arg(long, default_value_t = false)]
    pub(crate) ablate_conflict_routing: bool,

    /// Ablation: disable the structured restored re-entry clamp.
    #[arg(long, default_value_t = false)]
    pub(crate) ablate_reentry_clamp: bool,

    /// Ablation: disable the crystal ratchet that tightens motifs under structured streaks.
    #[arg(long, default_value_t = false)]
    pub(crate) ablate_crystal_ratchet: bool,

    /// Ablation: disable early promotion override shortcuts for clamp-driven reasoning motifs.
    #[arg(long, default_value_t = false)]
    pub(crate) ablate_promotion_override: bool,

    // Dev-only: runtime overrides for routing/scorer constants (use 0.0 for default).
    #[arg(long, default_value_t = 0.0)]
    pub(crate) dev_structured_candidate_task_sim: f32,
    #[arg(long, default_value_t = 0.0)]
    pub(crate) dev_structured_candidate_bonus_scale: f32,
    #[arg(long, default_value_t = 0.0)]
    pub(crate) dev_neutral_basin_penalty_scale: f32,
    #[arg(long, default_value_t = 0.0)]
    pub(crate) dev_task_utility_bonus_scale: f32,
    #[arg(long, default_value_t = 0.0)]
    pub(crate) dev_fragmentation_discount: f32,
    #[arg(long, default_value_t = 0.0)]
    pub(crate) dev_restored_topology_floor_signal: f32,
    #[arg(long, default_value_t = 0.0)]
    pub(crate) dev_restored_topology_floor_tightness: f32,
    #[arg(long, default_value_t = 0.0)]
    pub(crate) dev_structured_candidate_escalation_topology: f32,
    #[arg(long, default_value_t = 0.0)]
    pub(crate) dev_structured_candidate_escalation_task: f32,
    #[arg(long, default_value_t = 0.0)]
    pub(crate) dev_routing_stickiness_bonus: f32,
    #[arg(long, default_value_t = 0.0)]
    pub(crate) dev_routing_stickiness_ticks: f32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum RuntimeMode {
    Research,
    Agency,
    Clean,
}

impl RuntimeMode {
    pub(crate) fn is_agency(self) -> bool {
        matches!(self, Self::Agency)
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Research => "research",
            Self::Agency => "agency",
            Self::Clean => "clean",
        }
    }

    pub(crate) fn teaches_control_language(self) -> bool {
        matches!(self, Self::Research | Self::Agency)
    }

    pub(crate) fn uses_control_shield(self) -> bool {
        matches!(self, Self::Clean)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum OutputContractMode {
    #[value(name = "off", alias = "legacy")]
    Off,
    #[value(name = "auto")]
    Auto,
    #[value(
        name = "collaborative_transparency",
        alias = "collaborative-transparency"
    )]
    CollaborativeTransparency,
    #[value(name = "exact_form_delivery", alias = "exact-form-delivery")]
    ExactFormDelivery,
}

impl OutputContractMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::CollaborativeTransparency => "collaborative_transparency",
            Self::ExactFormDelivery => "exact_form_delivery",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum SpecialistMemoryWorkerMode {
    Off,
    Shadow,
    Influence,
}

impl SpecialistMemoryWorkerMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Shadow => "shadow",
            Self::Influence => "influence",
        }
    }

    pub(crate) fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

/// Resolved worker-influence activation window (decoded assistant tokens).
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SpecialistMemoryWorkerInfluenceScope {
    Full,
    AnswerWindow,
    PreAnswer,
    PreEarned,
    /// Fires only on the single decode step where assistant text first crosses from
    /// `pre_answer_active=true` (still in reasoning) to `answer_window_active=true`
    /// (working answer marker just emitted). Single-shot injection.
    AtAnswerBoundary,
    /// Inclusive decode-token ordinal **during assistant generation**. Token 1 is the first sampled
    /// token (`current_step == 0`).
    TokenRange {
        start_token_1_based: usize,
        end_token_1_based: usize,
    },
}

pub(crate) fn parse_specialist_memory_worker_influence_scope_arg(
    raw: &str,
) -> anyhow::Result<SpecialistMemoryWorkerInfluenceScope> {
    let s = raw.trim();
    if let Some(spec) = s
        .strip_prefix("token_range:")
        .or_else(|| s.strip_prefix("token-range:"))
    {
        let spec = spec.trim();
        let (start_s, end_s) = spec
            .split_once(':')
            .map(|(a, b)| (a.trim(), b.trim()))
            .or_else(|| spec.split_once('-').map(|(a, b)| (a.trim(), b.trim())))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "token_range expects token_range:<start>:<end> or token_range:<start>-<end> (inclusive 1-based tokens)"
                )
            })?;
        let start_token_1_based: usize = start_s
            .parse()
            .map_err(|_| anyhow::anyhow!("bad token_range start integer in '{raw}'"))?;
        let end_token_1_based: usize = end_s
            .parse()
            .map_err(|_| anyhow::anyhow!("bad token_range end integer in '{raw}'"))?;
        if start_token_1_based == 0 || end_token_1_based == 0 {
            anyhow::bail!(
                "--specialist-memory-worker-influence-scope token_range endpoints must be positive (1-based tokens)"
            );
        }
        if start_token_1_based > end_token_1_based {
            anyhow::bail!(
                "token_range start must be <= end (got {start_token_1_based}-{end_token_1_based})"
            );
        }
        return Ok(SpecialistMemoryWorkerInfluenceScope::TokenRange {
            start_token_1_based,
            end_token_1_based,
        });
    }

    match s.replace('_', "-").to_ascii_lowercase().as_str() {
        "full" => Ok(SpecialistMemoryWorkerInfluenceScope::Full),
        "answer-window" => Ok(SpecialistMemoryWorkerInfluenceScope::AnswerWindow),
        "pre-answer" => Ok(SpecialistMemoryWorkerInfluenceScope::PreAnswer),
        "pre-earned" => Ok(SpecialistMemoryWorkerInfluenceScope::PreEarned),
        "at-answer-boundary" => Ok(SpecialistMemoryWorkerInfluenceScope::AtAnswerBoundary),
        other => anyhow::bail!(
            "unknown specialist worker influence scope '{other}'. Use full, answer-window, pre-answer, pre-earned, at-answer-boundary, or token_range:<start>:<end>"
        ),
    }
}

pub(crate) fn parse_specialist_memory_worker_influence_layers_arg(
    raw: &str,
) -> anyhow::Result<(usize, usize)> {
    let s = raw.trim();
    let (a, b) = s.split_once('-').ok_or_else(|| {
        anyhow::anyhow!(
            "--specialist-memory-worker-influence-layers expects <min>-<max> inclusive (e.g., 0-8)"
        )
    })?;
    let lo = a
        .trim()
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("bad layer integer in '{}'", a.trim()))?;
    let hi = b
        .trim()
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("bad layer integer in '{}'", b.trim()))?;
    if lo > hi {
        anyhow::bail!("worker influence layer min must be <= max (got {lo}-{hi})");
    }
    Ok((lo, hi))
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum SpecialistMemoryWorkerInfluenceDirection {
    Target,
    Residual64,
    Delta64,
}

impl SpecialistMemoryWorkerInfluenceDirection {
    pub(crate) fn telemetry_label(self) -> &'static str {
        match self {
            Self::Target => "decoded_64d_normalized",
            Self::Residual64 => "decoded_64d_residual_normalized",
            Self::Delta64 => "decoded_64d_delta_normalized",
        }
    }
}
