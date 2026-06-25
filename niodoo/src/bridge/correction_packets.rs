//! VQ-keyed correction-packet store. The "scar tissue → reflex" primitive.
//!
//! Each packet stores a 64D latent target plus rule parameters, indexed by the codebook
//! bucket (`vq_code`) of the failure state it was minted from. Per step, the runtime
//! encodes the probe via the trained codec, looks up packets matching the bucket, and for
//! every firing packet returns a 64D pull-toward-target delta. The caller (apply_forces)
//! converts each delta into a 4096D hidden-state force via `codec.decode(z + delta) -
//! codec.decode(z)` and adds it to `probe_force` — same pipeline the rule-based phase2
//! specialist already uses, generalized from 2 dims to all 64.
//!
//! North Star alignment:
//! - "compress the correction from explicit memory into route/reflex form" — each
//!   packet IS that compressed correction.
//! - "recognize related future situations" — vq_code lookup is O(1) bucket recall.
//! - "steer away from old failure basins" — packet pulls probe latent toward the
//!   stored target, away from the failure region the bucket represents.
//!
//! The packet store is read-only at runtime; minting happens out-of-band (REMEMBER tag
//! → packet writer, future iteration).
//!
//! Storage format: JSONL, one packet per line. Records are loaded into an in-memory
//! `HashMap<u8, Vec<CorrectionPacket>>` for O(1) lookup keyed on `vq_code`.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub const PACKET_AUTHORITY_ALLOW_THRESHOLD: f32 = 0.65;

#[derive(Debug, Clone, Copy)]
pub struct PacketAuthorityContext<'a> {
    pub source_target_override: Option<&'a str>,
    pub prompt_family_matched: bool,
    pub current_prompt_hash: Option<&'a str>,
    pub route_margin: f32,
    pub nearest_ghost_present: bool,
    pub target_distance: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PacketAuthorityDecision {
    pub score: f32,
    pub allowed: bool,
    pub reason: String,
    pub blocked_reason: String,
}

impl PacketAuthorityDecision {
    pub fn allow_all(reason: &str) -> Self {
        Self {
            score: 1.0,
            allowed: true,
            reason: reason.to_string(),
            blocked_reason: "none".to_string(),
        }
    }
}

/// One stored correction reflex. Keyed on the codebook bucket (`vq_code`) of the
/// failure-state that minted it. Fire condition: 64D distance from probe latent to
/// `target_z_64d` exceeds `distance_threshold`.
#[derive(Debug, Clone)]
pub struct CorrectionPacket {
    pub packet_id: String,
    pub vq_code: u8,
    pub target_z_64d: [f32; 64],
    /// Optional factual payload direction in the same 64D packet space. When present
    /// and the runtime enables payload blending, this is orthogonalized against the
    /// target-pull direction and mixed into the force vector so the packet can carry
    /// more than a generic pull toward a centroid.
    pub payload_z_64d: Option<[f32; 64]>,
    pub pull_strength: f32,
    pub distance_threshold: f32,
    pub source_label: String,
    pub created_step: u64,
    /// Optional hybrid packet metadata fields carried from the minting path.
    /// These are inspectable continuity hints (text_fact/route/agency/force/LOCK
    /// boundary) and should survive load/state-out process boundaries even when
    /// the packet runtime does not directly consume them for force computation.
    pub text_fact: Option<String>,
    pub route_code: Option<String>,
    pub route_motif_id: Option<String>,
    pub target_ghost_id: Option<String>,
    pub nearest_ghost_distance: Option<f32>,
    pub second_nearest_ghost_distance: Option<f32>,
    pub route_margin: Option<f32>,
    pub agency_transition: Option<String>,
    pub force_policy: Option<String>,
    pub force_pull_strength: Option<f32>,
    pub force_distance_threshold: Option<f32>,
    pub force_decay_rate: Option<f32>,
    pub force_unfold_factor: Option<f32>,
    pub force_unfold_retry_factor: Option<f32>,
    pub answer_lock_boundary: Option<String>,
    pub projection_strategy: Option<String>,
    pub ghost_pull_delta_norm: Option<f32>,
    pub fire_count: AtomicU64Wrapper,
    pub last_fire_step: AtomicU64Wrapper,
    /// Per-packet decay rate override. When `Some(r)` with `0 < r < 1`, this packet's
    /// `effective_pull = pull_strength * r ^ fire_count` regardless of any global decay
    /// rate the runtime configured. When `Some(1.0)` (or any value outside `(0, 1)`),
    /// no decay is applied — used by LOCK-derived earned-answer packets that should
    /// persist with full pull. When `None`, the engine's global `--correction-packet-decay-rate`
    /// is used — typical for REMEMBER scaffolding and end-of-run captures.
    pub decay_rate: Option<f32>,
    /// Per-packet unfold-factor override. Symmetric to `decay_rate`: when the relapse
    /// trigger fires (vq_encode_error spike), the runtime multiplies firing packets'
    /// effective_pull by an unfold factor. When this field is `Some(f)`, that value
    /// replaces the engine global for THIS packet only. LOCK-derived earned packets
    /// stamp `Some(1.0)` so they ignore relapse boosting (they're already preserved
    /// at full pull and don't need extra). When `None`, the engine's global
    /// `--correction-packet-unfold-factor` is used — typical for REMEMBER scaffolding.
    pub unfold_factor: Option<f32>,
    /// Invalidation flag. When true, `forward_with_pull` returns `None` so the packet
    /// no longer fires. Set by the runtime when the user contradicts a prior LOCK
    /// (the contradicted packet's payload-key matches `lh_<hash>` in this packet's
    /// `packet_id`). Persisted via the JSONL writer so invalidation survives process
    /// boundaries — invalidated scar tissue stays out of the way until explicitly
    /// re-validated.
    pub invalidated: AtomicBoolWrapper,
    /// Optional agency payload key (e.g. `final` from `final=ship_thursday`). Used
    /// by `invalidate_by_payload_key` for semantic invalidation: any packet sharing
    /// the contradicted LOCK's key gets switched off, even when the exact payload
    /// string differs from the contradicted one. Empty when the packet doesn't carry
    /// an agency-shaped payload.
    pub payload_key: Option<String>,
    /// Per-packet retry-source unfold factor override (mirror of `unfold_factor`
    /// but specifically for the mistake_reflex_retry trigger of §10ae). When
    /// `Some(f)`, this packet uses `f` as its retry-relapse factor regardless of
    /// the engine global. Earned packets stamp `Some(1.0)` to ignore retry-boost
    /// just as they ignore OOD-boost. When `None`, the engine's global
    /// `--correction-packet-unfold-retry-factor` (or `unfold_factor` fallback)
    /// applies — typical for REMEMBER scaffolding.
    pub unfold_retry_factor: Option<f32>,
}

/// Wrapper around `AtomicBool` mirroring `AtomicU64Wrapper`. Used for the per-packet
/// `invalidated` flag, which the runtime sets when a LOCK contradiction arrives —
/// invalidated packets stay in the store but `forward_with_pull` returns `None` for
/// them so they no longer fire.
#[derive(Debug)]
pub struct AtomicBoolWrapper(pub AtomicBool);

impl Clone for AtomicBoolWrapper {
    fn clone(&self) -> Self {
        Self(AtomicBool::new(self.0.load(Ordering::Relaxed)))
    }
}

impl AtomicBoolWrapper {
    pub fn new(value: bool) -> Self {
        Self(AtomicBool::new(value))
    }

    pub fn load(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    pub fn store(&self, value: bool) {
        self.0.store(value, Ordering::Relaxed);
    }
}

/// Wrapper around `AtomicU64` that derives Clone. Each clone produces a new atomic
/// initialized to the current value of the source — lets the runtime hold its own
/// copy of fire counters in the per-step lookup without sharing mutation state across
/// HashMap clones (the shared store is `Arc`'d at the engine boundary).
#[derive(Debug)]
pub struct AtomicU64Wrapper(pub AtomicU64);

impl Clone for AtomicU64Wrapper {
    fn clone(&self) -> Self {
        Self(AtomicU64::new(self.0.load(Ordering::Relaxed)))
    }
}

impl AtomicU64Wrapper {
    pub fn new(value: u64) -> Self {
        Self(AtomicU64::new(value))
    }

    pub fn load(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    pub fn store(&self, value: u64) {
        self.0.store(value, Ordering::Relaxed);
    }

    pub fn fetch_add(&self, delta: u64) -> u64 {
        self.0.fetch_add(delta, Ordering::Relaxed)
    }
}

fn orthogonal_payload_unit(
    payload_z_64d: &[f32; 64],
    target_direction_unit: &[f32; 64],
) -> Option<[f32; 64]> {
    let mut payload_norm_sq = 0.0f32;
    for value in payload_z_64d {
        if !value.is_finite() {
            return None;
        }
        payload_norm_sq += value * value;
    }
    let payload_norm = payload_norm_sq.sqrt();
    if !payload_norm.is_finite() || payload_norm <= 1e-6 {
        return None;
    }

    let mut payload_unit = [0f32; 64];
    let mut dot = 0.0f32;
    for i in 0..64 {
        payload_unit[i] = payload_z_64d[i] / payload_norm;
        dot += payload_unit[i] * target_direction_unit[i];
    }

    let mut orthogonal = [0f32; 64];
    let mut orthogonal_norm_sq = 0.0f32;
    for i in 0..64 {
        orthogonal[i] = payload_unit[i] - dot * target_direction_unit[i];
        orthogonal_norm_sq += orthogonal[i] * orthogonal[i];
    }
    let orthogonal_norm = orthogonal_norm_sq.sqrt();
    if !orthogonal_norm.is_finite() || orthogonal_norm <= 1e-6 {
        return None;
    }
    for slot in orthogonal.iter_mut() {
        *slot /= orthogonal_norm;
    }
    Some(orthogonal)
}

#[derive(Deserialize)]
struct CorrectionPacketJson {
    packet_id: String,
    vq_code: u8,
    /// Optional numeric target (legacy + preferred).
    #[serde(default)]
    target_z_64d: Option<Vec<f64>>,
    /// Optional Unicode secret_sauce V3 transport for the same 64D target.
    /// When `target_z_64d` is absent, loaders decode this into the live
    /// `CorrectionPacket.target_z_64d`.
    #[serde(default)]
    target_z_unicode_v3: Option<String>,
    /// Optional factual payload direction in 64D packet space. Legacy packet stores
    /// omit this and remain direction-only.
    #[serde(default)]
    payload_z_64d: Option<Vec<f64>>,
    /// Optional Unicode transport for `payload_z_64d`.
    #[serde(default)]
    payload_z_unicode_v3: Option<String>,
    pull_strength: f64,
    distance_threshold: f64,
    #[serde(default)]
    source_label: String,
    #[serde(default)]
    created_step: u64,
    /// Optional hybrid packet metadata fields. Absent in legacy packet stores.
    #[serde(default)]
    text_fact: Option<String>,
    #[serde(default)]
    route_code: Option<String>,
    #[serde(default)]
    route_motif_id: Option<String>,
    #[serde(default)]
    target_ghost_id: Option<String>,
    #[serde(default)]
    nearest_ghost_distance: Option<f32>,
    #[serde(default)]
    second_nearest_ghost_distance: Option<f32>,
    #[serde(default)]
    route_margin: Option<f32>,
    #[serde(default)]
    agency_transition: Option<String>,
    #[serde(default)]
    force_policy: Option<String>,
    #[serde(default)]
    force_pull_strength: Option<f32>,
    #[serde(default)]
    force_distance_threshold: Option<f32>,
    #[serde(default)]
    force_decay_rate: Option<f32>,
    #[serde(default)]
    force_unfold_factor: Option<f32>,
    #[serde(default)]
    force_unfold_retry_factor: Option<f32>,
    #[serde(default)]
    answer_lock_boundary: Option<String>,
    #[serde(default)]
    projection_strategy: Option<String>,
    #[serde(default)]
    ghost_pull_delta_norm: Option<f32>,
    /// Persisted counter from prior runs. Default 0 (fresh packet) for backwards
    /// compatibility with packet stores written before persistence existed.
    #[serde(default)]
    fire_count: u64,
    /// Most recent step at which this packet fired (across all sessions). Default 0.
    #[serde(default)]
    last_fire_step: u64,
    /// Optional per-packet decay rate override. Absent in legacy packets.
    #[serde(default)]
    decay_rate: Option<f32>,
    /// Optional per-packet unfold-factor override. Absent in legacy packets.
    #[serde(default)]
    unfold_factor: Option<f32>,
    /// Persisted invalidation flag. Default false. Loaders honour this on read so a
    /// previously contradicted packet stays inactive across process boundaries.
    #[serde(default)]
    invalidated: bool,
    /// Optional payload key used for semantic invalidation. e.g., for an agency-hands
    /// payload `final=ship_thursday`, the key is `final`. Lets the runtime invalidate
    /// any packet whose key matches a contradicted LOCK's key, even if the exact
    /// payload string (and thus `lh_<hash>`) differs.
    #[serde(default)]
    payload_key: Option<String>,
    /// Optional per-packet retry-source unfold factor override (§10ah). Absent in
    /// legacy packets.
    #[serde(default)]
    unfold_retry_factor: Option<f32>,
}

impl CorrectionPacket {
    /// Build from JSON record. Returns Err if `target_z_64d` is not exactly 64 long.
    fn from_json(raw: CorrectionPacketJson) -> Result<Self, Box<dyn std::error::Error>> {
        let mut target = [0f32; 64];
        if let Some(vec) = raw.target_z_64d.as_ref() {
            if vec.len() != 64 {
                return Err(format!(
                    "correction packet '{}': target_z_64d must be 64 dims, got {}",
                    raw.packet_id,
                    vec.len()
                )
                .into());
            }
            for (i, &v) in vec.iter().enumerate() {
                target[i] = v as f32;
            }
        } else if let Some(unicode) = raw.target_z_unicode_v3.as_deref() {
            let decoded = super::secret_sauce::decode_v3_sentence_anchor(unicode).map_err(|e| {
                let msg = format!(
                    "correction packet '{}': failed to decode target_z_unicode_v3: {e}",
                    raw.packet_id
                );
                std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
            })?;
            if decoded.len() != 64 {
                return Err(format!(
                    "correction packet '{}': decoded target_z_unicode_v3 must be 64 dims, got {}",
                    raw.packet_id,
                    decoded.len()
                )
                .into());
            }
            for (i, &v) in decoded.iter().enumerate() {
                target[i] = v;
            }
        } else {
            return Err(format!(
                "correction packet '{}': missing target (need target_z_64d or target_z_unicode_v3)",
                raw.packet_id
            )
            .into());
        }
        let payload_z_64d = if let Some(vec) = raw.payload_z_64d.as_ref() {
            if vec.len() != 64 {
                return Err(format!(
                    "correction packet '{}': payload_z_64d must be 64 dims, got {}",
                    raw.packet_id,
                    vec.len()
                )
                .into());
            }
            let mut payload = [0f32; 64];
            for (i, &v) in vec.iter().enumerate() {
                payload[i] = v as f32;
            }
            Some(payload)
        } else if let Some(unicode) = raw.payload_z_unicode_v3.as_deref() {
            let decoded = super::secret_sauce::decode_v3_sentence_anchor(unicode).map_err(|e| {
                let msg = format!(
                    "correction packet '{}': failed to decode payload_z_unicode_v3: {e}",
                    raw.packet_id
                );
                std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
            })?;
            if decoded.len() != 64 {
                return Err(format!(
                    "correction packet '{}': decoded payload_z_unicode_v3 must be 64 dims, got {}",
                    raw.packet_id,
                    decoded.len()
                )
                .into());
            }
            let mut payload = [0f32; 64];
            for (i, &v) in decoded.iter().enumerate() {
                payload[i] = v;
            }
            Some(payload)
        } else {
            None
        };
        Ok(Self {
            packet_id: raw.packet_id,
            vq_code: raw.vq_code,
            target_z_64d: target,
            payload_z_64d,
            pull_strength: raw.pull_strength as f32,
            distance_threshold: raw.distance_threshold as f32,
            source_label: raw.source_label,
            created_step: raw.created_step,
            text_fact: raw.text_fact,
            route_code: raw.route_code,
            route_motif_id: raw.route_motif_id,
            target_ghost_id: raw.target_ghost_id,
            nearest_ghost_distance: raw.nearest_ghost_distance,
            second_nearest_ghost_distance: raw.second_nearest_ghost_distance,
            route_margin: raw.route_margin,
            agency_transition: raw.agency_transition,
            force_policy: raw.force_policy,
            force_pull_strength: raw.force_pull_strength,
            force_distance_threshold: raw.force_distance_threshold,
            force_decay_rate: raw.force_decay_rate,
            force_unfold_factor: raw.force_unfold_factor,
            force_unfold_retry_factor: raw.force_unfold_retry_factor,
            answer_lock_boundary: raw.answer_lock_boundary,
            projection_strategy: raw.projection_strategy,
            ghost_pull_delta_norm: raw.ghost_pull_delta_norm,
            // Carry persisted counters forward so decay and unfold computed against
            // `effective_pull` reflect cross-session experience, not just within-run
            // history. Missing fields default to 0 — equivalent to a fresh packet.
            fire_count: AtomicU64Wrapper::new(raw.fire_count),
            last_fire_step: AtomicU64Wrapper::new(raw.last_fire_step),
            decay_rate: raw.decay_rate,
            unfold_factor: raw.unfold_factor,
            invalidated: AtomicBoolWrapper::new(raw.invalidated),
            payload_key: raw.payload_key,
            unfold_retry_factor: raw.unfold_retry_factor,
        })
    }

    /// RC5 live-mint helper: rebuild a `CorrectionPacket` from the exact
    /// `serde_json::Value` record the JSONL writer emitted. Goes through the same
    /// `CorrectionPacketJson` deserialize + `from_json` path the on-disk loader
    /// uses, so a packet inserted live this session is byte-identical to the one a
    /// future process would load from the out-file. Returns Err on a malformed
    /// round-trip (caller logs and skips the live insert — the file write already
    /// succeeded, so a future process still recovers the packet).
    pub fn from_json_value(
        value: serde_json::Value,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let raw: CorrectionPacketJson = serde_json::from_value(value)?;
        Self::from_json(raw)
    }

    /// Resolve which unfold factor to apply to this packet when relapse fires.
    /// Per-packet `unfold_factor.unwrap_or(engine_unfold_factor)`. Bounded to
    /// `>= 0.0`. Earned packets typically stamp `Some(1.0)` to skip the boost.
    pub fn effective_unfold_factor(&self, engine_unfold_factor: f32) -> f32 {
        self.unfold_factor.unwrap_or(engine_unfold_factor).max(0.0)
    }

    /// Resolve which retry-source unfold factor to apply to this packet when
    /// retry-relapse fires. Per-packet
    /// `unfold_retry_factor.unwrap_or(engine_retry_factor)`, bounded to `>= 0.0`.
    /// Earned packets typically stamp `Some(1.0)` to skip retry-boost just as
    /// they skip OOD-boost. Mirror of `effective_unfold_factor`.
    pub fn effective_unfold_retry_factor(&self, engine_retry_factor: f32) -> f32 {
        self.unfold_retry_factor
            .unwrap_or(engine_retry_factor)
            .max(0.0)
    }

    /// Compute the 64D pull-toward-target delta with the packet's stored `pull_strength`.
    /// Convenience wrapper around `forward_with_pull(probe_z, self.pull_strength)`.
    pub fn forward(&self, probe_z: &[f32; 64]) -> Option<[f32; 64]> {
        self.forward_with_pull(probe_z, self.pull_strength)
    }

    /// Compute the 64D pull-toward-target delta with a caller-supplied `effective_pull`.
    /// Used by the runtime decay path: `effective_pull = pull_strength * (decay_rate ^ fire_count)`
    /// so packets that fire often gradually weaken — the "decay scaffolding as competence
    /// improves" North Star primitive.
    ///
    /// Returns `None` when the packet is invalidated, when the probe is within
    /// `distance_threshold` of the target (rule doesn't fire), or when the resulting
    /// magnitude is below 1e-9. The returned vector points from probe toward target
    /// with L2 norm equal to `effective_pull` (raw, unclamped — caller is responsible
    /// for any downstream 4096D clamping).
    pub fn forward_with_pull(&self, probe_z: &[f32; 64], effective_pull: f32) -> Option<[f32; 64]> {
        self.forward_with_payload_blend(probe_z, effective_pull, 0.0)
    }

    /// Compute the 64D pull-toward-target delta with optional factual-payload blend.
    ///
    /// `payload_blend` is clamped to `[0, 1]`. `0` is exactly the legacy path. When
    /// blending is active and the packet has `payload_z_64d`, the payload vector is
    /// normalized, projected off the target-pull direction, mixed with that direction,
    /// normalized again, then scaled to `effective_pull`. The returned vector keeps the
    /// same L2 budget as the legacy pull while carrying an orthogonal payload component.
    pub fn forward_with_payload_blend(
        &self,
        probe_z: &[f32; 64],
        effective_pull: f32,
        payload_blend: f32,
    ) -> Option<[f32; 64]> {
        if self.invalidated.load() {
            return None;
        }
        if !effective_pull.is_finite() || effective_pull.abs() < 1e-9 {
            return None;
        }
        let mut direction = [0f32; 64];
        let mut dist_sq = 0f32;
        for i in 0..64 {
            direction[i] = self.target_z_64d[i] - probe_z[i];
            dist_sq += direction[i] * direction[i];
        }
        let dist = dist_sq.sqrt();
        if !dist.is_finite() || dist <= self.distance_threshold {
            return None;
        }
        for slot in direction.iter_mut() {
            *slot /= dist.max(1e-6);
        }

        let mut unit = direction;
        let blend = payload_blend.clamp(0.0, 1.0);
        if blend > 1e-6 {
            if let Some(payload) = self.payload_z_64d.as_ref() {
                if let Some(payload_unit) = orthogonal_payload_unit(payload, &direction) {
                    let mut combined = [0f32; 64];
                    let mut combined_norm_sq = 0f32;
                    let direction_weight = 1.0 - blend;
                    for i in 0..64 {
                        combined[i] = direction[i] * direction_weight + payload_unit[i] * blend;
                        combined_norm_sq += combined[i] * combined[i];
                    }
                    let combined_norm = combined_norm_sq.sqrt();
                    if combined_norm.is_finite() && combined_norm > 1e-6 {
                        for i in 0..64 {
                            unit[i] = combined[i] / combined_norm;
                        }
                    }
                }
            }
        }

        let mut out = [0f32; 64];
        for i in 0..64 {
            out[i] = unit[i] * effective_pull;
        }
        Some(out)
    }

    /// Mark this packet invalidated. Subsequent `forward_with_pull` calls return
    /// `None`, and the writer persists `invalidated: true` in the JSONL so the
    /// invalidation survives process restarts.
    pub fn invalidate(&self) {
        self.invalidated.store(true);
    }

    /// Mark this packet active again — the inverse of `invalidate`. Used when the
    /// user re-affirms a previously contradicted LOCK ("never mind, the original
    /// was right"). Subsequent `forward_with_pull` calls fire as normal; the
    /// writer omits the `invalidated` field on the next persistence so a fresh
    /// load picks up the active state cleanly.
    pub fn revalidate(&self) {
        self.invalidated.store(false);
    }

    /// Compute the per-fire effective pull strength under exponential decay.
    /// `effective_pull = pull_strength * resolved_decay.powi(fire_count_capped)`.
    /// `fire_count` is read at lookup time (before any increment from this fire).
    /// `resolved_decay` is `self.decay_rate.unwrap_or(engine_decay_rate)` — the
    /// per-packet override takes precedence over the engine global, letting
    /// LOCK-derived earned-answer packets stamp `decay_rate=Some(1.0)` to skip
    /// decay even when the engine decays scaffolding aggressively.
    /// Returns the raw `pull_strength` when the resolved decay is outside `(0, 1)`.
    pub fn effective_pull(&self, engine_decay_rate: f32) -> f32 {
        let resolved_decay = self.decay_rate.unwrap_or(engine_decay_rate);
        if !(resolved_decay > 0.0 && resolved_decay < 1.0) {
            return self.pull_strength;
        }
        let fire_count = self.fire_count.load();
        let exponent = fire_count.min(i32::MAX as u64) as i32;
        let factor = resolved_decay.powi(exponent);
        if !factor.is_finite() {
            return 0.0;
        }
        self.pull_strength * factor
    }

    pub fn record_fire(&self, current_step: u64) {
        self.fire_count.fetch_add(1);
        self.last_fire_step.store(current_step);
    }
}

pub fn decide_packet_authority(
    packet: &CorrectionPacket,
    ctx: PacketAuthorityContext<'_>,
) -> PacketAuthorityDecision {
    let source_label = packet.source_label.to_ascii_lowercase();
    let packet_id = packet.packet_id.to_ascii_lowercase();
    let mut score = 0.0f32;
    let mut reasons: Vec<&'static str> = Vec::new();
    let mut blocked: Vec<&'static str> = Vec::new();

    let source_target_match = ctx
        .source_target_override
        .map(|target| {
            let marker = format!("target_id={}", target.to_ascii_lowercase());
            source_label.contains(&marker)
        })
        .unwrap_or(false);
    if source_target_match {
        score += 0.42;
        reasons.push("source_target_match");
    } else if ctx.source_target_override.is_some() {
        blocked.push("source_target_mismatch");
    } else {
        blocked.push("source_target_unknown");
    }

    if ctx.prompt_family_matched {
        score += 0.10;
        reasons.push("prompt_family_match");
    } else if !source_target_match {
        blocked.push("prompt_family_unknown");
    }

    if let Some(prompt_hash) = ctx.current_prompt_hash {
        if !prompt_hash.is_empty() && packet_id.contains(&format!("ph_{prompt_hash}")) {
            score += 0.08;
            reasons.push("prompt_hash_match");
        }
    }

    let strong_vector_metadata = source_label.contains("authority=human_correction")
        || source_label.contains("authority=earned_decision")
        || source_label.contains("positive=")
        || source_label.contains("answer_boundary")
        || packet_id.starts_with("lock_correction::");
    let generic_lock_vector =
        packet_id.starts_with("lock::") || source_label.starts_with("earned:");
    if strong_vector_metadata {
        score += 0.20;
        reasons.push("strong_vector_metadata");
    } else if generic_lock_vector {
        score += 0.04;
        blocked.push("generic_lock_vector");
    } else {
        blocked.push("weak_vector_metadata");
    }

    if packet.payload_key.is_some() {
        score += 0.04;
        reasons.push("payload_key_present");
    }

    if ctx.nearest_ghost_present && ctx.route_margin.is_finite() {
        if ctx.route_margin >= 0.05 {
            score += 0.14;
            reasons.push("strong_route_margin");
        } else if ctx.route_margin >= 0.01 {
            score += 0.08;
            reasons.push("usable_route_margin");
        } else {
            blocked.push("route_margin_weak");
        }
    } else {
        blocked.push("route_unknown");
    }

    if ctx.target_distance.is_finite() {
        if ctx.target_distance <= 0.25 {
            score += 0.18;
            reasons.push("near_packet_target");
        } else if ctx.target_distance <= 0.45 {
            score += 0.10;
            reasons.push("usable_packet_target_distance");
        } else {
            blocked.push("packet_target_distance_weak");
        }
    } else {
        blocked.push("packet_target_distance_unknown");
    }

    let score = score.clamp(0.0, 1.0);
    let allowed = score >= PACKET_AUTHORITY_ALLOW_THRESHOLD
        && source_target_match
        && strong_vector_metadata
        && ctx.nearest_ghost_present
        && ctx.target_distance.is_finite()
        && ctx.target_distance <= 0.45;

    if !allowed {
        blocked.push("score_below_authority_threshold");
    }

    PacketAuthorityDecision {
        score,
        allowed,
        reason: if reasons.is_empty() {
            "no_positive_authority_evidence".to_string()
        } else {
            reasons.join("+")
        },
        blocked_reason: if allowed {
            "none".to_string()
        } else {
            blocked.join("+")
        },
    }
}

/// Read-only-at-runtime store of correction packets indexed by `vq_code`.
#[derive(Debug, Clone, Default)]
pub struct CorrectionPacketStore {
    by_vq_code: HashMap<u8, Vec<CorrectionPacket>>,
    total: usize,
    /// Packet ids that were inserted into this store *during the live session*
    /// via `insert_live` (RC5: same-process mint->insert->fire loop), as opposed
    /// to loaded from disk at startup. Lets telemetry distinguish a fire driven by
    /// a correction minted THIS session from one carried in from a prior process,
    /// without widening the per-packet struct (which would ripple through every
    /// test constructor). A live-minted packet is otherwise byte-identical to its
    /// on-disk form, so firing/decay logic treats it exactly like a loaded packet.
    live_minted_ids: HashSet<String>,
    /// RC1 outcome feedback: per-packet effectiveness EMA (packet_id -> value in
    /// [-1, 1]). Absent = never measured. Folded each step from the next-step
    /// target-distance delta after a fire; positive = the correction moved the probe
    /// toward its target. Scales applied force magnitude, turning blind fire_count
    /// decay into measured error reduction.
    effectiveness_ema: HashMap<String, f32>,
    /// RC1: packets fired last step awaiting a post-force distance measurement.
    /// packet_id -> (pre-fire probe->target distance, step armed).
    pending_outcome: HashMap<String, (f32, u64)>,
    /// RC2: per-packet within-block residual SHAPE (unit-normalized 4096D vector
    /// capturing the structure bucket-mean discards). When present, the force
    /// projection rotates the flat 64-block smear toward this shape so the steering is
    /// no longer piecewise-constant. Populated for live-minted packets this session;
    /// cross-process persistence (JSONL) is a follow-up.
    residual_shape: HashMap<String, Vec<f32>>,
}

impl CorrectionPacketStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load packets from a JSONL file. Each line is a `CorrectionPacketJson` record.
    /// Empty lines and lines starting with `#` are skipped. Bad records are reported
    /// via the returned error and abort the whole load.
    pub fn load_from_jsonl(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let body = std::fs::read_to_string(path)?;
        Self::from_jsonl_str(&body)
    }

    /// Parse a JSONL string. Mirrors `load_from_jsonl` for in-process tests.
    pub fn from_jsonl_str(body: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut store = Self::new();
        for (idx, line) in body.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let raw: CorrectionPacketJson = serde_json::from_str(trimmed)
                .map_err(|e| format!("correction packet line {}: {}", idx + 1, e))?;
            let packet = CorrectionPacket::from_json(raw)?;
            store.insert(packet);
        }
        Ok(store)
    }

    pub fn insert(&mut self, packet: CorrectionPacket) {
        self.by_vq_code
            .entry(packet.vq_code)
            .or_default()
            .push(packet);
        self.total += 1;
    }

    /// RC5: insert a packet minted during the live session so it can fire on a
    /// LATER step of the SAME process. Identical to `insert` except the packet's
    /// id is also recorded in `live_minted_ids` so telemetry can attribute fires
    /// to this-session corrections. The packet itself must already be the exact
    /// `CorrectionPacket` a future process would load from the out-file (build it
    /// via `CorrectionPacket::from_json_value` on the same record the writer
    /// emitted) so same-session firing and next-session firing are byte-identical.
    pub fn insert_live(&mut self, packet: CorrectionPacket) {
        self.live_minted_ids.insert(packet.packet_id.clone());
        self.insert(packet);
    }

    /// Of the supplied fired packet ids, how many correspond to packets that were
    /// live-minted into this store this session. This is the load-bearing proof
    /// field for the RC5 closed loop: a nonzero value on a later turn means a
    /// correction minted earlier in the same process actually steered the model.
    pub fn count_live_minted_fired(&self, packet_ids: &[String]) -> usize {
        packet_ids
            .iter()
            .filter(|id| self.live_minted_ids.contains(*id))
            .count()
    }

    /// RC1: look up a packet's 64D target by id (copied out). Used by
    /// `settle_outcomes` to recompute post-force distance.
    fn target_for_id(&self, packet_id: &str) -> Option<[f32; 64]> {
        for packets in self.by_vq_code.values() {
            for p in packets {
                if p.packet_id == packet_id {
                    return Some(p.target_z_64d);
                }
            }
        }
        None
    }

    /// RC1: arm a fired packet for next-step outcome measurement. `pre_distance` is
    /// the probe->target distance at fire time (before this step's force lands).
    pub fn arm_outcome(&mut self, packet_id: &str, pre_distance: f32, step: u64) {
        if pre_distance.is_finite() {
            self.pending_outcome
                .insert(packet_id.to_string(), (pre_distance, step));
        }
    }

    /// RC1: settle all armed outcomes against the current probe — which IS the
    /// post-force hidden state of the step the fires were armed on, so no synchronous
    /// re-read is needed. For each armed packet, `delta = ((pre - post)/pre)` clamped
    /// to [-1, 1] (positive = the correction moved the probe toward its target). Fold
    /// into the packet's effectiveness EMA with smoothing `alpha`. Clears pending.
    /// Returns the number of outcomes settled.
    pub fn settle_outcomes(&mut self, probe_z: &[f32; 64], alpha: f32) -> usize {
        if self.pending_outcome.is_empty() {
            return 0;
        }
        let pending: Vec<(String, (f32, u64))> = self.pending_outcome.drain().collect();
        let mut settled = 0usize;
        for (id, (pre, _step)) in pending {
            let Some(target) = self.target_for_id(&id) else {
                continue;
            };
            let mut sq = 0f32;
            for i in 0..64 {
                let d = probe_z[i] - target[i];
                sq += d * d;
            }
            let post = sq.sqrt();
            if !(pre > 1e-6) || !post.is_finite() {
                continue;
            }
            let delta = ((pre - post) / pre).clamp(-1.0, 1.0);
            let new_ema = match self.effectiveness_ema.get(&id) {
                Some(&e) => (1.0 - alpha) * e + alpha * delta,
                None => delta,
            };
            self.effectiveness_ema.insert(id, new_ema);
            settled += 1;
        }
        settled
    }

    /// RC1: current effectiveness EMA for a packet (None = never measured).
    pub fn effectiveness(&self, packet_id: &str) -> Option<f32> {
        self.effectiveness_ema.get(packet_id).copied()
    }

    /// RC2: attach a within-block residual shape to a packet (by id).
    pub fn set_residual_shape(&mut self, packet_id: &str, shape: Vec<f32>) {
        self.residual_shape.insert(packet_id.to_string(), shape);
    }

    /// RC2: the residual shape for a packet, if one was captured this session.
    pub fn residual_shape(&self, packet_id: &str) -> Option<&[f32]> {
        self.residual_shape.get(packet_id).map(|v| v.as_slice())
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Look up all packets keyed to this vq_code. Returns &[] when the bucket is empty.
    pub fn packets_for_code(&self, vq_code: u8) -> &[CorrectionPacket] {
        self.by_vq_code
            .get(&vq_code)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Run all packets keyed to `vq_code` against `probe_z`. Returns a `Vec` of
    /// (packet, 64D delta, effective_pull_used) for every packet that fires (distance >
    /// threshold). When `decay_rate` is `Some(r)` with `0 < r < 1`, each packet's
    /// effective pull is scaled by `r ^ fire_count` BEFORE the fire is recorded — so a
    /// packet that has already fired N times produces a smaller delta than a fresh one.
    /// `decay_rate=None` or `>=1.0` disables decay (legacy behavior).
    pub fn forward_with_decay(
        &self,
        vq_code: u8,
        probe_z: &[f32; 64],
        decay_rate: Option<f32>,
    ) -> Vec<(&CorrectionPacket, [f32; 64], f32)> {
        let mut firings = Vec::new();
        for packet in self.packets_for_code(vq_code) {
            let effective_pull = match decay_rate {
                Some(r) if r > 0.0 && r < 1.0 => packet.effective_pull(r),
                _ => packet.pull_strength,
            };
            if let Some(delta) = packet.forward_with_pull(probe_z, effective_pull) {
                firings.push((packet, delta, effective_pull));
            }
        }
        firings
    }

    /// Backwards-compatible: `forward_with_decay(vq_code, probe_z, None)` — no decay.
    pub fn forward(&self, vq_code: u8, probe_z: &[f32; 64]) -> Vec<(&CorrectionPacket, [f32; 64])> {
        self.forward_with_decay(vq_code, probe_z, None)
            .into_iter()
            .map(|(p, d, _)| (p, d))
            .collect()
    }

    /// Record fires after the runtime has committed to applying force. This keeps
    /// packet_shadow/top-K discarded candidates from consuming decay budget.
    pub fn record_fires_by_id(&self, packet_ids: &[String], current_step: u64) -> usize {
        let mut count = 0usize;
        for packets in self.by_vq_code.values() {
            for packet in packets {
                if packet_ids.iter().any(|id| id == &packet.packet_id) {
                    packet.record_fire(current_step);
                    count += 1;
                }
            }
        }
        count
    }

    /// Mark every packet whose `packet_id` contains `::lh_<lh_hash>::` as invalidated.
    /// Used by the LOCK-contradiction handler: when the user contradicts a prior LOCK,
    /// pre-existing packets minted from the contradicted payload's hash get switched
    /// off so they no longer fire. Returns the count of packets newly invalidated
    /// (already-invalidated matches do not count again).
    pub fn invalidate_by_lh_hash(&self, lh_hash: &str) -> usize {
        let needle = format!("::lh_{lh_hash}::");
        let mut count = 0usize;
        for packets in self.by_vq_code.values() {
            for packet in packets {
                if packet.packet_id.contains(&needle) && !packet.invalidated.load() {
                    packet.invalidate();
                    count += 1;
                }
            }
        }
        count
    }

    /// Reverse of `invalidate_by_lh_hash`: clear the invalidated flag on every
    /// packet whose `packet_id` contains `::lh_<lh_hash>::`. Used when the user
    /// re-affirms a previously contradicted LOCK by emitting it again — the prior
    /// invalidation rolls back. Only the exact-hash form is revalidated (the
    /// semantic-key form is intentionally NOT auto-rolled-back to avoid ping-pong
    /// behaviour when the user oscillates between values for the same key).
    /// Returns count of packets newly revalidated (already-active matches do not
    /// count again).
    pub fn revalidate_by_lh_hash(&self, lh_hash: &str) -> usize {
        let needle = format!("::lh_{lh_hash}::");
        let mut count = 0usize;
        for packets in self.by_vq_code.values() {
            for packet in packets {
                if packet.packet_id.contains(&needle) && packet.invalidated.load() {
                    packet.revalidate();
                    count += 1;
                }
            }
        }
        count
    }

    /// Mark every packet whose `payload_key == Some(key)` as invalidated. Used for
    /// semantic invalidation: when the user contradicts a `final=ship_thursday`
    /// LOCK, this catches any packet whose key is `final` regardless of the value
    /// (`final=anything`). Pairs with `invalidate_by_lh_hash` to switch off both the
    /// exact-string match (iter 13) and the semantic match (iter 14). Returns the
    /// count of newly invalidated packets.
    pub fn invalidate_by_payload_key(&self, key: &str) -> usize {
        if key.is_empty() {
            return 0;
        }
        let mut count = 0usize;
        for packets in self.by_vq_code.values() {
            for packet in packets {
                if packet.invalidated.load() {
                    continue;
                }
                if packet
                    .payload_key
                    .as_deref()
                    .map(|k| k.eq_ignore_ascii_case(key))
                    .unwrap_or(false)
                {
                    packet.invalidate();
                    count += 1;
                }
            }
        }
        count
    }

    /// Iterate over every packet in the store, regardless of bucket. Useful for
    /// cross-cutting operations like global invalidation queries.
    pub fn iter_packets(&self) -> impl Iterator<Item = &CorrectionPacket> {
        self.by_vq_code.values().flat_map(|v| v.iter())
    }

    /// Evict packets whose effective_pull has decayed below
    /// `floor_ratio × pull_strength`. Earned packets (decay_rate=Some(1.0)) never
    /// qualify since their effective_pull stays at full pull_strength regardless
    /// of fire_count. Invalidated packets are also removed if their decayed pull
    /// falls below the floor — they're already inactive and just take up space.
    /// Returns the count of packets evicted.
    ///
    /// Use cases:
    /// - Long-running sessions where scaffolding packets fire many times and
    ///   their pull decays toward zero. Evicting them reduces lookup cost and
    ///   JSONL size without affecting earned-answer preservation.
    /// - State-out cleanup before persistence: evict, then write_to_jsonl.
    ///
    /// `floor_ratio = 0.0` evicts only literally-zero-pull packets (essentially
    /// a noop unless the resolved decay collapses to zero). Reasonable values
    /// are 0.01–0.1 (1–10% of original pull).
    pub fn evict_below_floor(&mut self, engine_decay_rate: f32, floor_ratio: f32) -> usize {
        if floor_ratio <= 0.0 || !floor_ratio.is_finite() {
            return 0;
        }
        let mut evicted = 0usize;
        let mut dropped_total = 0usize;
        self.by_vq_code.retain(|_, packets| {
            let before = packets.len();
            packets.retain(|packet| {
                if packet.pull_strength <= 0.0 {
                    return true; // Don't evict zero-strength packets (could be intentional).
                }
                let ep = packet.effective_pull(engine_decay_rate);
                let ratio = ep / packet.pull_strength;
                let keep = ratio >= floor_ratio;
                if !keep {
                    dropped_total += 1;
                }
                keep
            });
            evicted += before - packets.len();
            !packets.is_empty()
        });
        self.total = self.total.saturating_sub(dropped_total);
        evicted
    }

    /// Serialize the entire store to a JSONL file (overwrite, not append). Each line is
    /// a `CorrectionPacket` with current `fire_count` and `last_fire_step` counters
    /// captured atomically. Used to persist decay/unfold state across process boundaries.
    /// On restart, `load_from_jsonl` rehydrates the counters and the next session resumes
    /// the same effective_pull.
    pub fn write_to_jsonl(&self, path: &Path) -> std::io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let tmp_path = path.with_extension("jsonl.tmp");
        let mut written = 0usize;
        {
            let mut file = std::fs::File::create(&tmp_path)?;
            for (_, packets) in self.by_vq_code.iter() {
                for packet in packets {
                    let target_vec: Vec<f32> = packet.target_z_64d.to_vec();
                    let mut record = serde_json::json!({
                        "packet_id": &packet.packet_id,
                        "vq_code": packet.vq_code,
                        "target_z_64d": target_vec,
                        "pull_strength": packet.pull_strength,
                        "distance_threshold": packet.distance_threshold,
                        "source_label": &packet.source_label,
                        "created_step": packet.created_step,
                        "fire_count": packet.fire_count.load(),
                        "last_fire_step": packet.last_fire_step.load(),
                    });
                    if let Some(payload) = packet.payload_z_64d.as_ref() {
                        record["payload_z_64d"] = serde_json::Value::from(payload.to_vec());
                    }
                    if let Some(decay_rate) = packet.decay_rate {
                        record["decay_rate"] = serde_json::Value::from(decay_rate);
                    }
                    if let Some(unfold_factor) = packet.unfold_factor {
                        record["unfold_factor"] = serde_json::Value::from(unfold_factor);
                    }
                    if packet.invalidated.load() {
                        record["invalidated"] = serde_json::Value::from(true);
                    }
                    if let Some(key) = packet.payload_key.as_ref() {
                        record["payload_key"] = serde_json::Value::from(key.clone());
                    }
                    if let Some(retry_factor) = packet.unfold_retry_factor {
                        record["unfold_retry_factor"] = serde_json::Value::from(retry_factor);
                    }
                    if let Some(v) = packet.text_fact.as_ref() {
                        record["text_fact"] = serde_json::Value::from(v.clone());
                    }
                    if let Some(v) = packet.route_code.as_ref() {
                        record["route_code"] = serde_json::Value::from(v.clone());
                    }
                    if let Some(v) = packet.route_motif_id.as_ref() {
                        record["route_motif_id"] = serde_json::Value::from(v.clone());
                    }
                    if let Some(v) = packet.target_ghost_id.as_ref() {
                        record["target_ghost_id"] = serde_json::Value::from(v.clone());
                    }
                    if let Some(v) = packet.nearest_ghost_distance {
                        record["nearest_ghost_distance"] = serde_json::Value::from(v);
                    }
                    if let Some(v) = packet.second_nearest_ghost_distance {
                        record["second_nearest_ghost_distance"] = serde_json::Value::from(v);
                    }
                    if let Some(v) = packet.route_margin {
                        record["route_margin"] = serde_json::Value::from(v);
                    }
                    if let Some(v) = packet.agency_transition.as_ref() {
                        record["agency_transition"] = serde_json::Value::from(v.clone());
                    }
                    if let Some(v) = packet.force_policy.as_ref() {
                        record["force_policy"] = serde_json::Value::from(v.clone());
                    }
                    if let Some(v) = packet.force_pull_strength {
                        record["force_pull_strength"] = serde_json::Value::from(v);
                    }
                    if let Some(v) = packet.force_distance_threshold {
                        record["force_distance_threshold"] = serde_json::Value::from(v);
                    }
                    if let Some(v) = packet.force_decay_rate {
                        record["force_decay_rate"] = serde_json::Value::from(v);
                    }
                    if let Some(v) = packet.force_unfold_factor {
                        record["force_unfold_factor"] = serde_json::Value::from(v);
                    }
                    if let Some(v) = packet.force_unfold_retry_factor {
                        record["force_unfold_retry_factor"] = serde_json::Value::from(v);
                    }
                    if let Some(v) = packet.answer_lock_boundary.as_ref() {
                        record["answer_lock_boundary"] = serde_json::Value::from(v.clone());
                    }
                    if let Some(v) = packet.projection_strategy.as_ref() {
                        record["projection_strategy"] = serde_json::Value::from(v.clone());
                    }
                    if let Some(v) = packet.ghost_pull_delta_norm {
                        record["ghost_pull_delta_norm"] = serde_json::Value::from(v);
                    }
                    writeln!(file, "{}", record)?;
                    written += 1;
                }
            }
            file.sync_all()?;
        }
        // Atomic rename so the destination is never partially written.
        std::fs::rename(&tmp_path, path)?;
        Ok(written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_packet(vq_code: u8, packet_id: &str) -> CorrectionPacket {
        let mut target = [0f32; 64];
        target[0] = 1.0;
        target[1] = 1.0;
        CorrectionPacket {
            packet_id: packet_id.into(),
            vq_code,
            target_z_64d: target,
            payload_z_64d: None,
            pull_strength: 0.5,
            distance_threshold: 0.05,
            source_label: "test fixture".into(),
            created_step: 0,
            text_fact: None,
            route_code: None,
            route_motif_id: None,
            target_ghost_id: None,
            nearest_ghost_distance: None,
            second_nearest_ghost_distance: None,
            route_margin: None,
            agency_transition: None,
            force_policy: None,
            force_pull_strength: None,
            force_distance_threshold: None,
            force_decay_rate: None,
            force_unfold_factor: None,
            force_unfold_retry_factor: None,
            answer_lock_boundary: None,
            projection_strategy: None,
            ghost_pull_delta_norm: None,
            fire_count: AtomicU64Wrapper::new(0),
            last_fire_step: AtomicU64Wrapper::new(0),
            decay_rate: None,
            unfold_factor: None,
            invalidated: AtomicBoolWrapper::new(false),
            payload_key: None,
            unfold_retry_factor: None,
        }
    }

    #[test]
    fn packet_authority_allows_structured_source_target_packet() {
        let mut packet = fixture_packet(
            7,
            "structured_artifact_triage::target_beta_current_mechanism::ph_abcd",
        );
        packet.source_label = "structured_artifact_triage target_id=beta_current_mechanism positive=current_mechanism_repair authority=human_correction".to_string();
        packet.payload_key = Some("artifact_triage_trajectory".to_string());

        let decision = decide_packet_authority(
            &packet,
            PacketAuthorityContext {
                source_target_override: Some("beta_current_mechanism"),
                prompt_family_matched: true,
                current_prompt_hash: Some("abcd"),
                route_margin: 0.06,
                nearest_ghost_present: true,
                target_distance: 0.18,
            },
        );

        assert!(decision.allowed, "{decision:?}");
        assert!(decision.score >= PACKET_AUTHORITY_ALLOW_THRESHOLD);
        assert_eq!(decision.blocked_reason, "none");
        assert!(decision.reason.contains("source_target_match"));
        assert!(decision.reason.contains("strong_vector_metadata"));
    }

    #[test]
    fn packet_authority_blocks_generic_lock_vector_without_route_context() {
        let mut packet = fixture_packet(7, "lock::req_x::ph_abcd::lh_1234::step_00001");
        packet.source_label = "earned: final=use_the_current_mechanism".to_string();
        packet.payload_key = Some("final".to_string());

        let decision = decide_packet_authority(
            &packet,
            PacketAuthorityContext {
                source_target_override: None,
                prompt_family_matched: false,
                current_prompt_hash: Some("wxyz"),
                route_margin: 0.0,
                nearest_ghost_present: false,
                target_distance: 0.70,
            },
        );

        assert!(!decision.allowed);
        assert!(decision.score < PACKET_AUTHORITY_ALLOW_THRESHOLD);
        assert!(decision.blocked_reason.contains("source_target_unknown"));
        assert!(decision.blocked_reason.contains("generic_lock_vector"));
        assert!(decision.blocked_reason.contains("route_unknown"));
    }

    #[test]
    fn forward_returns_none_within_threshold() {
        let packet = fixture_packet(7, "p1");
        // probe ~ target → distance below threshold → None
        let mut probe = packet.target_z_64d;
        probe[0] += 0.01;
        let out = packet.forward(&probe);
        assert!(
            out.is_none(),
            "expected no fire when probe within threshold"
        );
    }

    #[test]
    fn forward_returns_normalized_pull_when_far() {
        let packet = fixture_packet(7, "p1");
        let probe = [0f32; 64];
        let delta = packet.forward(&probe).expect("should fire");
        let mag: f32 = delta.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (mag - 0.5).abs() < 1e-5,
            "delta magnitude should equal pull_strength=0.5, got {}",
            mag
        );
        // Direction points toward target (positive on dims 0, 1).
        assert!(delta[0] > 0.0);
        assert!(delta[1] > 0.0);
    }

    #[test]
    fn payload_blend_zero_preserves_legacy_direction() {
        let mut packet = fixture_packet(7, "p1");
        let mut payload = [0f32; 64];
        payload[2] = 1.0;
        packet.payload_z_64d = Some(payload);
        let probe = [0f32; 64];

        let legacy = packet.forward_with_pull(&probe, 0.5).expect("legacy");
        let blended = packet
            .forward_with_payload_blend(&probe, 0.5, 0.0)
            .expect("blend disabled");

        for i in 0..64 {
            assert!(
                (legacy[i] - blended[i]).abs() < 1e-6,
                "dim {i}: legacy {} != blended {}",
                legacy[i],
                blended[i]
            );
        }
    }

    #[test]
    fn payload_blend_adds_orthogonal_component_without_changing_pull_budget() {
        let mut packet = fixture_packet(7, "p1");
        packet.target_z_64d = [0f32; 64];
        packet.target_z_64d[0] = 1.0;
        let mut payload = [0f32; 64];
        payload[0] = 1.0;
        payload[1] = 1.0;
        packet.payload_z_64d = Some(payload);

        let probe = [0f32; 64];
        let legacy = packet.forward_with_pull(&probe, 0.5).expect("legacy");
        let blended = packet
            .forward_with_payload_blend(&probe, 0.5, 0.5)
            .expect("payload blend");
        let blended_mag: f32 = blended.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!((legacy[0] - 0.5).abs() < 1e-6);
        assert!(legacy[1].abs() < 1e-6);
        assert!(blended[0] > 0.0);
        assert!(blended[0] < legacy[0]);
        assert!(blended[1] > 0.0, "orthogonal payload should add dim 1");
        assert!(
            (blended_mag - 0.5).abs() < 1e-6,
            "blend should preserve effective pull norm, got {blended_mag}"
        );
    }

    #[test]
    fn store_roundtrip_jsonl() {
        let body = r#"
{"packet_id":"a","vq_code":7,"target_z_64d":[1.0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"pull_strength":0.5,"distance_threshold":0.05,"source_label":"unit test","created_step":1}
# comment line
{"packet_id":"b","vq_code":42,"target_z_64d":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"pull_strength":0.1,"distance_threshold":1.0,"source_label":"unit test 2","created_step":2}
"#;
        let store = CorrectionPacketStore::from_jsonl_str(body).expect("parse");
        assert_eq!(store.total(), 2);
        assert_eq!(store.packets_for_code(7).len(), 1);
        assert_eq!(store.packets_for_code(42).len(), 1);
        assert_eq!(store.packets_for_code(99).len(), 0);
    }

    #[test]
    fn payload_z_64d_loads_and_persists_through_jsonl() {
        let target = vec![0.0f32; 64];
        let mut payload = vec![0.0f32; 64];
        payload[3] = 0.75;
        payload[9] = -0.25;
        let line = serde_json::json!({
            "packet_id": "payload_packet",
            "vq_code": 9,
            "target_z_64d": target,
            "payload_z_64d": payload,
            "pull_strength": 0.2,
            "distance_threshold": 0.05,
        })
        .to_string();

        let store = CorrectionPacketStore::from_jsonl_str(&line).expect("load");
        let packet = &store.packets_for_code(9)[0];
        let loaded_payload = packet.payload_z_64d.expect("payload vector");
        assert!((loaded_payload[3] - 0.75).abs() < 1e-6);
        assert!((loaded_payload[9] + 0.25).abs() < 1e-6);

        let path =
            std::env::temp_dir().join(format!("payload_z_roundtrip_{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        store.write_to_jsonl(&path).expect("write");
        let reloaded = CorrectionPacketStore::load_from_jsonl(&path).expect("reload");
        let reloaded_payload = reloaded.packets_for_code(9)[0]
            .payload_z_64d
            .expect("reloaded payload vector");
        assert!((reloaded_payload[3] - 0.75).abs() < 1e-6);
        assert!((reloaded_payload[9] + 0.25).abs() < 1e-6);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn store_forward_fires_only_for_matching_vq_code() {
        let mut store = CorrectionPacketStore::new();
        store.insert(fixture_packet(7, "p1"));
        store.insert(fixture_packet(42, "p2"));
        let probe = [0f32; 64];

        let firings_7 = store.forward(7, &probe);
        assert_eq!(firings_7.len(), 1);
        assert_eq!(firings_7[0].0.packet_id, "p1");

        let firings_42 = store.forward(42, &probe);
        assert_eq!(firings_42.len(), 1);
        assert_eq!(firings_42[0].0.packet_id, "p2");

        let firings_99 = store.forward(99, &probe);
        assert!(firings_99.is_empty());
    }

    #[test]
    fn store_rejects_wrong_dim_target() {
        let bad = r#"{"packet_id":"x","vq_code":0,"target_z_64d":[1.0,2.0,3.0],"pull_strength":0.5,"distance_threshold":0.05}"#;
        assert!(CorrectionPacketStore::from_jsonl_str(bad).is_err());
    }

    #[test]
    fn invalidated_packet_does_not_fire() {
        let packet = fixture_packet(7, "p1");
        let probe = [0f32; 64];
        // Sanity: fires before invalidation.
        assert!(packet.forward(&probe).is_some());
        packet.invalidate();
        assert!(packet.forward(&probe).is_none());
        assert!(packet.forward_with_pull(&probe, 1.0).is_none());
        assert!(packet.invalidated.load());
    }

    #[test]
    fn evict_below_floor_removes_decayed_scaffolding_keeps_earned() {
        let mut store = CorrectionPacketStore::new();
        // Earned packet with decay=1.0 → effective_pull stays at pull_strength
        // regardless of fire_count. Should NEVER be evicted.
        let mut earned = fixture_packet(7, "earned");
        earned.decay_rate = Some(1.0);
        // Many fires
        for _ in 0..50 {
            earned.record_fire(0);
        }
        // Scaffolding with decay=0.5, fire_count=10 → effective_pull =
        // pull × 0.5^10 ≈ pull × 0.000977 = ~0.001 if pull=0.5.
        // Ratio ~ 0.000977.
        let mut scaffolding = fixture_packet(11, "scaffolding");
        scaffolding.decay_rate = Some(0.5);
        for _ in 0..10 {
            scaffolding.record_fire(0);
        }
        // Fresh scaffolding with no fires → effective_pull = pull_strength;
        // ratio = 1.0. Should not be evicted at any reasonable floor.
        let fresh = fixture_packet(13, "fresh");
        store.insert(earned);
        store.insert(scaffolding);
        store.insert(fresh);
        assert_eq!(store.total(), 3);

        // Evict at floor 0.05 (5% of original pull). The 0.001 scaffolding goes;
        // earned (ratio=1.0) and fresh (ratio=1.0) stay.
        let evicted = store.evict_below_floor(0.5, 0.05);
        assert_eq!(evicted, 1);
        assert_eq!(store.total(), 2);
        // Verify the right packet was removed.
        let remaining: Vec<String> = store.iter_packets().map(|p| p.packet_id.clone()).collect();
        assert!(remaining.contains(&"earned".to_string()));
        assert!(remaining.contains(&"fresh".to_string()));
        assert!(!remaining.contains(&"scaffolding".to_string()));

        // floor=0 is a no-op.
        assert_eq!(store.evict_below_floor(0.5, 0.0), 0);
        assert_eq!(store.total(), 2);

        // floor=2.0 (impossible — no packet has ratio above 1.0) evicts both.
        let evicted_all = store.evict_below_floor(0.5, 2.0);
        assert_eq!(evicted_all, 2);
        assert_eq!(store.total(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn revalidate_by_lh_hash_undoes_invalidation() {
        let mut store = CorrectionPacketStore::new();
        let p1 = CorrectionPacket {
            packet_id: "lock::req_a::ph_x::lh_abcd::step_00001".into(),
            ..fixture_packet(7, "_unused")
        };
        let p2 = CorrectionPacket {
            packet_id: "lock::req_b::ph_x::lh_abcd::step_00002".into(),
            ..fixture_packet(7, "_unused")
        };
        let p3 = CorrectionPacket {
            packet_id: "lock::req_c::ph_x::lh_efgh::step_00003".into(),
            ..fixture_packet(7, "_unused")
        };
        store.insert(p1);
        store.insert(p2);
        store.insert(p3);

        // Round 1: invalidate two packets sharing lh_abcd.
        assert_eq!(store.invalidate_by_lh_hash("abcd"), 2);
        assert_eq!(store.invalidate_by_lh_hash("efgh"), 1);
        // Round 2: revalidate just the abcd ones.
        let revalidated = store.revalidate_by_lh_hash("abcd");
        assert_eq!(revalidated, 2);
        // Repeat: already active.
        assert_eq!(store.revalidate_by_lh_hash("abcd"), 0);

        // efgh is still invalidated (revalidation is not contagious).
        let bucket = store.packets_for_code(7);
        let efgh = bucket
            .iter()
            .find(|p| p.packet_id.contains("lh_efgh"))
            .unwrap();
        assert!(efgh.invalidated.load());

        // The abcd packets actually fire again now.
        let probe = [0f32; 64];
        for p in bucket.iter().filter(|p| p.packet_id.contains("lh_abcd")) {
            assert!(
                p.forward(&probe).is_some(),
                "revalidated packet should fire"
            );
        }
    }

    #[test]
    fn invalidate_by_payload_key_targets_matching_packets() {
        let mut store = CorrectionPacketStore::new();
        let p1 = CorrectionPacket {
            packet_id: "lock::req_a::ph_xx::lh_aa::step_00001".into(),
            payload_key: Some("final".into()),
            ..fixture_packet(7, "_unused")
        };
        let p2 = CorrectionPacket {
            packet_id: "lock_correction::req_b::ph_xx::lh_bb::step_00002".into(),
            payload_key: Some("final".into()),
            ..fixture_packet(7, "_unused")
        };
        let p3 = CorrectionPacket {
            packet_id: "lock::req_c::ph_xx::lh_cc::step_00003".into(),
            payload_key: Some("priority".into()),
            ..fixture_packet(7, "_unused")
        };
        let p4 = CorrectionPacket {
            packet_id: "remember::req_d::ph_xx::rh_dd::step_00004".into(),
            payload_key: None, // No key — should not match.
            ..fixture_packet(7, "_unused")
        };
        store.insert(p1);
        store.insert(p2);
        store.insert(p3);
        store.insert(p4);

        let invalidated = store.invalidate_by_payload_key("final");
        assert_eq!(invalidated, 2, "two packets share key=final");
        // priority and None remain active.
        let bucket = store.packets_for_code(7);
        let active: Vec<&CorrectionPacket> =
            bucket.iter().filter(|p| !p.invalidated.load()).collect();
        assert_eq!(active.len(), 2);

        // Case-insensitive match.
        let again = store.invalidate_by_payload_key("FINAL");
        assert_eq!(again, 0, "already invalidated");

        // Empty key invalidates nothing.
        let empty = store.invalidate_by_payload_key("");
        assert_eq!(empty, 0);

        let priority_inv = store.invalidate_by_payload_key("priority");
        assert_eq!(priority_inv, 1);
    }

    #[test]
    fn write_to_jsonl_persists_payload_key_when_set() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "payload_key_persist_test_{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);

        let mut store = CorrectionPacketStore::new();
        let with_key = CorrectionPacket {
            payload_key: Some("final".into()),
            ..fixture_packet(7, "with_key")
        };
        let without_key = fixture_packet(7, "without_key"); // payload_key = None
        store.insert(with_key);
        store.insert(without_key);
        store.write_to_jsonl(&path).unwrap();

        let body = std::fs::read_to_string(&path).unwrap();
        let with_line = body
            .lines()
            .find(|l| l.contains(r#""packet_id":"with_key""#))
            .unwrap();
        let without_line = body
            .lines()
            .find(|l| l.contains(r#""packet_id":"without_key""#))
            .unwrap();
        assert!(with_line.contains(r#""payload_key":"final""#));
        assert!(!without_line.contains("payload_key"));

        let reloaded = CorrectionPacketStore::load_from_jsonl(&path).unwrap();
        let bucket = reloaded.packets_for_code(7);
        let with = bucket.iter().find(|p| p.packet_id == "with_key").unwrap();
        let without = bucket
            .iter()
            .find(|p| p.packet_id == "without_key")
            .unwrap();
        assert_eq!(with.payload_key.as_deref(), Some("final"));
        assert!(without.payload_key.is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn invalidate_by_lh_hash_targets_matching_packets() {
        let mut store = CorrectionPacketStore::new();
        let p1 = CorrectionPacket {
            packet_id: "lock::req_a::ph_111::lh_aaaa::step_00001".into(),
            ..fixture_packet(7, "_unused")
        };
        let p2 = CorrectionPacket {
            packet_id: "lock::req_b::ph_222::lh_aaaa::step_00002".into(),
            ..fixture_packet(7, "_unused")
        };
        let p3 = CorrectionPacket {
            packet_id: "lock::req_c::ph_333::lh_bbbb::step_00003".into(),
            ..fixture_packet(7, "_unused")
        };
        store.insert(p1);
        store.insert(p2);
        store.insert(p3);

        let invalidated = store.invalidate_by_lh_hash("aaaa");
        assert_eq!(invalidated, 2, "two packets share lh_aaaa");
        let bucket = store.packets_for_code(7);
        let aaaa: Vec<_> = bucket
            .iter()
            .filter(|p| p.packet_id.contains("::lh_aaaa::"))
            .collect();
        assert!(aaaa.iter().all(|p| p.invalidated.load()));
        let bbbb: Vec<_> = bucket
            .iter()
            .filter(|p| p.packet_id.contains("::lh_bbbb::"))
            .collect();
        assert!(bbbb.iter().all(|p| !p.invalidated.load()));

        // Repeat call returns 0 because already invalidated.
        let again = store.invalidate_by_lh_hash("aaaa");
        assert_eq!(again, 0);
    }

    #[test]
    fn write_to_jsonl_persists_invalidated_flag() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "invalidate_persist_test_{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);

        let mut store = CorrectionPacketStore::new();
        let p1 = CorrectionPacket {
            packet_id: "lock::req_a::ph_111::lh_aaaa::step_00001".into(),
            ..fixture_packet(7, "_unused")
        };
        p1.invalidate();
        let p2 = fixture_packet(7, "active");
        store.insert(p1);
        store.insert(p2);
        store.write_to_jsonl(&path).unwrap();

        let body = std::fs::read_to_string(&path).unwrap();
        let invalidated_line = body
            .lines()
            .find(|l| l.contains("lh_aaaa"))
            .expect("invalidated packet line");
        let active_line = body
            .lines()
            .find(|l| l.contains(r#""packet_id":"active""#))
            .expect("active packet line");
        assert!(invalidated_line.contains(r#""invalidated":true"#));
        assert!(!active_line.contains("invalidated"));

        let reloaded = CorrectionPacketStore::load_from_jsonl(&path).unwrap();
        let bucket = reloaded.packets_for_code(7);
        let inv = bucket
            .iter()
            .find(|p| p.packet_id.contains("lh_aaaa"))
            .unwrap();
        let act = bucket.iter().find(|p| p.packet_id == "active").unwrap();
        assert!(inv.invalidated.load());
        assert!(!act.invalidated.load());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn record_fire_updates_counters() {
        let packet = fixture_packet(7, "p1");
        packet.record_fire(42);
        packet.record_fire(43);
        assert_eq!(packet.fire_count.load(), 2);
        assert_eq!(packet.last_fire_step.load(), 43);
    }

    #[test]
    fn effective_pull_decays_with_fire_count() {
        let packet = fixture_packet(7, "p1");
        // Decay rate 0.5: each fire halves the pull.
        assert!((packet.effective_pull(0.5) - 0.5).abs() < 1e-6);
        packet.record_fire(1);
        assert!((packet.effective_pull(0.5) - 0.25).abs() < 1e-6);
        packet.record_fire(2);
        assert!((packet.effective_pull(0.5) - 0.125).abs() < 1e-6);
    }

    #[test]
    fn per_packet_decay_rate_overrides_engine_global() {
        let mut packet = fixture_packet(7, "earned");
        packet.decay_rate = Some(1.0); // No-decay override (earned answer)
        for _ in 0..10 {
            packet.record_fire(0);
        }
        // Engine wants aggressive decay, but the packet's own override wins.
        assert!((packet.effective_pull(0.5) - packet.pull_strength).abs() < 1e-6);

        let mut packet_slow = fixture_packet(7, "slow");
        packet_slow.decay_rate = Some(0.99); // Custom slow decay
        packet_slow.record_fire(0);
        // Slow per-packet override is independent of the aggressive engine global.
        let slow_pull = packet_slow.effective_pull(0.5);
        assert!(slow_pull > packet_slow.pull_strength * 0.99 - 1e-6);
        assert!(slow_pull < packet_slow.pull_strength);
    }

    #[test]
    fn per_packet_unfold_retry_factor_overrides_engine_global() {
        let mut earned = fixture_packet(7, "earned");
        earned.unfold_retry_factor = Some(1.0); // earned ignores retry-boost
        let mut scaffolding = fixture_packet(7, "scaffolding");
        scaffolding.unfold_retry_factor = None; // scaffolding inherits engine

        // engine_retry_factor=5.0 — say retry-relapse boosts more than OOD
        assert!((earned.effective_unfold_retry_factor(5.0) - 1.0).abs() < 1e-6);
        assert!((scaffolding.effective_unfold_retry_factor(5.0) - 5.0).abs() < 1e-6);

        // Negative override clamps to 0
        let mut weird = fixture_packet(7, "weird");
        weird.unfold_retry_factor = Some(-2.0);
        assert!((weird.effective_unfold_retry_factor(5.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn write_to_jsonl_emits_unfold_retry_factor_when_set() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "unfold_retry_factor_test_{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);

        let mut store = CorrectionPacketStore::new();
        let mut earned = fixture_packet(11, "earned");
        earned.unfold_retry_factor = Some(1.0);
        earned.unfold_factor = Some(1.0);
        earned.decay_rate = Some(1.0);
        let scaffold = fixture_packet(11, "scaffold");
        store.insert(earned);
        store.insert(scaffold);
        store.write_to_jsonl(&path).unwrap();

        let body = std::fs::read_to_string(&path).unwrap();
        let earned_line = body
            .lines()
            .find(|l| l.contains(r#""packet_id":"earned""#))
            .unwrap();
        let scaffold_line = body
            .lines()
            .find(|l| l.contains(r#""packet_id":"scaffold""#))
            .unwrap();
        assert!(earned_line.contains(r#""unfold_retry_factor":1.0"#));
        assert!(!scaffold_line.contains("unfold_retry_factor"));

        let reloaded = CorrectionPacketStore::load_from_jsonl(&path).unwrap();
        let bucket = reloaded.packets_for_code(11);
        let e = bucket.iter().find(|p| p.packet_id == "earned").unwrap();
        let s = bucket.iter().find(|p| p.packet_id == "scaffold").unwrap();
        assert_eq!(e.unfold_retry_factor, Some(1.0));
        assert!(s.unfold_retry_factor.is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn per_packet_unfold_factor_overrides_engine_global() {
        let mut earned = fixture_packet(7, "earned");
        earned.unfold_factor = Some(1.0); // earned ignores boost
        let mut scaffolding = fixture_packet(7, "scaffolding");
        scaffolding.unfold_factor = None; // scaffolding inherits engine

        // engine_global=3.0 — typical relapse boost
        assert!((earned.effective_unfold_factor(3.0) - 1.0).abs() < 1e-6);
        assert!((scaffolding.effective_unfold_factor(3.0) - 3.0).abs() < 1e-6);

        // Per-packet override clamps to >= 0
        let mut weird = fixture_packet(7, "weird");
        weird.unfold_factor = Some(-2.0);
        assert!((weird.effective_unfold_factor(3.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn write_to_jsonl_emits_unfold_factor_when_set() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("unfold_factor_test_{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let mut store = CorrectionPacketStore::new();
        let mut earned = fixture_packet(11, "earned");
        earned.unfold_factor = Some(1.0);
        earned.decay_rate = Some(1.0);
        let scaffolding = fixture_packet(11, "scaffolding"); // both None
        store.insert(earned);
        store.insert(scaffolding);
        store.write_to_jsonl(&path).unwrap();

        let body = std::fs::read_to_string(&path).unwrap();
        let earned_line = body
            .lines()
            .find(|l| l.contains(r#""packet_id":"earned""#))
            .unwrap();
        let scaffold_line = body
            .lines()
            .find(|l| l.contains(r#""packet_id":"scaffolding""#))
            .unwrap();
        assert!(
            earned_line.contains(r#""unfold_factor":1.0"#),
            "earned must serialise unfold_factor; got {earned_line}"
        );
        assert!(
            !scaffold_line.contains("unfold_factor"),
            "scaffolding (None) must omit the field; got {scaffold_line}"
        );

        let reloaded = CorrectionPacketStore::load_from_jsonl(&path).unwrap();
        let earned_reloaded = reloaded
            .packets_for_code(11)
            .iter()
            .find(|p| p.packet_id == "earned")
            .unwrap();
        let scaffold_reloaded = reloaded
            .packets_for_code(11)
            .iter()
            .find(|p| p.packet_id == "scaffolding")
            .unwrap();
        assert_eq!(earned_reloaded.unfold_factor, Some(1.0));
        assert_eq!(scaffold_reloaded.unfold_factor, None);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn write_to_jsonl_emits_decay_rate_when_set() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("decay_rate_test_{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let mut store = CorrectionPacketStore::new();
        let mut earned = fixture_packet(7, "earned");
        earned.decay_rate = Some(1.0);
        let scaffolding = fixture_packet(7, "scaffolding"); // None
        store.insert(earned);
        store.insert(scaffolding);
        store.write_to_jsonl(&path).unwrap();

        let body = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = body.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 2);
        let earned_line = lines
            .iter()
            .find(|l| l.contains(r#""packet_id":"earned""#))
            .unwrap();
        let scaffold_line = lines
            .iter()
            .find(|l| l.contains(r#""packet_id":"scaffolding""#))
            .unwrap();
        assert!(
            earned_line.contains(r#""decay_rate":1.0"#),
            "earned packet must serialise its decay_rate; got {earned_line}"
        );
        assert!(
            !scaffold_line.contains("decay_rate"),
            "scaffolding packet (decay_rate=None) must omit the field; got {scaffold_line}"
        );

        // Roundtrip preserves the override.
        let reloaded = CorrectionPacketStore::load_from_jsonl(&path).unwrap();
        let earned_reloaded = reloaded
            .packets_for_code(7)
            .iter()
            .find(|p| p.packet_id == "earned")
            .unwrap();
        let scaffold_reloaded = reloaded
            .packets_for_code(7)
            .iter()
            .find(|p| p.packet_id == "scaffolding")
            .unwrap();
        assert_eq!(earned_reloaded.decay_rate, Some(1.0));
        assert_eq!(scaffold_reloaded.decay_rate, None);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn effective_pull_no_decay_when_rate_one_or_zero() {
        let packet = fixture_packet(7, "p1");
        for _ in 0..10 {
            packet.record_fire(0);
        }
        // decay_rate >= 1.0 disables decay
        assert!((packet.effective_pull(1.0) - packet.pull_strength).abs() < 1e-6);
        assert!((packet.effective_pull(1.5) - packet.pull_strength).abs() < 1e-6);
        // decay_rate <= 0 also disables decay (out of valid range)
        assert!((packet.effective_pull(0.0) - packet.pull_strength).abs() < 1e-6);
    }

    #[test]
    fn loader_carries_persisted_fire_count() {
        let body = r#"
{"packet_id":"a","vq_code":7,"target_z_64d":[1.0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"pull_strength":0.5,"distance_threshold":0.05,"fire_count":42,"last_fire_step":99}
"#;
        let store = CorrectionPacketStore::from_jsonl_str(body).expect("parse");
        let packet = &store.packets_for_code(7)[0];
        assert_eq!(packet.fire_count.load(), 42);
        assert_eq!(packet.last_fire_step.load(), 99);
    }

    #[test]
    fn write_to_jsonl_roundtrip_preserves_fire_count() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "correction_packets_test_{}.jsonl",
            std::process::id()
        ));

        let mut store = CorrectionPacketStore::new();
        let packet = fixture_packet(7, "p1");
        packet.record_fire(11);
        packet.record_fire(22);
        store.insert(packet);

        let written = store.write_to_jsonl(&path).expect("write");
        assert_eq!(written, 1);

        let reloaded = CorrectionPacketStore::load_from_jsonl(&path).expect("read");
        let reloaded_packet = &reloaded.packets_for_code(7)[0];
        assert_eq!(reloaded_packet.packet_id, "p1");
        assert_eq!(reloaded_packet.fire_count.load(), 2);
        assert_eq!(reloaded_packet.last_fire_step.load(), 22);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn write_to_jsonl_roundtrip_preserves_hybrid_metadata_fields() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "correction_packets_hybrid_fields_test_{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);

        let mut store = CorrectionPacketStore::new();
        let mut packet = fixture_packet(7, "hybrid");
        packet.text_fact = Some("owner=Jason".into());
        packet.route_code = Some("vq_110".into());
        packet.route_motif_id =
            Some("live_hidden::raw64::002::trap_task::neutral::hinge_window".into());
        packet.target_ghost_id = Some("ba8a856d-d63b-48f4-9def-c1a9747b94a5".into());
        packet.nearest_ghost_distance = Some(0.1337);
        packet.second_nearest_ghost_distance = Some(0.1555);
        packet.route_margin = Some(0.0020872438);
        packet.agency_transition = Some("REMEMBER->LOCK".into());
        packet.force_policy = Some("lock_earned".into());
        packet.force_pull_strength = Some(0.3);
        packet.force_distance_threshold = Some(0.05);
        packet.force_decay_rate = Some(1.0);
        packet.force_unfold_factor = Some(1.0);
        packet.force_unfold_retry_factor = Some(1.0);
        packet.answer_lock_boundary = Some("lock_payload".into());
        packet.projection_strategy = Some("vq_correction_packet".into());
        packet.ghost_pull_delta_norm = Some(10.0);
        store.insert(packet);

        let written = store.write_to_jsonl(&path).expect("write");
        assert_eq!(written, 1);

        let body = std::fs::read_to_string(&path).expect("read");
        let line = body.lines().find(|l| !l.trim().is_empty()).expect("line");
        let json: serde_json::Value = serde_json::from_str(line).expect("packet json");
        let approx = |field: &str, expected: f64| {
            let got = json[field].as_f64().expect(field);
            assert!(
                (got - expected).abs() < 1e-6,
                "{field}: expected {expected}, got {got}"
            );
        };
        assert_eq!(json["text_fact"], "owner=Jason");
        assert_eq!(json["route_code"], "vq_110");
        assert_eq!(
            json["route_motif_id"],
            "live_hidden::raw64::002::trap_task::neutral::hinge_window"
        );
        assert_eq!(
            json["target_ghost_id"],
            "ba8a856d-d63b-48f4-9def-c1a9747b94a5"
        );
        approx("nearest_ghost_distance", 0.1337);
        approx("second_nearest_ghost_distance", 0.1555);
        approx("route_margin", 0.0020872438);
        assert_eq!(json["agency_transition"], "REMEMBER->LOCK");
        assert_eq!(json["force_policy"], "lock_earned");
        approx("force_pull_strength", 0.3);
        approx("force_distance_threshold", 0.05);
        approx("force_decay_rate", 1.0);
        approx("force_unfold_factor", 1.0);
        approx("force_unfold_retry_factor", 1.0);
        assert_eq!(json["answer_lock_boundary"], "lock_payload");
        assert_eq!(json["projection_strategy"], "vq_correction_packet");
        approx("ghost_pull_delta_norm", 10.0);

        let reloaded = CorrectionPacketStore::load_from_jsonl(&path).expect("reload");
        let p = &reloaded.packets_for_code(7)[0];
        assert_eq!(p.packet_id, "hybrid");
        assert_eq!(p.text_fact.as_deref(), Some("owner=Jason"));
        assert_eq!(p.route_code.as_deref(), Some("vq_110"));
        assert_eq!(
            p.route_motif_id.as_deref(),
            Some("live_hidden::raw64::002::trap_task::neutral::hinge_window")
        );
        assert_eq!(
            p.target_ghost_id.as_deref(),
            Some("ba8a856d-d63b-48f4-9def-c1a9747b94a5")
        );
        assert_eq!(p.nearest_ghost_distance, Some(0.1337));
        assert_eq!(p.second_nearest_ghost_distance, Some(0.1555));
        assert_eq!(p.route_margin, Some(0.0020872438));
        assert_eq!(p.agency_transition.as_deref(), Some("REMEMBER->LOCK"));
        assert_eq!(p.force_policy.as_deref(), Some("lock_earned"));
        assert_eq!(p.force_pull_strength, Some(0.3));
        assert_eq!(p.force_distance_threshold, Some(0.05));
        assert_eq!(p.force_decay_rate, Some(1.0));
        assert_eq!(p.force_unfold_factor, Some(1.0));
        assert_eq!(p.force_unfold_retry_factor, Some(1.0));
        assert_eq!(p.answer_lock_boundary.as_deref(), Some("lock_payload"));
        assert_eq!(
            p.projection_strategy.as_deref(),
            Some("vq_correction_packet")
        );
        assert_eq!(p.ghost_pull_delta_norm, Some(10.0));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn forward_with_decay_returns_smaller_delta_after_fires() {
        let mut store = CorrectionPacketStore::new();
        store.insert(fixture_packet(7, "p1"));
        let probe = [0f32; 64];

        let firings_initial = store.forward_with_decay(7, &probe, Some(0.5));
        assert_eq!(firings_initial.len(), 1);
        let initial_pull = firings_initial[0].2;
        let initial_norm: f32 = firings_initial[0]
            .1
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!((initial_norm - initial_pull).abs() < 1e-5);

        // Simulate one fire on the packet.
        firings_initial[0].0.record_fire(1);

        let firings_after = store.forward_with_decay(7, &probe, Some(0.5));
        assert_eq!(firings_after.len(), 1);
        let after_pull = firings_after[0].2;
        let after_norm: f32 = firings_after[0].1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            after_pull < initial_pull,
            "decay should reduce effective pull"
        );
        assert!((after_norm - after_pull).abs() < 1e-5);
    }
}
