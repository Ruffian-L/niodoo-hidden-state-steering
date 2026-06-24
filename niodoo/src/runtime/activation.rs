//! Activation gates, vector/tensor utilities, and correction-packet arbitration.
//! Extracted from main.rs as part of the comprehensive refactor
//! (pre-refactor-main-split-20260508 backup).

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::runtime::control_surface::RequestType;
use crate::{
    NIODOO_PRESSURE_GATE_FULL, NIODOO_PRESSURE_GATE_START, NIODOO_VISIBLE_REQUEST_GATE_FLOOR,
    NIODOO_VISIBLE_REQUEST_GATE_RAMP_FULL, NIODOO_VISIBLE_REQUEST_GATE_RAMP_START,
};

pub const NIODOO_WOBBLE: f32 = 0.06; // The "Spark"

// 1.5. PHASE 2: ORBITAL MECHANICS CONSTANTS (Elastic Config v2.2)
// 🌟 NIODOO v3.1: THE GENIUS CONFIG - Verified to solve Drying Towels
// Run 11: blend=1.5, rep=-0.5, grav=0.2 -> WOBBLE-SNAP-BACK to correct "1 hour"
pub const ORBIT_SPEED: f32 = 0.1; // Stable flow
pub const ORBIT_TOP_K: usize = 50;
pub const GRAVITY_WELL: f32 = 0.2; // 🌟 HIGH ELASTICITY (allows thinking phase)

// Helper to manage vector math
pub(crate) fn normalize(v: &mut Vec<f32>) {
    let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag > 1e-6 {
        for x in v.iter_mut() {
            *x /= mag;
        }
    }
}

pub(crate) fn tensor_to_vec_f32(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

pub(crate) fn smoothstep01(value: f32) -> f32 {
    let t = value.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub(crate) fn pressure_activation_gate(ghost_pre_norm: f32) -> f32 {
    if ghost_pre_norm <= NIODOO_PRESSURE_GATE_START {
        0.0
    } else if ghost_pre_norm >= NIODOO_PRESSURE_GATE_FULL {
        1.0
    } else {
        smoothstep01(
            (ghost_pre_norm - NIODOO_PRESSURE_GATE_START)
                / (NIODOO_PRESSURE_GATE_FULL - NIODOO_PRESSURE_GATE_START),
        )
    }
}

pub(crate) fn visible_request_activation_gate(
    enabled: bool,
    current_step: usize,
    last_request_token: usize,
    last_request: Option<RequestType>,
    adrenaline: f32,
    focus_lock_remaining_ticks: usize,
) -> f32 {
    if !enabled {
        return 0.0;
    }
    let Some(req) = last_request else {
        return 0.0;
    };

    let age = current_step.saturating_sub(last_request_token);
    let still_active = match req {
        RequestType::Spike | RequestType::Explore | RequestType::Remember => {
            adrenaline > 0.0 || age <= 10
        }
        RequestType::Focus => focus_lock_remaining_ticks > 0 || age <= 10,
        RequestType::Reset => age <= 6,
    };
    if !still_active {
        return 0.0;
    }

    let ramp = if age <= NIODOO_VISIBLE_REQUEST_GATE_RAMP_START {
        NIODOO_VISIBLE_REQUEST_GATE_FLOOR
    } else if age >= NIODOO_VISIBLE_REQUEST_GATE_RAMP_FULL {
        1.0
    } else {
        let span =
            (NIODOO_VISIBLE_REQUEST_GATE_RAMP_FULL - NIODOO_VISIBLE_REQUEST_GATE_RAMP_START) as f32;
        let t = (age - NIODOO_VISIBLE_REQUEST_GATE_RAMP_START) as f32 / span;
        NIODOO_VISIBLE_REQUEST_GATE_FLOOR
            + (1.0 - NIODOO_VISIBLE_REQUEST_GATE_FLOOR) * smoothstep01(t)
    };

    match req {
        RequestType::Spike | RequestType::Explore | RequestType::Remember => {
            let adrenaline_gate = (adrenaline / 5.0).clamp(NIODOO_VISIBLE_REQUEST_GATE_FLOOR, 1.0);
            ramp.max(adrenaline_gate)
        }
        RequestType::Focus => ramp.clamp(0.55, 0.85),
        RequestType::Reset => ramp.min(0.55),
    }
}

pub(crate) fn vec_to_tensor_f32(data: &[f32], dim: usize, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(data.to_vec(), (dim,), &Device::Cpu)?.to_device(device)?)
}

pub(crate) fn clamp_f32(v: f32, lo: f32, hi: f32) -> f32 {
    v.max(lo).min(hi)
}

pub(crate) fn clamp_usize(v: usize, lo: usize, hi: usize) -> usize {
    v.max(lo).min(hi)
}

/// Parse "start:end" into Some((start, end)) inclusive bounds.
/// Empty/invalid input returns None (gate disabled).
pub(crate) fn parse_step_window(s: &str) -> Option<(usize, usize)> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let (a, b) = s.split_once(':')?;
    let start = a.trim().parse::<usize>().ok()?;
    let end = b.trim().parse::<usize>().ok()?;
    if end < start {
        return None;
    }
    Some((start, end))
}

/// Parse `<substring>:<K>,...` for prompt-level correction-packet top-K routing.
/// Invalid/empty pairs are ignored so a partially bad exploratory map does not
/// change default behavior for all prompts. `rsplit_once` lets substrings contain
/// colons if a future prompt family needs them.
pub(crate) fn parse_correction_packet_prompt_top_k_map(s: &str) -> Vec<(String, usize)> {
    s.split(',')
        .filter_map(|pair| {
            let pair = pair.trim();
            if pair.is_empty() {
                return None;
            }
            let (sub, k) = pair.rsplit_once(':')?;
            let k: usize = k.trim().parse().ok()?;
            let sub = sub.trim().to_lowercase();
            if sub.is_empty() {
                return None;
            }
            Some((sub, k))
        })
        .collect()
}

pub(crate) fn resolve_correction_packet_prompt_top_k_override(
    prompt: &str,
    map: &[(String, usize)],
) -> Option<usize> {
    resolve_correction_packet_prompt_top_k_match(prompt, map).map(|(_, k)| *k)
}

/// §10cm bullet-10 observability: returns the (substring, K) pair
/// from the §10ck top-K map that matched the prompt, so telemetry
/// can record which rule fired. None when map is empty or no
/// substring matches the prompt.
pub(crate) fn resolve_correction_packet_prompt_top_k_match<'a>(
    prompt: &str,
    map: &'a [(String, usize)],
) -> Option<&'a (String, usize)> {
    if map.is_empty() {
        return None;
    }
    let prompt_lc = prompt.to_lowercase();
    map.iter().find(|(sub, _)| prompt_lc.contains(sub.as_str()))
}

pub(crate) fn should_suppress_correction_packets_for_prompt(
    suppress_when_no_prompt_match: bool,
    prompt_top_k_map_nonempty: bool,
    prompt_top_k_match_found: bool,
) -> bool {
    suppress_when_no_prompt_match && prompt_top_k_map_nonempty && !prompt_top_k_match_found
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CorrectionPacketArbitrationMode {
    Disabled,
    Auto,
    NoPacket,
    PacketShadow,
    PacketForce,
}

impl CorrectionPacketArbitrationMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Auto => "auto",
            Self::NoPacket => "no_packet",
            Self::PacketShadow => "packet_shadow",
            Self::PacketForce => "packet_force",
        }
    }
}

pub(crate) fn parse_correction_packet_arbitration_mode(
    raw: &str,
) -> anyhow::Result<CorrectionPacketArbitrationMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "disabled" | "off" => Ok(CorrectionPacketArbitrationMode::Disabled),
        "auto" => Ok(CorrectionPacketArbitrationMode::Auto),
        "no_packet" | "no-packet" | "none" => Ok(CorrectionPacketArbitrationMode::NoPacket),
        "packet_shadow" | "packet-shadow" | "shadow" => {
            Ok(CorrectionPacketArbitrationMode::PacketShadow)
        }
        "packet_force" | "packet-force" | "force" => {
            Ok(CorrectionPacketArbitrationMode::PacketForce)
        }
        other => anyhow::bail!(
            "--correction-packet-arbitration expects disabled|auto|no_packet|packet_shadow|packet_force, got '{other}'"
        ),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CorrectionPacketArbitrationChoice {
    NoPacket,
    PacketShadow,
    PacketForce,
}

impl CorrectionPacketArbitrationChoice {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::NoPacket => "no_packet",
            Self::PacketShadow => "packet_shadow",
            Self::PacketForce => "packet_force",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CorrectionPacketArbitrationDecision {
    pub(crate) choice: CorrectionPacketArbitrationChoice,
    pub(crate) reason: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CorrectionPacketArbitrationInput {
    pub(crate) candidate_count: usize,
    pub(crate) min_target_distance: f32,
    pub(crate) competence_factor: f32,
    pub(crate) healthy_factor_threshold: f32,
    pub(crate) stale_distance_threshold: f32,
}

pub(crate) fn choose_correction_packet_arbitration(
    mode: CorrectionPacketArbitrationMode,
    input: CorrectionPacketArbitrationInput,
) -> CorrectionPacketArbitrationDecision {
    match mode {
        CorrectionPacketArbitrationMode::Disabled
        | CorrectionPacketArbitrationMode::PacketForce => {
            return CorrectionPacketArbitrationDecision {
                choice: CorrectionPacketArbitrationChoice::PacketForce,
                reason: mode.as_str(),
            };
        }
        CorrectionPacketArbitrationMode::NoPacket => {
            return CorrectionPacketArbitrationDecision {
                choice: CorrectionPacketArbitrationChoice::NoPacket,
                reason: "explicit_no_packet",
            };
        }
        CorrectionPacketArbitrationMode::PacketShadow => {
            return CorrectionPacketArbitrationDecision {
                choice: CorrectionPacketArbitrationChoice::PacketShadow,
                reason: "explicit_packet_shadow",
            };
        }
        CorrectionPacketArbitrationMode::Auto => {}
    }

    if input.candidate_count == 0 {
        return CorrectionPacketArbitrationDecision {
            choice: CorrectionPacketArbitrationChoice::NoPacket,
            reason: "no_candidate_packets",
        };
    }

    if input.healthy_factor_threshold > 0.0
        && input.competence_factor.is_finite()
        && input.competence_factor < input.healthy_factor_threshold
    {
        return CorrectionPacketArbitrationDecision {
            choice: CorrectionPacketArbitrationChoice::PacketShadow,
            reason: "healthy_route_proxy",
        };
    }

    if input.stale_distance_threshold > 0.0
        && input.min_target_distance.is_finite()
        && input.min_target_distance > input.stale_distance_threshold
    {
        return CorrectionPacketArbitrationDecision {
            choice: CorrectionPacketArbitrationChoice::PacketShadow,
            reason: "stale_or_wrong_basin_distance",
        };
    }

    CorrectionPacketArbitrationDecision {
        choice: CorrectionPacketArbitrationChoice::PacketForce,
        reason: "auto_allow_force",
    }
}
