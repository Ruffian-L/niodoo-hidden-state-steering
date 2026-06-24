//! Secret-sauce codec: v1/v2/v3 encode/decode + 64D compression utilities.
//! Extracted from main.rs as part of the comprehensive refactor
//! (pre-refactor-main-split-20260508 backup).

use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::{LoadedModelArch, StatePacketSecretSauce};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) enum SecretSauceVersion {
    V1,
    V2,
    V3,
}

impl SecretSauceVersion {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::V2 => "v2",
            Self::V3 => "v3",
        }
    }

    fn glyph_len(self) -> usize {
        match self {
            Self::V1 => 64,
            Self::V2 => 128,
            Self::V3 => 64,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum SecretSauceInputVersion {
    Auto,
    V1,
    V2,
    V3,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct SecretSauceSegments {
    pub(crate) hidden_64: Vec<f32>,
    #[serde(default)]
    pub(crate) sentence_32: Vec<f32>,
    #[serde(default)]
    pub(crate) momentum_16: Vec<f32>,
    #[serde(default)]
    pub(crate) scalar_8: Vec<f32>,
    #[serde(default)]
    pub(crate) control_8: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct SecretSauceDecoded {
    pub(crate) version: SecretSauceVersion,
    pub(crate) segments: SecretSauceSegments,
}

pub(crate) fn build_state_packet_secret_sauce(
    version: SecretSauceVersion,
    unicode_string: String,
    segments: SecretSauceSegments,
    vector_64d: Vec<f32>,
) -> StatePacketSecretSauce {
    StatePacketSecretSauce {
        version,
        unicode_string,
        segments,
        vector_64d,
    }
}

pub(crate) const SECRET_SAUCE_SCALE_HIDDEN: f32 = 0.75;
pub(crate) const SECRET_SAUCE_SCALE_SENTENCE: f32 = 0.75;
pub(crate) const SECRET_SAUCE_SCALE_MOMENTUM: f32 = 0.25;
pub(crate) const SECRET_SAUCE_SCALE_LAST_MOTIF: f32 = 0.20;
pub(crate) const SECRET_SAUCE_SCALE_LAST_RECOVERY: f32 = 0.10;
pub(crate) const SECRET_SAUCE_SCALE_LAST_ABSENCE: f32 = 1.50;
pub(crate) const SECRET_SAUCE_SCALE_LAST_TRAP: f32 = 1.00;
pub(crate) const SECRET_SAUCE_SCALE_STRESS: f32 = 5.00;
pub(crate) const SECRET_SAUCE_SCALE_BOREDOM: f32 = 2.00;
pub(crate) const SECRET_SAUCE_SCALE_DYNAMIC_GRAVITY: f32 = 1.00;
pub(crate) const SECRET_SAUCE_SCALE_DYNAMIC_REPULSION: f32 = 3.00;
pub(crate) const SECRET_SAUCE_SCALE_PHYSICS_BLEND: f32 = 3.00;
pub(crate) const SECRET_SAUCE_SCALE_REQUEST_COUNT: f32 = 5.00;
pub(crate) const SECRET_SAUCE_SCALE_INSIGHT_PERSISTENCE: f32 = 8.00;
pub(crate) const SECRET_SAUCE_SCALE_EMPATHY_SPIKE: f32 = 1.00;
pub(crate) const SECRET_SAUCE_RESTORE_DECAY_STEPS: usize = 8;
pub(crate) const SECRET_SAUCE_RESTORE_HIDDEN_WEIGHT: f32 = 0.35;
pub(crate) const SECRET_SAUCE_RESTORE_SENTENCE_WEIGHT: f32 = 0.16;
pub(crate) const SECRET_SAUCE_RESTORE_MOMENTUM_WEIGHT: f32 = 0.0;
pub(crate) const SECRET_SAUCE_RESTORE_SENTENCE_ALIGNMENT_FLOOR: f32 = 0.15;

pub(crate) fn compress_slice_to_dim(raw: &[f32], out_dim: usize) -> Vec<f32> {
    if out_dim == 0 {
        return Vec::new();
    }
    if raw.is_empty() {
        return vec![0.0; out_dim];
    }

    let mut buckets = vec![0.0f32; out_dim];
    let mut counts = vec![0usize; out_dim];
    for (idx, value) in raw.iter().enumerate() {
        let bucket = idx * out_dim / raw.len();
        buckets[bucket] += *value;
        counts[bucket] += 1;
    }
    for (bucket, count) in buckets.iter_mut().zip(counts.iter()) {
        if *count > 0 {
            *bucket /= *count as f32;
        }
    }
    buckets
}

pub(crate) fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn compress_slice_to_hash_sparse_route64(raw: &[f32]) -> Vec<f32> {
    const OUT_DIM: usize = 64;
    const PROBES_PER_DIM: usize = 3;
    const SEED: u64 = 17;

    if raw.is_empty() {
        return vec![0.0; OUT_DIM];
    }

    let mut buckets = vec![0.0f32; OUT_DIM];
    for (idx, value) in raw.iter().enumerate() {
        let base = SEED ^ (idx as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
        for probe in 0..PROBES_PER_DIM {
            let hash = splitmix64(base ^ (probe as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
            let bucket = (hash as usize) & (OUT_DIM - 1);
            let sign = if ((hash >> 6) & 1) == 0 { 1.0 } else { -1.0 };
            buckets[bucket] += *value * sign;
        }
    }

    let scale = ((raw.len() * PROBES_PER_DIM) as f32 / OUT_DIM as f32).sqrt();
    if scale > 0.0 {
        for bucket in &mut buckets {
            *bucket /= scale;
        }
    }
    buckets
}

pub(crate) fn compress_runtime_hidden_64d(
    raw: &[f32],
    model_arch: LoadedModelArch,
) -> (&'static str, Vec<f32>) {
    match model_arch {
        LoadedModelArch::Qwen35 if raw.len() == 5120 => (
            "qwen35_route64_hash_sparse_k3_seed17_v2",
            compress_slice_to_hash_sparse_route64(raw),
        ),
        _ => ("bucket_mean64_v1", compress_slice_to_dim(raw, 64)),
    }
}

pub(crate) fn compress_tensor_to_dim(hidden: &Tensor, out_dim: usize) -> Result<Vec<f32>> {
    let flat = hidden
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    Ok(compress_slice_to_dim(&flat, out_dim))
}

pub(crate) fn compress_hidden_state_to_64d(hidden: &Tensor) -> Result<Vec<f32>> {
    compress_tensor_to_dim(hidden, 64)
}

/// Cosine similarity between two f32 vectors. Returns value in [-1, 1].
pub(crate) fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = norm_a * norm_b;
    if denom < 1e-8 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

fn scaled_segment(values: &[f32], scale: f32) -> Vec<f32> {
    values
        .iter()
        .map(|value| (value / scale.max(1e-6)).tanh())
        .collect()
}

fn unscaled_segment(values: &[f32], scale: f32) -> Vec<f32> {
    values
        .iter()
        .map(|value| {
            let clamped = value.clamp(-0.999_999, 0.999_999);
            0.5 * ((1.0 + clamped) / (1.0 - clamped)).ln() * scale
        })
        .collect()
}

const SECRET_SAUCE_BLOCKS: [(&str, u32, u32); 4] = [
    ("braille", 0x2800, 0x28FF),
    ("cuneiform", 0x12000, 0x123FF),
    ("math_bold", 0x1D400, 0x1D433),
    ("math_script", 0x1D49C, 0x1D4CF),
];

#[allow(dead_code)]
fn normalized_scalar(value: f32, scale: f32) -> f32 {
    (value / scale.max(1e-6)).tanh()
}

fn denormalized_scalar(value: f32, scale: f32) -> f32 {
    let clamped = value.clamp(-0.999_999, 0.999_999);
    0.5 * ((1.0 + clamped) / (1.0 - clamped)).ln() * scale
}

fn encode_normalized_secret_sauce(values: &[f32]) -> Result<String> {
    let mut output = String::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        let (_name, start, end) = SECRET_SAUCE_BLOCKS[idx % SECRET_SAUCE_BLOCKS.len()];
        let span = (end - start) as f32;
        let normalized = ((*value + 1.0) / 2.0).clamp(0.0, 1.0);
        let offset = (normalized * span).round() as u32;
        let codepoint = start + offset;
        let glyph = char::from_u32(codepoint)
            .ok_or_else(|| anyhow::anyhow!("invalid secret sauce codepoint U+{:04X}", codepoint))?;
        output.push(glyph);
    }
    Ok(output)
}

fn decode_normalized_secret_sauce(secret_sauce: &str) -> Result<Vec<f32>> {
    let chars: Vec<char> = secret_sauce.chars().collect();
    let mut vector = Vec::with_capacity(chars.len());
    for (idx, glyph) in chars.into_iter().enumerate() {
        let (_name, start, end) = SECRET_SAUCE_BLOCKS[idx % SECRET_SAUCE_BLOCKS.len()];
        let codepoint = glyph as u32;
        if codepoint < start || codepoint > end {
            anyhow::bail!(
                "secret sauce glyph '{}' at index {} is outside expected block U+{:04X}-U+{:04X}",
                glyph,
                idx,
                start,
                end
            );
        }
        let span = (end - start) as f32;
        let normalized = (codepoint - start) as f32 / span.max(1.0);
        vector.push(normalized * 2.0 - 1.0);
    }
    Ok(vector)
}

#[allow(dead_code)]
pub(crate) fn encode_secret_sauce_v1(vector_64d: &[f32]) -> Result<String> {
    if vector_64d.len() != 64 {
        anyhow::bail!(
            "secret sauce v1 requires 64 values, got {}",
            vector_64d.len()
        );
    }
    encode_normalized_secret_sauce(&scaled_segment(vector_64d, SECRET_SAUCE_SCALE_HIDDEN))
}

pub(crate) fn decode_secret_sauce_v1(secret_sauce: &str) -> Result<SecretSauceDecoded> {
    let chars: Vec<char> = secret_sauce.chars().collect();
    if chars.len() != SecretSauceVersion::V1.glyph_len() {
        anyhow::bail!(
            "secret sauce v1 must contain exactly {} unicode scalars, got {}",
            SecretSauceVersion::V1.glyph_len(),
            chars.len()
        );
    }
    let normalized = decode_normalized_secret_sauce(secret_sauce)?;
    let hidden_64 = unscaled_segment(&normalized, SECRET_SAUCE_SCALE_HIDDEN);
    Ok(SecretSauceDecoded {
        version: SecretSauceVersion::V1,
        segments: SecretSauceSegments {
            hidden_64: hidden_64.clone(),
            sentence_32: Vec::new(),
            momentum_16: Vec::new(),
            scalar_8: Vec::new(),
            control_8: Vec::new(),
        },
    })
}

pub(crate) fn encode_secret_sauce_v3(sentence_anchor_64d: &[f32]) -> Result<String> {
    if sentence_anchor_64d.len() != 64 {
        anyhow::bail!(
            "secret sauce v3 sentence anchor requires 64 values, got {}",
            sentence_anchor_64d.len()
        );
    }
    encode_normalized_secret_sauce(&scaled_segment(
        sentence_anchor_64d,
        SECRET_SAUCE_SCALE_SENTENCE,
    ))
}

pub(crate) fn decode_secret_sauce_v3(secret_sauce: &str) -> Result<SecretSauceDecoded> {
    let chars: Vec<char> = secret_sauce.chars().collect();
    if chars.len() != SecretSauceVersion::V3.glyph_len() {
        anyhow::bail!(
            "secret sauce v3 must contain exactly {} unicode scalars, got {}",
            SecretSauceVersion::V3.glyph_len(),
            chars.len()
        );
    }
    let normalized = decode_normalized_secret_sauce(secret_sauce)?;
    let sentence_anchor_64 = unscaled_segment(&normalized, SECRET_SAUCE_SCALE_SENTENCE);
    Ok(SecretSauceDecoded {
        version: SecretSauceVersion::V3,
        segments: SecretSauceSegments {
            hidden_64: sentence_anchor_64,
            sentence_32: Vec::new(),
            momentum_16: Vec::new(),
            scalar_8: Vec::new(),
            control_8: Vec::new(),
        },
    })
}

#[allow(dead_code)]
pub(crate) fn encode_secret_sauce_v2(segments: &SecretSauceSegments) -> Result<String> {
    if segments.hidden_64.len() != 64 {
        anyhow::bail!(
            "secret sauce v2 hidden_64 must contain 64 values, got {}",
            segments.hidden_64.len()
        );
    }
    if segments.sentence_32.len() != 32 {
        anyhow::bail!(
            "secret sauce v2 sentence_32 must contain 32 values, got {}",
            segments.sentence_32.len()
        );
    }
    if segments.momentum_16.len() != 16 {
        anyhow::bail!(
            "secret sauce v2 momentum_16 must contain 16 values, got {}",
            segments.momentum_16.len()
        );
    }
    if segments.scalar_8.len() != 8 {
        anyhow::bail!(
            "secret sauce v2 scalar_8 must contain 8 values, got {}",
            segments.scalar_8.len()
        );
    }
    if segments.control_8.len() != 8 {
        anyhow::bail!(
            "secret sauce v2 control_8 must contain 8 values, got {}",
            segments.control_8.len()
        );
    }

    let mut normalized = Vec::with_capacity(SecretSauceVersion::V2.glyph_len());
    normalized.extend(scaled_segment(
        &segments.hidden_64,
        SECRET_SAUCE_SCALE_HIDDEN,
    ));
    normalized.extend(scaled_segment(
        &segments.sentence_32,
        SECRET_SAUCE_SCALE_SENTENCE,
    ));
    normalized.extend(scaled_segment(
        &segments.momentum_16,
        SECRET_SAUCE_SCALE_MOMENTUM,
    ));
    normalized.extend([
        normalized_scalar(segments.scalar_8[0], SECRET_SAUCE_SCALE_LAST_MOTIF),
        normalized_scalar(segments.scalar_8[1], SECRET_SAUCE_SCALE_LAST_RECOVERY),
        normalized_scalar(segments.scalar_8[2], SECRET_SAUCE_SCALE_LAST_ABSENCE),
        normalized_scalar(segments.scalar_8[3], SECRET_SAUCE_SCALE_LAST_TRAP),
        normalized_scalar(segments.scalar_8[4], SECRET_SAUCE_SCALE_STRESS),
        normalized_scalar(segments.scalar_8[5], SECRET_SAUCE_SCALE_BOREDOM),
        normalized_scalar(segments.scalar_8[6], SECRET_SAUCE_SCALE_DYNAMIC_GRAVITY),
        normalized_scalar(segments.scalar_8[7], SECRET_SAUCE_SCALE_DYNAMIC_REPULSION),
    ]);
    normalized.extend([
        normalized_scalar(segments.control_8[0], SECRET_SAUCE_SCALE_PHYSICS_BLEND),
        if segments.control_8[1] >= 0.0 {
            1.0
        } else {
            -1.0
        },
        if segments.control_8[2] >= 0.0 {
            1.0
        } else {
            -1.0
        },
        normalized_scalar(segments.control_8[3], SECRET_SAUCE_SCALE_REQUEST_COUNT),
        if segments.control_8[4] >= 0.0 {
            1.0
        } else {
            -1.0
        },
        normalized_scalar(
            segments.control_8[5],
            SECRET_SAUCE_SCALE_INSIGHT_PERSISTENCE,
        ),
        normalized_scalar(segments.control_8[6], SECRET_SAUCE_SCALE_EMPATHY_SPIKE),
        normalized_scalar(segments.control_8[7], 1.0),
    ]);
    debug_assert_eq!(normalized.len(), SecretSauceVersion::V2.glyph_len());
    encode_normalized_secret_sauce(&normalized)
}

pub(crate) fn decode_secret_sauce_v2(secret_sauce: &str) -> Result<SecretSauceDecoded> {
    let chars: Vec<char> = secret_sauce.chars().collect();
    if chars.len() != SecretSauceVersion::V2.glyph_len() {
        anyhow::bail!(
            "secret sauce v2 must contain exactly {} unicode scalars, got {}",
            SecretSauceVersion::V2.glyph_len(),
            chars.len()
        );
    }
    let normalized = decode_normalized_secret_sauce(secret_sauce)?;
    let hidden_64 = unscaled_segment(&normalized[0..64], SECRET_SAUCE_SCALE_HIDDEN);
    let sentence_32 = unscaled_segment(&normalized[64..96], SECRET_SAUCE_SCALE_SENTENCE);
    let momentum_16 = unscaled_segment(&normalized[96..112], SECRET_SAUCE_SCALE_MOMENTUM);
    let scalar_8 = vec![
        denormalized_scalar(normalized[112], SECRET_SAUCE_SCALE_LAST_MOTIF),
        denormalized_scalar(normalized[113], SECRET_SAUCE_SCALE_LAST_RECOVERY),
        denormalized_scalar(normalized[114], SECRET_SAUCE_SCALE_LAST_ABSENCE),
        denormalized_scalar(normalized[115], SECRET_SAUCE_SCALE_LAST_TRAP),
        denormalized_scalar(normalized[116], SECRET_SAUCE_SCALE_STRESS),
        denormalized_scalar(normalized[117], SECRET_SAUCE_SCALE_BOREDOM),
        denormalized_scalar(normalized[118], SECRET_SAUCE_SCALE_DYNAMIC_GRAVITY),
        denormalized_scalar(normalized[119], SECRET_SAUCE_SCALE_DYNAMIC_REPULSION),
    ];
    let control_8 = vec![
        denormalized_scalar(normalized[120], SECRET_SAUCE_SCALE_PHYSICS_BLEND),
        if normalized[121] >= 0.0 { 1.0 } else { -1.0 },
        if normalized[122] >= 0.0 { 1.0 } else { -1.0 },
        denormalized_scalar(normalized[123], SECRET_SAUCE_SCALE_REQUEST_COUNT),
        if normalized[124] >= 0.0 { 1.0 } else { -1.0 },
        denormalized_scalar(normalized[125], SECRET_SAUCE_SCALE_INSIGHT_PERSISTENCE),
        denormalized_scalar(normalized[126], SECRET_SAUCE_SCALE_EMPATHY_SPIKE),
        denormalized_scalar(normalized[127], 1.0),
    ];
    Ok(SecretSauceDecoded {
        version: SecretSauceVersion::V2,
        segments: SecretSauceSegments {
            hidden_64,
            sentence_32,
            momentum_16,
            scalar_8,
            control_8,
        },
    })
}

pub(crate) fn decode_secret_sauce(
    secret_sauce: &str,
    requested_version: SecretSauceInputVersion,
) -> Result<SecretSauceDecoded> {
    let glyph_count = secret_sauce.chars().count();
    match (requested_version, glyph_count) {
        (SecretSauceInputVersion::V1, 64) => decode_secret_sauce_v1(secret_sauce),
        (SecretSauceInputVersion::V2, 128) => decode_secret_sauce_v2(secret_sauce),
        (SecretSauceInputVersion::V3, 64) => decode_secret_sauce_v3(secret_sauce),
        (SecretSauceInputVersion::Auto, 128) => decode_secret_sauce_v2(secret_sauce),
        (SecretSauceInputVersion::Auto, 64) => decode_secret_sauce_v3(secret_sauce),
        _ => anyhow::bail!(
            "secret sauce version/length mismatch: requested={:?} expected lengths v1={} v2={} v3={} got {}",
            requested_version,
            SecretSauceVersion::V1.glyph_len(),
            SecretSauceVersion::V2.glyph_len(),
            SecretSauceVersion::V3.glyph_len(),
            glyph_count
        ),
    }
}
