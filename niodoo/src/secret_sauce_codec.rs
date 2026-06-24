//! Pure encode/decode for secret_sauce V3 round-trip tests (matches `main.rs` scalar transport).

use anyhow::Result;

pub const SECRET_SAUCE_SCALE_SENTENCE: f32 = 0.75;

const SECRET_SAUCE_BLOCKS: [(&str, u32, u32); 4] = [
    ("braille", 0x2800, 0x28FF),
    ("cuneiform", 0x12000, 0x123FF),
    ("math_bold", 0x1D400, 0x1D433),
    ("math_script", 0x1D49C, 0x1D4CF),
];

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

/// Decodes a 64-glyph V3 payload to the 64D sentence anchor (same as `decode_secret_sauce_v3` in main).
pub fn decode_v3_sentence_anchor(secret_sauce: &str) -> Result<Vec<f32>> {
    let n = secret_sauce.chars().count();
    if n != 64 {
        anyhow::bail!(
            "secret sauce v3 must contain exactly 64 unicode scalars, got {}",
            n
        );
    }
    let normalized = decode_normalized_secret_sauce(secret_sauce)?;
    Ok(unscaled_segment(&normalized, SECRET_SAUCE_SCALE_SENTENCE))
}

/// Self-encode for property tests (roundtrip after quantize).
pub fn encode_v3_sentence_anchor(sentence_anchor_64d: &[f32]) -> Result<String> {
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

pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
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

pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn quantize_roundtrip_high_cosine() {
        let v: Vec<f32> = (0..64)
            .map(|i| (i as f32 * 0.031 - 1.0).sin() * 0.4)
            .collect();
        let s = encode_v3_sentence_anchor(&v).unwrap();
        let w = decode_v3_sentence_anchor(&s).unwrap();
        let c = cosine_similarity_f32(&v, &w);
        assert!(
            c >= 0.999,
            "cosine similarity after quantize roundtrip: {}",
            c
        );
    }
}
