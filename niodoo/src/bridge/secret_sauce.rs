//! Minimal secret_sauce V3 Unicode decode for bridge-crossing artifacts.
//!
//! This file intentionally duplicates only the small decode needed by the
//! correction-packet Unicode smoke so the same `bridge::` module tree compiles
//! in BOTH contexts:
//! - as the library module (`niodoo::bridge`) and
//! - as the runtime binary's local `mod bridge;` tree in `main.rs`.

use anyhow::Result;

const SECRET_SAUCE_SCALE_SENTENCE: f32 = 0.75;

const SECRET_SAUCE_BLOCKS: [(&str, u32, u32); 4] = [
    ("braille", 0x2800, 0x28FF),
    ("cuneiform", 0x12000, 0x123FF),
    ("math_bold", 0x1D400, 0x1D433),
    ("math_script", 0x1D49C, 0x1D4CF),
];

fn unscaled_segment(values: &[f32], scale: f32) -> Vec<f32> {
    values
        .iter()
        .map(|value| {
            let clamped = value.clamp(-0.999_999, 0.999_999);
            0.5 * ((1.0 + clamped) / (1.0 - clamped)).ln() * scale
        })
        .collect()
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

/// Decode a 64-glyph V3 payload to a 64D sentence anchor vector.
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
