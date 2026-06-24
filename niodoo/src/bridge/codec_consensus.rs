//! Codec correspondence metric (DEEP_DIVE_ROADMAP P3-B, reframed to match source).
//!
//! Deep dive `2026-05-03_179` ("The Latent Transport Layer") names a real gap: the
//! three codecs encode the *same* 64D state into three disjoint output spaces —
//! `CodebookVQ` (discrete index, L2-nearest of 256 centroids), `SecretSauce V3`
//! (64 Unicode glyphs), and `RAVE` (4096D hidden latent) — and **no metric answers
//! whether they agree on the underlying state identity** ("if the codebook says index
//! 42 and SecretSauce says U+1D40X, do they represent the same state?").
//!
//! The roadmap's original sketch (`codec_consensus_score` = pick lowest reconstruction
//! MSE) is degenerate here: SecretSauce V3 is near-lossless by construction (cosine
//! > 0.999), so it would win every MSE contest and the picker would be trivial. These
//! codecs are not interchangeable compressors competing on fidelity — they are
//! different transport layers. So instead of *selecting* a codec, we measure
//! **correspondence**: decode each codec back to 64D and compare the reconstructions.
//!
//! High pairwise agreement => the state is cleanly captured by all codecs. Low
//! agreement => a hinge / transition state the discrete codebook cannot pin (179:
//! hinge windows have *zero* exact codebook matches), which is a confidence/risk
//! signal for any correction keyed on that state.
//!
//! This is a pure measurement with no effect on live correction routing, in line with
//! the repo's telemetry-truth discipline.
//!
//! ## Dependency injection
//!
//! The `bridge` module compiles into both the library crate and the `niodoo` binary
//! crate, which expose the SecretSauce V3 codec at *different* module paths
//! (`crate::secret_sauce_codec` vs `crate::runtime::secret_sauce_codec`). To stay
//! valid in both, this module takes no hard dependency on either: the caller supplies
//! the SecretSauce roundtrip as a closure, using whichever path is valid in its crate.
//! RAVE participates only via [`CodecAgreement::with_rave`] because it operates on
//! 4096D hidden tensors the caller must roundtrip itself.

use crate::bridge::codebook::CodebookVQ;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = na * nb;
    if denom < 1e-8 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Cross-codec correspondence for a single 64D state vector.
///
/// All cosines are in `[-1, 1]`; `agreement` is the clamped mean of the participating
/// pairwise cosines, in `[0, 1]`, where `1.0` means every codec reconstructs the same
/// state and a low value flags disagreement (a risk signal for corrections).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CodecAgreement {
    /// Cosine between the input state and the codebook's nearest centroid.
    pub codebook_cos: f32,
    /// L2 distance from the input to its nearest codebook centroid (quant error).
    pub codebook_l2: f32,
    /// Cosine between the input and its SecretSauce V3 roundtrip (~1.0 by construction).
    pub secret_sauce_cos: f32,
    /// L2 between the input and its SecretSauce V3 roundtrip (~0 by construction).
    pub secret_sauce_l2: f32,
    /// Cosine between the codebook centroid and the SecretSauce roundtrip — the
    /// cross-codec correspondence: do the discrete and Unicode codecs land on the
    /// same state?
    pub cross_cos: f32,
    /// Number of codecs that participated (2 for the 64D-native pair; 3 after
    /// [`CodecAgreement::with_rave`]).
    pub participating: u8,
    /// Scalar agreement in `[0, 1]`: clamped mean of the participating pairwise
    /// cosines. High = codecs agree on the state; low = disagreement / risk.
    pub agreement: f32,
}

impl CodecAgreement {
    /// Fold a RAVE hidden-state roundtrip cosine into the agreement.
    ///
    /// RAVE operates on 4096D hidden states, not the 64D state, so its roundtrip
    /// cosine (`cosine(hidden, decode(encode(hidden)))`) must be computed by the
    /// caller where the hidden tensor and loaded codec are available, then passed
    /// here. `agreement` is recomputed as the clamped mean of all four cosines.
    /// Note this mixes a 4096D cosine with the three 64D cosines as a coarse joint
    /// confidence — it is intentionally a blunt aggregate, not a calibrated score.
    pub fn with_rave(mut self, rave_cos: f32) -> Self {
        let sum =
            self.codebook_cos + self.secret_sauce_cos + self.cross_cos + rave_cos.clamp(-1.0, 1.0);
        self.agreement = (sum / 4.0).clamp(0.0, 1.0);
        self.participating = 3;
        self
    }
}

/// Compute codec correspondence for a 64D state across the two 64D-native codecs
/// (`CodebookVQ` + `SecretSauce V3`).
///
/// `secret_sauce_roundtrip` must encode the state via SecretSauce V3 and decode it
/// back to a 64D vector, returning `None` on failure. Callers pass it using whichever
/// `secret_sauce_codec` module path is valid in their crate (see module docs).
///
/// Returns `None` when the state is not exactly 64D, contains non-finite values, or
/// the SecretSauce roundtrip fails / returns a non-64D vector — callers should treat
/// that as "no signal" rather than a low score.
pub fn codec_agreement_64d<F>(
    state: &[f32],
    codebook: &CodebookVQ,
    secret_sauce_roundtrip: F,
) -> Option<CodecAgreement>
where
    F: FnOnce(&[f32]) -> Option<Vec<f32>>,
{
    if state.len() != 64 || state.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // CodebookVQ roundtrip: nearest of 256 centroids (lossy, discrete).
    let code = codebook.encode(state);
    let codebook_recon = codebook.decode(code);
    let codebook_l2 = codebook.encode_error(state, code);
    let codebook_cos = cosine(state, codebook_recon);

    // SecretSauce V3 roundtrip: near-lossless Unicode transport (cosine ~1.0).
    let ss_recon = secret_sauce_roundtrip(state)?;
    if ss_recon.len() != 64 || ss_recon.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let secret_sauce_cos = cosine(state, &ss_recon);
    let secret_sauce_l2 = l2(state, &ss_recon);

    // Cross-codec correspondence: does the discrete centroid land where the
    // near-lossless Unicode codec confirms the state to be?
    let cross_cos = cosine(codebook_recon, &ss_recon);

    // Mean of the three pairwise cosines, clamped. The codebook term is the one that
    // moves: when the state sits between centroids (hinge windows), codebook_cos and
    // cross_cos drop while secret_sauce_cos stays ~1.0, pulling agreement down.
    let agreement = ((codebook_cos + secret_sauce_cos + cross_cos) / 3.0).clamp(0.0, 1.0);

    Some(CodecAgreement {
        codebook_cos,
        codebook_l2,
        secret_sauce_cos,
        secret_sauce_l2,
        cross_cos,
        participating: 2,
        agreement,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Codebook of 256 bounded, varied centroids (values in ~[-0.5, 0.5]).
    fn varied_codebook() -> CodebookVQ {
        let mut entries = Vec::with_capacity(256);
        for i in 0u16..256 {
            let mut arr = [0f32; 64];
            for (j, slot) in arr.iter_mut().enumerate() {
                *slot = ((i as f32 * 7.0 + j as f32 * 13.0) % 100.0) / 100.0 - 0.5;
            }
            entries.push(arr);
        }
        CodebookVQ { entries }
    }

    /// Stand-in for the near-lossless SecretSauce V3 roundtrip: identity (cosine 1.0).
    /// The real codec is wired at the call site; here we exercise the correspondence
    /// math with SecretSauce behaving as its >0.999 guarantee promises.
    fn lossless_roundtrip(state: &[f32]) -> Option<Vec<f32>> {
        Some(state.to_vec())
    }

    #[test]
    fn rejects_wrong_length_and_non_finite() {
        let cb = varied_codebook();
        assert!(codec_agreement_64d(&[0.1; 32], &cb, lossless_roundtrip).is_none());
        let mut bad = vec![0.1f32; 64];
        bad[3] = f32::NAN;
        assert!(codec_agreement_64d(&bad, &cb, lossless_roundtrip).is_none());
    }

    #[test]
    fn rejects_bad_roundtrip_output() {
        let cb = varied_codebook();
        let state = cb.decode(10).to_vec();
        // Roundtrip returns wrong-length / failure => no signal.
        assert!(codec_agreement_64d(&state, &cb, |_| Some(vec![0.0; 32])).is_none());
        assert!(codec_agreement_64d(&state, &cb, |_| None).is_none());
    }

    #[test]
    fn exact_centroid_yields_high_agreement() {
        let cb = varied_codebook();
        // A state that IS centroid 128 — the codebook reconstructs it exactly.
        let state = cb.decode(128).to_vec();
        let a = codec_agreement_64d(&state, &cb, lossless_roundtrip).expect("agreement");
        assert!(a.codebook_l2 < 1e-5, "codebook_l2={}", a.codebook_l2);
        assert!(a.codebook_cos > 0.999, "codebook_cos={}", a.codebook_cos);
        assert!(a.secret_sauce_cos > 0.999, "secret_sauce_cos={}", a.secret_sauce_cos);
        assert!(a.agreement > 0.999, "agreement={}", a.agreement);
        assert_eq!(a.participating, 2);
    }

    #[test]
    fn off_centroid_lowers_agreement() {
        let cb = varied_codebook();
        // A bounded state deliberately unlike the ramp-pattern centroids: alternating
        // +/-0.49. The codebook must snap to a distant centroid, so cross/codebook
        // cosines fall while SecretSauce (near-lossless) stays high.
        let state: Vec<f32> = (0..64)
            .map(|j| if j % 2 == 0 { 0.49 } else { -0.49 })
            .collect();
        let off = codec_agreement_64d(&state, &cb, lossless_roundtrip).expect("agreement");

        let exact_state = cb.decode(200).to_vec();
        let exact = codec_agreement_64d(&exact_state, &cb, lossless_roundtrip).expect("agreement");

        assert!(off.secret_sauce_cos > 0.999);
        assert!(
            off.agreement < exact.agreement,
            "off={} exact={}",
            off.agreement,
            exact.agreement
        );
        assert!(off.codebook_l2 > exact.codebook_l2);
    }

    #[test]
    fn with_rave_extends_to_three_codecs() {
        let cb = varied_codebook();
        let state = cb.decode(64).to_vec();
        let base = codec_agreement_64d(&state, &cb, lossless_roundtrip).expect("agreement");
        let extended = base.with_rave(1.0);
        assert_eq!(extended.participating, 3);
        assert!(extended.agreement > 0.999, "agreement={}", extended.agreement);
        let degraded = base.with_rave(-0.5);
        assert!(degraded.agreement < base.agreement);
    }
}
