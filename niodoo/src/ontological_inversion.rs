use anyhow::{bail, Result};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct InversionMeasurement {
    pub alpha: f32,
    pub p_before: f32,
    pub p_after_naive: f32,
    pub p_after_polarity: f32,
    pub target_p: f32,
    pub norm_before: f32,
    pub norm_after_naive: f32,
    pub norm_after_polarity: f32,
    pub cosine_before: f32,
    pub cosine_after_naive: f32,
    pub cosine_after_polarity: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChainmailMeasurement {
    pub angle_radians: f32,
    pub residual_norm: f32,
    pub start_norm: f32,
    pub returned_norm: f32,
    pub dot_start_returned: f32,
    pub cosine_start_returned: f32,
}

pub fn dot(a: &[f32], b: &[f32]) -> Result<f32> {
    ensure_same_len(a, b)?;
    Ok(a.iter().zip(b).map(|(x, y)| x * y).sum())
}

pub fn norm_sq(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum()
}

pub fn norm(v: &[f32]) -> f32 {
    norm_sq(v).sqrt()
}

pub fn cosine(a: &[f32], b: &[f32]) -> Result<f32> {
    let denom = norm(a) * norm(b);
    if denom <= f32::EPSILON {
        return Ok(0.0);
    }
    Ok(dot(a, b)? / denom)
}

pub fn normalize(v: &[f32]) -> Result<Vec<f32>> {
    let n = norm(v);
    if n <= f32::EPSILON {
        bail!("cannot normalize zero vector");
    }
    Ok(v.iter().map(|x| x / n).collect())
}

pub fn average_vectors(vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
    if vectors.is_empty() {
        bail!("cannot average zero vectors");
    }
    let dim = vectors[0].len();
    if dim == 0 {
        bail!("cannot average empty vectors");
    }
    let mut avg = vec![0.0f32; dim];
    for vector in vectors {
        if vector.len() != dim {
            bail!(
                "dimension mismatch while averaging: got {}, expected {}",
                vector.len(),
                dim
            );
        }
        for (out, value) in avg.iter_mut().zip(vector) {
            *out += *value;
        }
    }
    let scale = 1.0 / vectors.len() as f32;
    for value in &mut avg {
        *value *= scale;
    }
    Ok(avg)
}

pub fn projection_scalar(h: &[f32], target: &[f32]) -> Result<f32> {
    let denom = norm_sq(target);
    if denom <= f32::EPSILON {
        bail!("target vector has zero norm");
    }
    Ok(dot(h, target)? / denom)
}

pub fn naive_projection_flip(h: &[f32], target: &[f32], alpha: f32) -> Result<Vec<f32>> {
    ensure_same_len(h, target)?;
    let p = projection_scalar(h, target)?;
    Ok(h.iter()
        .zip(target)
        .map(|(hv, tv)| *hv - (1.0 + alpha) * p * *tv)
        .collect())
}

pub fn polarity_aware_inversion(h: &[f32], target: &[f32], alpha: f32) -> Result<Vec<f32>> {
    ensure_same_len(h, target)?;
    let p = projection_scalar(h, target)?;
    let target_p = polarity_target_projection(p, alpha);
    let correction = target_p - p;
    Ok(h.iter()
        .zip(target)
        .map(|(hv, tv)| *hv + correction * *tv)
        .collect())
}

pub fn polarity_target_projection(p_before: f32, alpha: f32) -> f32 {
    if p_before.abs() <= f32::EPSILON {
        -alpha
    } else {
        -alpha * p_before.abs()
    }
}

pub fn measure_inversion(h: &[f32], target: &[f32], alpha: f32) -> Result<InversionMeasurement> {
    let target_unit = normalize(target)?;
    let naive = naive_projection_flip(h, &target_unit, alpha)?;
    let polarity = polarity_aware_inversion(h, &target_unit, alpha)?;
    let p_before = projection_scalar(h, &target_unit)?;
    Ok(InversionMeasurement {
        alpha,
        p_before,
        p_after_naive: projection_scalar(&naive, &target_unit)?,
        p_after_polarity: projection_scalar(&polarity, &target_unit)?,
        target_p: polarity_target_projection(p_before, alpha),
        norm_before: norm(h),
        norm_after_naive: norm(&naive),
        norm_after_polarity: norm(&polarity),
        cosine_before: cosine(h, &target_unit)?,
        cosine_after_naive: cosine(&naive, &target_unit)?,
        cosine_after_polarity: cosine(&polarity, &target_unit)?,
    })
}

pub fn householder_reflect(x: &[f32], normal: &[f32]) -> Result<Vec<f32>> {
    ensure_same_len(x, normal)?;
    let denom = norm_sq(normal);
    if denom <= f32::EPSILON {
        bail!("householder normal has zero norm");
    }
    let scale = 2.0 * dot(x, normal)? / denom;
    Ok(x.iter()
        .zip(normal)
        .map(|(xv, nv)| *xv - scale * *nv)
        .collect())
}

pub fn reflection_commutator(x: &[f32], u: &[f32], v: &[f32]) -> Result<Vec<f32>> {
    let x = householder_reflect(x, v)?;
    let x = householder_reflect(&x, u)?;
    let x = householder_reflect(&x, v)?;
    householder_reflect(&x, u)
}

pub fn measure_chainmail_reflection_loop(angle_radians: f32) -> Result<ChainmailMeasurement> {
    let u = vec![1.0f32, 0.0];
    let v = vec![angle_radians.cos(), angle_radians.sin()];
    let start = vec![0.37f32, 0.91];
    let returned = reflection_commutator(&start, &u, &v)?;
    let residual: Vec<f32> = returned.iter().zip(&start).map(|(a, b)| a - b).collect();
    Ok(ChainmailMeasurement {
        angle_radians,
        residual_norm: norm(&residual),
        start_norm: norm(&start),
        returned_norm: norm(&returned),
        dot_start_returned: dot(&start, &returned)?,
        cosine_start_returned: cosine(&start, &returned)?,
    })
}

fn ensure_same_len(a: &[f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        bail!("dimension mismatch: {} vs {}", a.len(), b.len());
    }
    if a.is_empty() {
        bail!("empty vectors are not valid");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naive_flip_moves_negative_projection_toward_target() {
        let target = vec![1.0, 0.0];
        let h = vec![-0.25, 0.75];
        let naive = naive_projection_flip(&h, &target, 2.0).unwrap();

        assert!(projection_scalar(&h, &target).unwrap() < 0.0);
        assert!(projection_scalar(&naive, &target).unwrap() > 0.0);
    }

    #[test]
    fn polarity_aware_inversion_sets_requested_signed_component() {
        let target = vec![1.0, 0.0];
        let h = vec![-0.25, 0.75];
        let inverted = polarity_aware_inversion(&h, &target, 2.0).unwrap();

        let p = projection_scalar(&inverted, &target).unwrap();
        assert!((p + 0.5).abs() < 1e-6);
    }

    #[test]
    fn naive_and_polarity_agree_on_direction_when_projection_is_positive() {
        let target = vec![1.0, 0.0];
        let h = vec![0.25, 0.75];
        let naive = naive_projection_flip(&h, &target, 2.0).unwrap();
        let polarity = polarity_aware_inversion(&h, &target, 2.0).unwrap();

        assert!(projection_scalar(&naive, &target).unwrap() < 0.0);
        assert!((projection_scalar(&polarity, &target).unwrap() + 0.5).abs() < 1e-6);
    }

    #[test]
    fn householder_reflection_is_self_inverse() {
        let normal = vec![0.7, -0.2, 0.4];
        let x = vec![0.1, 0.5, -0.8];
        let once = householder_reflect(&x, &normal).unwrap();
        let twice = householder_reflect(&once, &normal).unwrap();

        for (actual, expected) in twice.iter().zip(&x) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn chainmail_loop_leaves_residual_when_reflections_are_offset() {
        let zero = measure_chainmail_reflection_loop(0.0).unwrap();
        let offset = measure_chainmail_reflection_loop(0.08).unwrap();

        assert!(zero.residual_norm < 1e-6);
        assert!(offset.residual_norm > 0.01);
        assert!((offset.start_norm - offset.returned_norm).abs() < 1e-5);
    }
}
