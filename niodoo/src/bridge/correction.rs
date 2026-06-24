//! # CorrectionDelta — Optional Correction Applied During Rollout
//!
//! Mirrors niodv4's TEDE correction format. Represents a correction vector
//! that can be applied to steering trajectories during shadow-mode or active mode.
//!
//! ## Fields
//!
//! - `correction_id`: UUID identifying this correction
//! - `correction_vector`: The correction delta (typically 3D for Vec3 steering)
//! - `gain`: Multiplicative gain factor for the correction
//! - `max_correction`: Clamp value to prevent runaway corrections
//! - `instability_threshold`: Instability signal level that triggers this correction

use serde::{Deserialize, Serialize};

/// A correction delta applied during steering rollout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionDelta {
    /// Unique UUID for this correction
    pub correction_id: String,
    /// The correction vector (typically 3D to match Vec3)
    pub correction_vector: Vec<f64>,
    /// Multiplicative gain factor [0, 1]
    pub gain: f64,
    /// Maximum correction magnitude clamp
    pub max_correction: f64,
    /// Instability threshold that triggers this correction
    pub instability_threshold: f64,
}

impl CorrectionDelta {
    /// Create a new CorrectionDelta with validated parameters.
    pub fn new(
        correction_id: String,
        correction_vector: Vec<f64>,
        gain: f64,
        max_correction: f64,
        instability_threshold: f64,
    ) -> Self {
        debug_assert!(
            (0.0..=1.0).contains(&gain),
            "Correction gain must be in [0, 1], got {}",
            gain
        );
        debug_assert!(
            max_correction > 0.0,
            "max_correction must be positive, got {}",
            max_correction
        );
        Self {
            correction_id,
            correction_vector,
            gain,
            max_correction,
            instability_threshold,
        }
    }

    /// Check if this correction should be applied given current instability.
    pub fn should_apply(&self, current_instability: f64) -> bool {
        current_instability >= self.instability_threshold
    }

    /// Apply the correction with gain and max clamp.
    pub fn apply(&self, current_vector: &[f64]) -> Vec<f64> {
        if !self.should_apply(0.0) {
            // Shadow mode: compute but don't apply
            return current_vector.to_vec();
        }

        let mut result = current_vector.to_vec();
        let dim = std::cmp::min(result.len(), self.correction_vector.len());

        for (i, &cv) in self.correction_vector.iter().enumerate().take(dim) {
            result[i] += cv * self.gain;
        }

        // Clamp to max_correction magnitude
        let magnitude = result.iter().map(|v| v * v).sum::<f64>().sqrt();
        if magnitude > self.max_correction && magnitude > 0.0 {
            let scale = self.max_correction / magnitude;
            for v in result.iter_mut() {
                *v *= scale;
            }
        }

        result
    }

    /// Get the correction magnitude.
    pub fn magnitude(&self) -> f64 {
        self.correction_vector
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correction_no_apply() {
        let correction =
            CorrectionDelta::new("corr-1".to_string(), vec![0.1, 0.2, 0.3], 0.5, 1.0, 0.8);
        assert!(!correction.should_apply(0.5));
    }

    #[test]
    fn test_correction_applies() {
        let correction =
            CorrectionDelta::new("corr-2".to_string(), vec![0.1, 0.2, 0.3], 0.5, 1.0, 0.8);
        assert!(correction.should_apply(0.9));
    }

    #[test]
    fn test_correction_magnitude() {
        let correction = CorrectionDelta::new("corr-3".to_string(), vec![3.0, 4.0], 0.5, 10.0, 0.8);
        assert_eq!(correction.magnitude(), 5.0);
    }

    #[test]
    fn test_correction_apply_with_clamp() {
        let correction = CorrectionDelta::new(
            "corr-4".to_string(),
            vec![10.0, 10.0],
            1.0,
            5.0,
            0.0, // Always applies
        );

        let input = vec![0.0, 0.0];
        let result = correction.apply(&input);

        // Should be clamped to magnitude 5.0
        let magnitude = result.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((magnitude - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_correction_serialization() {
        let correction =
            CorrectionDelta::new("corr-5".to_string(), vec![0.1, 0.2, 0.3], 0.5, 1.0, 0.8);

        let json = serde_json::to_string(&correction).expect("Serialization failed");
        let deserialized: CorrectionDelta =
            serde_json::from_str(&json).expect("Deserialization failed");

        assert_eq!(deserialized.correction_id, correction.correction_id);
        assert_eq!(deserialized.gain, correction.gain);
    }
}
