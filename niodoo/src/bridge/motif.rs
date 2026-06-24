//! # Motif — Phase Metadata and Candidate Parameters
//!
//! Mirrors niodv4's motif registry entry format. Represents a phase-level
//! attractor pattern that drives Möbius retrieval dynamics in niodoo.
//!
//! ## Fields
//!
//! - `motif_id`: UUID identifying this motif
//! - `phase`: Phase label (e.g., "phase1_deep_sweep")
//! - `pull_strength`: Attractor/repulsor magnitude for steering
//! - `candidate_params`: Parameters that define the motif's behavior

use serde::{Deserialize, Serialize};

/// A motif representing a recurring phase-level attractor pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Motif {
    /// Unique UUID for this motif
    pub motif_id: String,
    /// Phase label from niodv4 registry
    pub phase: String,
    /// Pull strength [-1, 1]: positive = attractor, negative = repulsor
    pub pull_strength: f64,
    /// Candidate parameters defining motif behavior
    pub candidate_params: MotifParams,
}

/// Parameters that define a motif's behavioral profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifParams {
    /// Number of rollout steps this motif was observed over
    pub rollout_steps: i32,
    /// Injection strength used during observation
    pub injection_strength: f64,
    /// Whether an expert model was loaded during observation
    pub expert_loaded: bool,
    /// Candidate name from niodv4 naming convention
    pub candidate_name: String,
}

impl Motif {
    /// Create a new Motif with validated pull strength range.
    pub fn new(
        motif_id: String,
        phase: String,
        pull_strength: f64,
        rollout_steps: i32,
        injection_strength: f64,
        expert_loaded: bool,
        candidate_name: String,
    ) -> Self {
        debug_assert!(
            pull_strength.abs() <= 1.0,
            "Motif pull_strength must be in [-1, 1], got {}",
            pull_strength
        );
        Self {
            motif_id,
            phase,
            pull_strength,
            candidate_params: MotifParams {
                rollout_steps,
                injection_strength,
                expert_loaded,
                candidate_name,
            },
        }
    }

    /// Check if this motif is an attractor (positive pull).
    pub fn is_attractor(&self) -> bool {
        self.pull_strength > 0.0
    }

    /// Check if this motif is a repulsor (negative pull).
    pub fn is_repulsor(&self) -> bool {
        self.pull_strength < 0.0
    }

    /// Get the absolute pull magnitude.
    pub fn pull_magnitude(&self) -> f64 {
        self.pull_strength.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motif_attractor() {
        let motif = Motif::new(
            "motif-1".to_string(),
            "phase1".to_string(),
            0.7,
            60,
            0.0,
            false,
            "baseline_60".to_string(),
        );
        assert!(motif.is_attractor());
        assert!(!motif.is_repulsor());
        assert_eq!(motif.pull_magnitude(), 0.7);
    }

    #[test]
    fn test_motif_repulsor() {
        let motif = Motif::new(
            "motif-2".to_string(),
            "phase1".to_string(),
            -0.5,
            60,
            0.0,
            false,
            "baseline_60".to_string(),
        );
        assert!(!motif.is_attractor());
        assert!(motif.is_repulsor());
        assert_eq!(motif.pull_magnitude(), 0.5);
    }

    #[test]
    fn test_motif_zero_pull() {
        let motif = Motif::new(
            "motif-3".to_string(),
            "phase1".to_string(),
            0.0,
            60,
            0.0,
            false,
            "baseline_60".to_string(),
        );
        assert!(!motif.is_attractor());
        assert!(!motif.is_repulsor());
        assert_eq!(motif.pull_magnitude(), 0.0);
    }

    #[test]
    fn test_motif_serialization() {
        let motif = Motif::new(
            "motif-4".to_string(),
            "phase1".to_string(),
            0.3,
            60,
            0.0,
            false,
            "baseline_60".to_string(),
        );

        let json = serde_json::to_string(&motif).expect("Serialization failed");
        let deserialized: Motif = serde_json::from_str(&json).expect("Deserialization failed");

        assert_eq!(deserialized.motif_id, motif.motif_id);
        assert_eq!(deserialized.pull_strength, motif.pull_strength);
    }
}
