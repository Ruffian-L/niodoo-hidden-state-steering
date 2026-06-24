//! # Specialist — Rule-Based Specialist with Target, Onset, Pull, Threshold
//!
//! Mirrors niodv4's specialist bank format. Each specialist defines a
//! rule-based scoring function that can be combined into consensus retrieval.
//!
//! ## Fields
//!
//! - `specialist_id`: UUID identifying this specialist
//! - `name`: Human-readable name
//! - `target`: What the specialist looks for (e.g., "temporal", "semantic")
//! - `onset`: When the specialist activates (threshold)
//! - `pull`: Attractor/repulsor strength when activated
//! - `threshold`: Minimum score to activate

use serde::{Deserialize, Serialize};

/// A rule-based specialist that scores memory nodes for retrieval.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Specialist {
    /// Unique UUID for this specialist
    pub specialist_id: String,
    /// Human-readable name
    pub name: String,
    /// What this specialist targets (e.g., "temporal", "semantic", "topological")
    pub target: String,
    /// Onset threshold — minimum score to activate
    pub onset: f64,
    /// Pull strength when activated [-1, 1]
    pub pull: f64,
    /// Activation threshold for scoring
    pub threshold: f64,
}

impl Specialist {
    /// Create a new Specialist with validated parameters.
    pub fn new(
        specialist_id: String,
        name: String,
        target: String,
        onset: f64,
        pull: f64,
        threshold: f64,
    ) -> Self {
        debug_assert!(
            pull.abs() <= 1.0,
            "Specialist pull must be in [-1, 1], got {}",
            pull
        );
        debug_assert!(
            (0.0..=1.0).contains(&onset),
            "Specialist onset must be in [0, 1], got {}",
            onset
        );
        debug_assert!(
            (0.0..=1.0).contains(&threshold),
            "Specialist threshold must be in [0, 1], got {}",
            threshold
        );
        Self {
            specialist_id,
            name,
            target,
            onset,
            pull,
            threshold,
        }
    }

    /// Check if this specialist is activated for a given raw score.
    pub fn is_activated(&self, raw_score: f64) -> bool {
        raw_score >= self.threshold
    }

    /// Get the weighted score when activated, 0 otherwise.
    pub fn weighted_score(&self, raw_score: f64) -> f64 {
        if self.is_activated(raw_score) {
            self.pull * raw_score
        } else {
            0.0
        }
    }

    /// Get the specialist's target category.
    pub fn target_category(&self) -> &str {
        &self.target
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_specialist_activated() {
        let spec = Specialist::new(
            "spec-1".to_string(),
            "temporal".to_string(),
            "temporal".to_string(),
            0.3,
            0.8,
            0.5,
        );
        assert!(spec.is_activated(0.6));
        assert!(!spec.is_activated(0.4));
    }

    #[test]
    fn test_specialist_weighted_score() {
        let spec = Specialist::new(
            "spec-2".to_string(),
            "semantic".to_string(),
            "semantic".to_string(),
            0.3,
            0.8,
            0.5,
        );
        // Activated: score * pull = 0.6 * 0.8 = 0.48
        assert_eq!(spec.weighted_score(0.6), 0.48);
        // Not activated: 0.0
        assert_eq!(spec.weighted_score(0.4), 0.0);
    }

    #[test]
    fn test_specialist_repulsor() {
        let spec = Specialist::new(
            "spec-3".to_string(),
            "repulsive".to_string(),
            "topological".to_string(),
            0.3,
            -0.6,
            0.5,
        );
        assert_eq!(spec.weighted_score(0.7), -0.42); // 0.7 * -0.6
    }

    #[test]
    fn test_specialist_serialization() {
        let spec = Specialist::new(
            "spec-4".to_string(),
            "temporal".to_string(),
            "temporal".to_string(),
            0.3,
            0.8,
            0.5,
        );

        let json = serde_json::to_string(&spec).expect("Serialization failed");
        let deserialized: Specialist = serde_json::from_str(&json).expect("Deserialization failed");

        assert_eq!(deserialized.specialist_id, spec.specialist_id);
        assert_eq!(deserialized.name, spec.name);
        assert_eq!(deserialized.pull, spec.pull);
    }
}
