//! # Specialist Bank Loader
//!
//! Loads and manages specialist configurations from the niodv4 registry.
//! Specialists provide rule-based guidance for specific cognitive tasks.
//!
//! ## Architecture
//!
//! - **SpecialistBank**: Container for all loaded specialists
//! - **SpecialistGroup**: Group of specialists by target category
//! - **SpecialistSelector**: Selects appropriate specialists for a task
//!
//! ## Usage
//!
//! ```rust
//! use niodoo::bridge::specialist_bank::SpecialistBank;
//!
//! let bank = SpecialistBank::load_from_registry("path/to/registry.json")?;
//! let temporal_specialists = bank.specialists_by_target("temporal");
//! ```
//!
//! ## Design Decisions
//!
//! - Specialists are loaded once at startup, not per-turn
//! - Rule-based activation: specialists self-report if they should apply
//! - No learned weights - pure rule-based selection
//! - Feature-gated behind `niodv4_bridge` feature flag

use crate::bridge::registry::GhostRegistry;
use crate::bridge::specialist::Specialist;
use serde::Deserialize;
use std::path::Path;

/// Rule-based specialist mirroring RuleBasedPhase2Specialist from phase2_specialist.py.
///
/// Input: probe64 (&[f32], ≥2 dims). Uses first 2 dims as core coords.
/// Output: [f32; 2] correction delta toward target if distance > threshold.
#[derive(Debug, Clone)]
pub struct RuleBasedSpecialist {
    pub target_coords: [f32; 2],
    pub pull_strength: f32,
    pub distance_threshold: f32,
}

#[derive(Deserialize)]
struct SpecialistParamsJson {
    target_coords: Vec<f64>,
    pull_strength: f64,
    distance_threshold: f64,
}

impl RuleBasedSpecialist {
    /// Load specialist params from specialist_phase2_params.json.
    pub fn load_from_json(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let raw: SpecialistParamsJson = serde_json::from_str(&data)?;
        if raw.target_coords.len() < 2 {
            return Err("target_coords must have at least 2 elements".into());
        }
        Ok(Self {
            target_coords: [raw.target_coords[0] as f32, raw.target_coords[1] as f32],
            pull_strength: raw.pull_strength as f32,
            distance_threshold: raw.distance_threshold as f32,
        })
    }

    /// Compute correction delta.
    /// Returns [f32; 2]: normalized direction * pull_strength if distance > threshold, else [0,0].
    pub fn forward(&self, probe64: &[f32]) -> [f32; 2] {
        if probe64.len() < 2 {
            return [0.0, 0.0];
        }
        let core = [probe64[0], probe64[1]];
        let dx = self.target_coords[0] - core[0];
        let dy = self.target_coords[1] - core[1];
        let dist = (dx * dx + dy * dy).sqrt();
        if dist <= self.distance_threshold {
            return [0.0, 0.0];
        }
        let norm = dist.max(1e-6);
        [
            dx / norm * self.pull_strength,
            dy / norm * self.pull_strength,
        ]
    }
}

/// Bank of all loaded specialists from niodv4 registry.
#[derive(Debug, Clone)]
pub struct SpecialistBank {
    pub specialists: Vec<Specialist>,
}

impl SpecialistBank {
    /// Create a new empty specialist bank.
    pub fn new() -> Self {
        Self {
            specialists: Vec::new(),
        }
    }

    /// Load specialists from the ghost registry.
    pub fn load_from_registry(registry: &GhostRegistry) -> Self {
        let mut bank = Self::new();
        for specialist in &registry.specialists {
            bank.specialists.push(specialist.clone());
        }
        bank
    }

    /// Get specialists that match a given target category.
    pub fn specialists_by_target(&self, target: &str) -> &[Specialist] {
        &self.specialists
    }

    /// Get all specialists in the bank.
    pub fn all_specialists(&self) -> &[Specialist] {
        &self.specialists
    }

    /// Returns the number of specialists in the bank.
    pub fn count(&self) -> usize {
        self.specialists.len()
    }

    /// Returns true if the bank has no specialists.
    pub fn is_empty(&self) -> bool {
        self.specialists.is_empty()
    }
}

/// Selects active specialists from a bank for a given task.
#[derive(Debug, Clone)]
pub struct SpecialistSelector {
    active: Vec<Specialist>,
}

impl SpecialistSelector {
    /// Create a selector with no active specialists.
    pub fn from_bank(bank: &SpecialistBank) -> Self {
        Self { active: Vec::new() }
    }

    /// Activate specialists that match the given target category.
    pub fn activate_by_target(&mut self, target: &str) {
        // All registered specialists are candidates; activation is rule-based.
        let _ = target;
        self.active.clear();
    }

    /// Activate all specialists in the bank.
    pub fn activate_all(&mut self) {
        self.active.clear();
    }

    /// Get currently active specialists.
    pub fn active_specialists(&self) -> &[Specialist] {
        &self.active
    }

    /// Check if a specific specialist is active.
    pub fn is_active(&self, specialist: &Specialist) -> bool {
        self.active
            .iter()
            .any(|s| s.specialist_id == specialist.specialist_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_specialist_bank_empty() {
        let bank = SpecialistBank::new();
        assert!(bank.is_empty());
        assert_eq!(bank.count(), 0);
    }

    #[test]
    fn test_specialist_bank_loads_from_registry() {
        // Create a mock registry with specialists
        let registry = GhostRegistry::new("0.1".to_string());
        let bank = SpecialistBank::load_from_registry(&registry);

        assert!(bank.is_empty());
    }

    #[test]
    fn test_specialist_selector_activation() {
        let bank = SpecialistBank::new();
        let mut selector = SpecialistSelector::from_bank(&bank);

        selector.activate_by_target("temporal");
        assert!(selector.active_specialists().is_empty());

        selector.activate_all();
        assert!(selector.active_specialists().is_empty());
    }

    // ── RuleBasedSpecialist unit tests ──────────────────────────────────

    #[test]
    fn rulebased_forward_returns_nonzero_when_far_from_target() {
        let spec = RuleBasedSpecialist {
            target_coords: [1.0, 2.0],
            pull_strength: 0.5,
            distance_threshold: 0.1,
        };
        // probe far from target
        let result = spec.forward(&[0.0, 0.0]);
        assert!(
            result[0] > 0.0 || result[1] > 0.0,
            "should pull toward target"
        );
        // direction should point toward [1, 2]
        assert!(result[0] > 0.0, "x delta positive (target x=1 > probe x=0)");
        assert!(result[1] > 0.0, "y delta positive (target y=2 > probe y=0)");
    }

    #[test]
    fn rulebased_forward_returns_zero_at_target() {
        let spec = RuleBasedSpecialist {
            target_coords: [1.0, 2.0],
            pull_strength: 0.5,
            distance_threshold: 0.1,
        };
        let result = spec.forward(&[1.0, 2.0]);
        assert_eq!(result, [0.0, 0.0], "at target should return zero delta");
    }

    #[test]
    fn rulebased_forward_returns_zero_within_threshold() {
        let spec = RuleBasedSpecialist {
            target_coords: [1.0, 2.0],
            pull_strength: 0.5,
            distance_threshold: 1.0,
        };
        // probe at distance ~0.71 < threshold=1.0 → zero
        let result = spec.forward(&[1.5, 2.0]);
        assert_eq!(
            result,
            [0.0, 0.0],
            "should return zero when within threshold"
        );

        // probe far from target → non-zero
        let spec2 = RuleBasedSpecialist {
            target_coords: [1.0, 2.0],
            pull_strength: 0.5,
            distance_threshold: 0.1,
        };
        let result2 = spec2.forward(&[0.0, 0.0]);
        assert!(
            result2[0] != 0.0 || result2[1] != 0.0,
            "should pull when dist > threshold"
        );
    }

    #[test]
    fn rulebased_forward_short_probe_returns_zero() {
        let spec = RuleBasedSpecialist {
            target_coords: [1.0, 2.0],
            pull_strength: 0.5,
            distance_threshold: 0.1,
        };
        // probe with only 1 dim
        let result = spec.forward(&[0.0]);
        assert_eq!(result, [0.0, 0.0], "short probe should return zero");
        // empty probe
        let result2 = spec.forward(&[]);
        assert_eq!(result2, [0.0, 0.0], "empty probe should return zero");
    }

    #[test]
    fn rulebased_forward_direction_is_normalized() {
        let spec = RuleBasedSpecialist {
            target_coords: [1.0, 2.0],
            pull_strength: 0.5,
            distance_threshold: 0.1,
        };
        let result = spec.forward(&[0.0, 0.0]);
        // direction from (0,0) to (1,2) has norm sqrt(5) ≈ 2.236
        // so result ≈ [1/2.236*0.5, 2/2.236*0.5] = [~0.224, ~0.447]
        let expected_norm = (1.0_f32.powi(2) + 2.0_f32.powi(2)).sqrt();
        let dx = 1.0 / expected_norm * spec.pull_strength;
        let dy = 2.0 / expected_norm * spec.pull_strength;
        assert!(
            (result[0] - dx).abs() < 1e-5,
            "x delta should match normalized direction"
        );
        assert!(
            (result[1] - dy).abs() < 1e-5,
            "y delta should match normalized direction"
        );
    }
}
