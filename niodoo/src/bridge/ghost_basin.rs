//! # Ghost Basin — 64D Vortex Attractor Artifact
//!
//! Mirrors niodv4's ghost candidate registry entry format.
//! Represents a stabilized attractor basin exported from the Python
//! ghost_candidate_registry_builder.py pipeline.
//!
//! ## Fields
//!
//! - `ghost_id`: UUID identifying this basin
//! - `coordinates`: 64D centroid vector (tail mean of rollout)
//! - `persistence_score`: [0,1] orbit stability metric
//! - `readiness_score`: [0,1] composite readiness for promotion
//! - `status`: candidate/promoted/minted/rejected
//! - `diagnostics`: flip_rate, orbit_count, energy, radius metrics

use serde::{Deserialize, Serialize};

/// A stabilized 64D vortex attractor basin exported from niodv4.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GhostBasin {
    /// Unique UUID for this ghost candidate
    pub ghost_id: String,
    /// ISO 8601 timestamp of export
    pub timestamp: String,
    /// Phase metadata (phase, rollout_steps, injection_strength)
    pub source_metadata: SourceMetadata,
    /// 64D centroid vector (mean of last 10 rollout states)
    pub coordinates: Vec<f64>,
    /// Triple-threat metrics from niodv4 scoring
    pub triple_threat_metrics: TripleThreatMetrics,
    /// CRDT metadata for eventual token promotion
    pub crdt_metadata: CrdtMetadata,
    /// Readiness summary with score and status
    pub readiness_summary: ReadinessSummary,
    /// Diagnostic metrics (flip_rate, orbit_count, etc.)
    pub diagnostics: Diagnostics,
}

/// Phase metadata from niodv4 candidate definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMetadata {
    pub phase: String,
    pub rollout_steps: i32,
    pub injection_strength: f64,
    pub candidate_name: String,
    pub expert_loaded: bool,
}

/// Triple-threat metrics from niodv4 scoring pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleThreatMetrics {
    pub persistence_score: f64,
    pub entropy_delta: f64,
    pub mean_displacement: f64,
    pub variance_wobble: f64,
}

/// CRDT metadata for eventual token promotion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrdtMetadata {
    pub replica_id: String,
    pub lamport_timestamp: i32,
    pub sequence_number: i32,
}

/// Readiness summary with composite score and status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessSummary {
    pub readiness_score: f64,
    pub status: GhostStatus,
}

/// Basin promotion status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GhostStatus {
    Candidate,
    Promoted,
    Minted,
    Rejected,
}

impl std::fmt::Display for GhostStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GhostStatus::Candidate => write!(f, "candidate"),
            GhostStatus::Promoted => write!(f, "promoted"),
            GhostStatus::Minted => write!(f, "minted"),
            GhostStatus::Rejected => write!(f, "rejected"),
        }
    }
}

/// Diagnostic metrics from niodv4 scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostics {
    pub flip_rate: f64,
    pub orbit_count: f64,
    pub max_pre_energy: f64,
    pub radius_mean: f64,
    pub radius_std: f64,
}

impl GhostBasin {
    /// Create a new GhostBasin with validated dimensions.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ghost_id: String,
        timestamp: String,
        source_metadata: SourceMetadata,
        coordinates: Vec<f64>,
        triple_threat_metrics: TripleThreatMetrics,
        crdt_metadata: CrdtMetadata,
        readiness_summary: ReadinessSummary,
        diagnostics: Diagnostics,
    ) -> Self {
        debug_assert!(
            coordinates.len() == 64,
            "Ghost basin coordinates must be exactly 64D, got {}",
            coordinates.len()
        );
        Self {
            ghost_id,
            timestamp,
            source_metadata,
            coordinates,
            triple_threat_metrics,
            crdt_metadata,
            readiness_summary,
            diagnostics,
        }
    }

    /// Check if this basin meets the minimum persistence threshold
    pub fn is_candidate(&self) -> bool {
        self.readiness_summary.status == GhostStatus::Candidate
            || self.readiness_summary.status == GhostStatus::Promoted
            || self.readiness_summary.status == GhostStatus::Minted
    }

    /// Get the 2D core coordinates (first two dimensions of 64D vector)
    pub fn core_2d(&self) -> (f64, f64) {
        if self.coordinates.len() >= 2 {
            (self.coordinates[0], self.coordinates[1])
        } else {
            (0.0, 0.0)
        }
    }

    /// Get the persistence score (primary quality metric)
    pub fn persistence_score(&self) -> f64 {
        self.triple_threat_metrics.persistence_score
    }

    /// Get the readiness score (composite metric)
    pub fn readiness_score(&self) -> f64 {
        self.readiness_summary.readiness_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ghost_basin_creation() {
        let basin = GhostBasin::new(
            "test-uuid".to_string(),
            "2025-01-01T00:00:00Z".to_string(),
            SourceMetadata {
                phase: "phase1_deep_sweep".to_string(),
                rollout_steps: 60,
                injection_strength: 0.0,
                candidate_name: "baseline_60".to_string(),
                expert_loaded: false,
            },
            vec![0.5_f64; 64],
            TripleThreatMetrics {
                persistence_score: 0.8,
                entropy_delta: 0.1,
                mean_displacement: 0.02,
                variance_wobble: 0.001,
            },
            CrdtMetadata {
                replica_id: "test-replica".to_string(),
                lamport_timestamp: 1,
                sequence_number: 1,
            },
            ReadinessSummary {
                readiness_score: 0.75,
                status: GhostStatus::Candidate,
            },
            Diagnostics {
                flip_rate: 0.02,
                orbit_count: 1.5,
                max_pre_energy: 0.3,
                radius_mean: 0.5,
                radius_std: 0.01,
            },
        );

        assert_eq!(basin.core_2d(), (0.5, 0.5));
        assert_eq!(basin.persistence_score(), 0.8);
        assert_eq!(basin.readiness_score(), 0.75);
        assert!(basin.is_candidate());
    }

    #[test]
    fn test_ghost_basin_rejected() {
        let basin = GhostBasin::new(
            "test-uuid".to_string(),
            "2025-01-01T00:00:00Z".to_string(),
            SourceMetadata {
                phase: "phase1_deep_sweep".to_string(),
                rollout_steps: 60,
                injection_strength: 0.0,
                candidate_name: "baseline_60".to_string(),
                expert_loaded: false,
            },
            vec![0.5_f64; 64],
            TripleThreatMetrics {
                persistence_score: 0.3,
                entropy_delta: 0.1,
                mean_displacement: 0.02,
                variance_wobble: 0.001,
            },
            CrdtMetadata {
                replica_id: "test-replica".to_string(),
                lamport_timestamp: 1,
                sequence_number: 1,
            },
            ReadinessSummary {
                readiness_score: 0.25,
                status: GhostStatus::Rejected,
            },
            Diagnostics {
                flip_rate: 0.02,
                orbit_count: 1.5,
                max_pre_energy: 0.3,
                radius_mean: 0.5,
                radius_std: 0.01,
            },
        );

        assert!(!basin.is_candidate());
    }

    #[test]
    fn test_ghost_basin_serialization() {
        let basin = GhostBasin::new(
            "test-uuid".to_string(),
            "2025-01-01T00:00:00Z".to_string(),
            SourceMetadata {
                phase: "phase1_deep_sweep".to_string(),
                rollout_steps: 60,
                injection_strength: 0.0,
                candidate_name: "baseline_60".to_string(),
                expert_loaded: false,
            },
            vec![0.5_f64; 64],
            TripleThreatMetrics {
                persistence_score: 0.8,
                entropy_delta: 0.1,
                mean_displacement: 0.02,
                variance_wobble: 0.001,
            },
            CrdtMetadata {
                replica_id: "test-replica".to_string(),
                lamport_timestamp: 1,
                sequence_number: 1,
            },
            ReadinessSummary {
                readiness_score: 0.75,
                status: GhostStatus::Candidate,
            },
            Diagnostics {
                flip_rate: 0.02,
                orbit_count: 1.5,
                max_pre_energy: 0.3,
                radius_mean: 0.5,
                radius_std: 0.01,
            },
        );

        let json = serde_json::to_string(&basin).expect("Serialization failed");
        let deserialized: GhostBasin = serde_json::from_str(&json).expect("Deserialization failed");

        assert_eq!(deserialized.ghost_id, basin.ghost_id);
        assert_eq!(deserialized.core_2d(), basin.core_2d());
        assert_eq!(deserialized.persistence_score(), basin.persistence_score());
    }
}
