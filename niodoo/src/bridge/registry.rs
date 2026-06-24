//! # GhostRegistry — Container for All Ghost Basins from niodv4 Export
//!
//! This module provides the Rust representation of niodv4's ghost_candidate_registry.json
//! format, along with loading and conversion utilities.
//!
//! ## Example
//!
//! ```rust
//! use niodoo::bridge::registry::GhostRegistry;
//!
//! // Load from niodv4 export
//! let registry = GhostRegistry::load_from_path("niodv4/data/results/summaries/ghost_candidate_registry.json")?;
//!
//! // Access ghost basins
//! for basin in &registry.basins {
//!     println!("Ghost {}: {}D vector", basin.id, basin.data.len());
//! }
//! ```
//!
//! ## JSON Schema (from niodv4)
//!
//! ```json
//! {
//!   "version": "1.0",
//!   "created_at": "2025-01-01T00:00:00Z",
//!   "basins": [
//!     {
//!       "id": "ghost_001",
//!       "dimension": 64,
//!       "data": [0.1, 0.2, ..., 0.9],
//!       "metadata": {
//!         "source": "niodv4",
//!         "created_at": "2025-01-01T00:00:00Z"
//!       }
//!     }
//!   ]
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Top-level registry container matching niodv4's ghost_candidate_registry.json structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GhostRegistry {
    /// Registry version
    #[serde(rename = "registry_version")]
    pub version: String,
    /// Persistence threshold used during construction
    pub persistence_threshold: f64,
    /// Total number of entries
    pub entry_count: usize,
    /// List of ghost basin entries
    #[serde(rename = "entries")]
    pub basins: Vec<GhostBasin>,
    /// Collection of specialists (optional)
    #[serde(default)]
    pub specialists: Vec<crate::bridge::specialist::Specialist>,
}

impl GhostRegistry {
    /// Create a new empty registry
    pub fn new(version: String) -> Self {
        GhostRegistry {
            version,
            persistence_threshold: 0.0,
            entry_count: 0,
            basins: Vec::new(),
            specialists: Vec::new(),
        }
    }

    /// Load registry from a JSON file path
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let data = fs::read_to_string(path)?;
        let registry: GhostRegistry = serde_json::from_str(&data)?;
        Ok(registry)
    }

    /// Load registry from a JSON string
    pub fn load_from_str(data: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let registry: GhostRegistry = serde_json::from_str(data)?;
        Ok(registry)
    }

    /// Export registry to JSON string
    pub fn export_json(&self) -> Result<String, Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        Ok(json)
    }

    /// Export registry to file
    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json = self.export_json()?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Find a basin by ID
    pub fn find_basin(&self, id: &str) -> Option<&GhostBasin> {
        self.basins.iter().find(|b| b.id == id)
    }

    /// Get all basin IDs
    pub fn basin_ids(&self) -> Vec<&str> {
        self.basins.iter().map(|b| b.id.as_str()).collect()
    }

    /// Get basin count
    pub fn basin_count(&self) -> usize {
        self.basins.len()
    }

    /// Get candidate count (alias for basin_count for niodv4 compatibility)
    pub fn candidate_count(&self) -> usize {
        self.basins.len()
    }

    /// Check whether the registry has any candidate basins.
    pub fn has_candidates(&self) -> bool {
        !self.basins.is_empty()
    }

    /// Get all candidate basins (non-rejected).
    pub fn candidate_basins(&self) -> impl Iterator<Item = &GhostBasin> {
        self.basins.iter()
    }

    /// Motifs are not present in the current ghost registry export.
    pub fn motif_count(&self) -> usize {
        0
    }

    /// Get specialist count from the optional specialists array.
    pub fn specialist_count(&self) -> usize {
        self.specialists.len()
    }

    /// Corrections are not present in the current ghost registry export.
    pub fn correction_count(&self) -> usize {
        0
    }

    /// Get specialists by target category from the optional specialists array.
    pub fn specialists_by_target(
        &self,
        target: &str,
    ) -> Vec<&crate::bridge::specialist::Specialist> {
        self.specialists
            .iter()
            .filter(|specialist| specialist.target_category() == target)
            .collect()
    }

    /// Current ghost registry exports do not include active corrections.
    pub fn active_corrections(&self, _threshold: f64) -> Vec<&GhostBasin> {
        Vec::new()
    }
}

/// Represents a single ghost basin (64D vector) from niodv4
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GhostBasin {
    /// Unique identifier for this ghost
    #[serde(rename = "ghost_id")]
    pub id: String,
    /// Vector dimension (typically 64)
    #[serde(default)]
    pub dimension: usize,
    /// The actual 64D vector data
    #[serde(rename = "coordinates")]
    pub data: Vec<f32>,
    /// Additional metadata from niodv4
    pub metadata: Option<serde_json::Value>,
}

impl GhostBasin {
    /// Create a new ghost basin
    pub fn new(id: String, dimension: usize, data: Vec<f32>) -> Self {
        GhostBasin {
            id,
            dimension,
            data,
            metadata: None,
        }
    }

    /// Create a ghost basin with metadata
    pub fn with_metadata(
        id: String,
        dimension: usize,
        data: Vec<f32>,
        metadata: serde_json::Value,
    ) -> Self {
        GhostBasin {
            id,
            dimension,
            data,
            metadata: Some(metadata),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_empty_registry() {
        let registry = GhostRegistry::new("1.0".to_string());
        assert_eq!(registry.version, "1.0");
        assert_eq!(registry.basin_count(), 0);
    }

    #[test]
    fn test_add_basin() {
        let mut registry = GhostRegistry::new("1.0".to_string());
        let basin = GhostBasin::new("test_001".to_string(), 64, vec![0.0; 64]);
        registry.basins.push(basin);
        assert_eq!(registry.basin_count(), 1);
    }

    #[test]
    fn test_find_basin() {
        let mut registry = GhostRegistry::new("1.0".to_string());
        let basin = GhostBasin::new("target_001".to_string(), 64, vec![0.0; 64]);
        registry.basins.push(basin);

        let found = registry.find_basin("target_001");
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "target_001");
    }

    #[test]
    fn test_find_nonexistent_basin() {
        let registry = GhostRegistry::new("1.0".to_string());
        let found = registry.find_basin("nonexistent");
        assert!(found.is_none());
    }

    #[test]
    fn test_basin_ids() {
        let mut registry = GhostRegistry::new("1.0".to_string());
        registry
            .basins
            .push(GhostBasin::new("a".to_string(), 64, vec![0.0; 64]));
        registry
            .basins
            .push(GhostBasin::new("b".to_string(), 64, vec![0.0; 64]));
        registry
            .basins
            .push(GhostBasin::new("c".to_string(), 64, vec![0.0; 64]));

        let ids = registry.basin_ids();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
    }
}
