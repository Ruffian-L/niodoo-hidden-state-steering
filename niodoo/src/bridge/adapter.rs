//! # Bridge Adapter — niodv4 to Rust Format Converter
//!
//! This module provides adapters for converting niodv4 JSON artifacts
//! into Rust-native GhostRegistry format for physics engine integration.
//!
//! ## Current Adapters
//!
//! - **GhostRegistryAdapter**: Converts niodv4 ghost_candidate_registry.json
//!
//! ## Usage
//!
//! ```rust
//! use niodoo::bridge::adapter::GhostRegistryAdapter;
//! use niodoo::bridge::registry::GhostRegistry;
//!
//! // Load and convert niodv4 registry
//! let adapter = GhostRegistryAdapter::new();
//! let registry: GhostRegistry = adapter.load_from_path("niodv4/data/results/summaries/ghost_candidate_registry.json")?;
//! ```
//!
//! ## Future Adapters (Planned)
//!
//! - **SpecialistBankAdapter**: Convert niodv4 specialist bank JSON
//! - **HiddenStateAdapter**: Convert captured hidden state tensors
//! - **CodecShadowAdapter**: Convert codec shadow mode configurations

use crate::bridge::registry::{GhostBasin, GhostRegistry};
use serde_json::Value;
use std::fs;
use std::path::Path;

/// Adapter for converting niodv4 ghost_candidate_registry.json to Rust format
pub struct GhostRegistryAdapter;

impl GhostRegistryAdapter {
    /// Create a new adapter instance
    pub fn new() -> Self {
        GhostRegistryAdapter
    }

    /// Load and convert niodv4 ghost registry from file path
    pub fn load_from_path<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<GhostRegistry, Box<dyn std::error::Error>> {
        let data = fs::read_to_string(path)?;
        self.load_from_str(&data)
    }

    /// Load and convert niodv4 ghost registry from JSON string
    pub fn load_from_str(
        &self,
        json_str: &str,
    ) -> Result<GhostRegistry, Box<dyn std::error::Error>> {
        let value: Value = serde_json::from_str(json_str)?;

        // Parse version
        let version = value
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("1.0")
            .to_string();

        // Parse timestamp
        let _created_at = value
            .get("created_at")
            .and_then(|v| v.as_str())
            .unwrap_or("1970-01-01T00:00:00Z")
            .to_string();

        // Parse basins array
        let basins: Vec<serde_json::Value> = value
            .get("basins")
            .and_then(|v| v.as_array())
            .map(|arr| arr.to_vec())
            .unwrap_or_default();
        let parsed_basins: Vec<_> = basins
            .iter()
            .map(|b| self.parse_basin(b))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(GhostRegistry::new(version)).map(|mut r| {
            r.basins = parsed_basins;
            r
        })
    }

    fn parse_basin(&self, value: &Value) -> Result<GhostBasin, Box<dyn std::error::Error>> {
        let id = value
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'id' field")?
            .to_string();

        let dimension = value
            .get("dimension")
            .and_then(|v| v.as_u64())
            .ok_or("Missing 'dimension' field")? as usize;

        let data = value
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or("Missing 'data' field")?
            .iter()
            .map(|v| v.as_f64().ok_or("Invalid data value").map(|f| f as f32))
            .collect::<Result<Vec<_>, _>>()?;

        let metadata = value.get("metadata").cloned();

        Ok(GhostBasin::with_metadata(
            id,
            dimension,
            data,
            metadata.unwrap_or_default(),
        ))
    }
}

impl Default for GhostRegistryAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_JSON: &str = r#"
    {
        "version": "1.0",
        "created_at": "2025-01-01T00:00:00Z",
        "basins": [
            {
                "id": "ghost_001",
                "dimension": 64,
                "data": [0.1, 0.2, 0.3, 0.4, 0.5],
                "metadata": {"source": "test"}
            }
        ]
    }
    "#;

    #[test]
    fn test_load_from_str() {
        let adapter = GhostRegistryAdapter::new();
        let result = adapter.load_from_str(SAMPLE_JSON);

        assert!(result.is_ok());
        let registry = result.unwrap();

        assert_eq!(registry.version, "1.0");
        assert_eq!(registry.basin_count(), 1);

        let basin = &registry.basins[0];
        assert_eq!(basin.id, "ghost_001");
        assert_eq!(basin.dimension, 64);
        assert_eq!(basin.data.len(), 5);
    }

    #[test]
    fn test_missing_id_field() {
        let adapter = GhostRegistryAdapter::new();
        let bad_json = r#"
        {
            "version": "1.0",
            "created_at": "2025-01-01T00:00:00Z",
            "basins": [
                {
                    "dimension": 64,
                    "data": [0.1, 0.2]
                }
            ]
        }
        "#;

        let result = adapter.load_from_str(bad_json);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_data_field() {
        let adapter = GhostRegistryAdapter::new();
        let bad_json = r#"
        {
            "version": "1.0",
            "created_at": "2025-01-01T00:00:00Z",
            "basins": [
                {
                    "id": "test",
                    "dimension": 64
                }
            ]
        }
        "#;

        let result = adapter.load_from_str(bad_json);
        assert!(result.is_err());
    }
}
