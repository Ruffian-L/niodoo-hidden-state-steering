//! # Bridge Startup Logger
//!
//! Provides startup logging for bridge artifact loading verification.
//! This module logs the exact state of bridge components when the
//! niodv4_bridge feature is enabled.
//!
//! ## Usage
//!
//! ```rust
//! use niodoo::bridge::startup_log::BridgeStartupLogger;
//!
//! let logger = BridgeStartupLogger::new();
//! logger.log_startup();
//! ```
//!
//! ## What Gets Logged
//!
//! When `niodv4_bridge` feature is enabled:
//! - `bridge_enabled=true`
//! - `ghost_registry_loaded=true/false`
//! - `ghost_basins_loaded=N`
//! - `specialist_bank_loaded=true/false`
//! - `specialists_loaded=N`
//! - `projection_strategy=<name>`
//!
//! When feature is disabled:
//! - `bridge_enabled=false`
//! - All other fields omitted

use crate::bridge::registry::GhostRegistry;
use crate::bridge::specialist_bank::SpecialistBank;

/// Logger that records bridge startup state.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "niodv4_bridge", derive(Default))]
pub struct BridgeStartupLogger {
    /// Whether the bridge feature is enabled
    pub bridge_enabled: bool,
    /// Whether the ghost registry was successfully loaded
    pub ghost_registry_loaded: bool,
    /// Number of ghost basins loaded
    pub ghost_basins_loaded: usize,
    /// Whether the specialist bank was successfully loaded
    pub specialist_bank_loaded: bool,
    /// Number of specialists loaded
    pub specialists_loaded: usize,
    /// Projection strategy in use
    pub projection_strategy: String,
}

impl BridgeStartupLogger {
    /// Create a new logger with default (disabled) state.
    pub fn new() -> Self {
        Self {
            bridge_enabled: false,
            ghost_registry_loaded: false,
            ghost_basins_loaded: 0,
            specialist_bank_loaded: false,
            specialists_loaded: 0,
            projection_strategy: "none".to_string(),
        }
    }

    /// Create a logger with full bridge state (feature enabled).
    pub fn with_bridge() -> Self {
        let mut logger = Self::new();
        logger.bridge_enabled = true;

        // Try to load registry
        let registry_path = "niodv4/data/results/summaries/ghost_candidate_registry.json";
        match GhostRegistry::load_from_path(registry_path) {
            Ok(registry) => {
                logger.ghost_registry_loaded = true;
                logger.ghost_basins_loaded = registry.basin_count();

                // Load specialist bank
                let bank = SpecialistBank::load_from_registry(&registry);
                logger.specialist_bank_loaded = !bank.is_empty();
                logger.specialists_loaded = bank.count();

                // Set projection strategy
                logger.projection_strategy = "simple".to_string();
            }
            Err(e) => {
                eprintln!(
                    "Bridge startup warning: failed to load registry from {}: {}",
                    registry_path, e
                );
                logger.ghost_registry_loaded = false;
            }
        }

        logger
    }

    /// Log startup state to stderr so token-streaming stdout stays clean.
    pub fn log_startup(&self) {
        eprintln!("\n=== BRIDGE STARTUP LOG ===");
        eprintln!("bridge_enabled={}", self.bridge_enabled);

        if self.bridge_enabled {
            eprintln!("ghost_registry_loaded={}", self.ghost_registry_loaded);
            eprintln!("ghost_basins_loaded={}", self.ghost_basins_loaded);
            eprintln!("specialist_bank_loaded={}", self.specialist_bank_loaded);
            eprintln!("specialists_loaded={}", self.specialists_loaded);
            eprintln!("projection_strategy={}", self.projection_strategy);
        }

        eprintln!("==========================\n");
    }

    /// Save startup state to a log file.
    pub fn save_log_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut content = String::new();
        content.push_str("\n=== BRIDGE STARTUP LOG ===\n");
        content.push_str(&format!("bridge_enabled={}\n", self.bridge_enabled));

        if self.bridge_enabled {
            content.push_str(&format!(
                "ghost_registry_loaded={}\n",
                self.ghost_registry_loaded
            ));
            content.push_str(&format!(
                "ghost_basins_loaded={}\n",
                self.ghost_basins_loaded
            ));
            content.push_str(&format!(
                "specialist_bank_loaded={}\n",
                self.specialist_bank_loaded
            ));
            content.push_str(&format!("specialists_loaded={}\n", self.specialists_loaded));
            content.push_str(&format!(
                "projection_strategy={}\n",
                self.projection_strategy
            ));
        }

        content.push_str("==========================\n");

        std::fs::write(path, content)?;
        Ok(())
    }

    /// Log startup state as JSON for programmatic consumption.
    pub fn log_startup_json(&self) {
        let json = serde_json::json!({
            "bridge_enabled": self.bridge_enabled,
            "ghost_registry_loaded": self.ghost_registry_loaded,
            "ghost_basins_loaded": self.ghost_basins_loaded,
            "specialist_bank_loaded": self.specialist_bank_loaded,
            "specialists_loaded": self.specialists_loaded,
            "projection_strategy": self.projection_strategy
        });
        eprintln!("{}", serde_json::to_string_pretty(&json).unwrap());
    }
}

/// Log bridge startup state based on feature flag.
///
/// When niodv4_bridge feature is enabled:
/// - Attempts to load bridge artifacts
/// - Reports exact load status and counts
///
/// When feature is disabled:
/// - Reports bridge_enabled=false
/// - All other fields omitted
pub fn log_bridge_startup() {
    #[cfg(feature = "niodv4_bridge")]
    {
        let logger = BridgeStartupLogger::with_bridge();
        logger.log_startup();
        logger.log_startup_json();
        let _ = logger.save_log_file("artifacts/bridge_startup_log.txt");
    }

    #[cfg(not(feature = "niodv4_bridge"))]
    {
        let logger = BridgeStartupLogger::new();
        logger.log_startup();
        let _ = logger.save_log_file("artifacts/bridge_startup_log.txt");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logger_default_state() {
        let logger = BridgeStartupLogger::new();
        assert!(!logger.bridge_enabled);
        assert!(!logger.ghost_registry_loaded);
        assert_eq!(logger.ghost_basins_loaded, 0);
        assert!(!logger.specialist_bank_loaded);
        assert_eq!(logger.specialists_loaded, 0);
        assert_eq!(logger.projection_strategy, "none");
    }

    #[test]
    fn test_logger_with_bridge() {
        let logger = BridgeStartupLogger::with_bridge();
        assert!(logger.bridge_enabled);
        // Registry may or may not exist depending on test environment
        // Just verify the structure is correct
        let _ = logger.ghost_registry_loaded;
        let _ = logger.ghost_basins_loaded;
        let _ = logger.specialist_bank_loaded;
        let _ = logger.specialists_loaded;
        assert!(logger.projection_strategy != "none");
    }
}
