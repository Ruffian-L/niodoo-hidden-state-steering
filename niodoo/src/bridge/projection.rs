//! # 64D → Vec3 Projection Stub
//!
//! Experimental projection of 64D ghost basins onto Vec3 steering space.
//! This is a placeholder implementation for future refinement.
//!
//! ## Design Decisions
//!
//! - **Feature-gated**: Only available with `niodv4_bridge` feature
//! - **Experimental**: Not production-ready, subject to change
//! - **Simple projection**: Uses first 3 dimensions as initial approximation
//! - **Configurable**: Allows different projection strategies
//!
//! ## Usage
//!
//! ```rust
//! use niodoo::bridge::projection::Projector;
//! use niodoo::bridge::ghost_basin::GhostBasin;
//!
//! let projector = Projector::new();
//! let vec3 = projector.project_to_vec3(&basin);
//! ```
//!
//! ## Future Work
//!
//! - PCA-based dimensionality reduction
//! - UMAP/t-SNE projections for visualization
//! - Learned projection weights from training data
//! - Multi-scale projection (global + local)

use crate::bridge::ghost_basin::GhostBasin;

// Local type definitions to avoid dependency on crate::types
pub type Point3 = [f32; 3];
pub type Vec3 = [f32; 3];

/// Projection configuration.
#[derive(Debug, Clone)]
pub struct ProjectionConfig {
    /// Number of dimensions to project to (default: 3)
    pub output_dim: usize,
    /// Projection strategy
    pub strategy: ProjectionStrategy,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            output_dim: 3,
            strategy: ProjectionStrategy::Simple,
        }
    }
}

/// Projection strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectionStrategy {
    /// Use first N dimensions (simple, fast)
    Simple,
    /// Use mean of all dimensions (conservative)
    Mean,
    /// Future: PCA-based projection
    Pca,
    /// Future: UMAP projection
    Umap,
}

/// Projector that converts 64D coordinates to Vec3.
#[derive(Debug, Clone)]
pub struct Projector {
    config: ProjectionConfig,
}

impl Projector {
    /// Create a new projector with default configuration.
    pub fn new() -> Self {
        Self {
            config: ProjectionConfig::default(),
        }
    }

    /// Create a projector with custom configuration.
    pub fn with_config(config: ProjectionConfig) -> Self {
        Self { config }
    }

    /// Project a GhostBasin's coordinates to Vec3.
    pub fn project_to_vec3(&self, basin: &GhostBasin) -> Option<Vec3> {
        let coords = &basin.coordinates;
        
        if coords.len() < self.config.output_dim {
            return None;
        }

        match self.config.strategy {
            ProjectionStrategy::Simple => {
                // Use first 3 dimensions as approximate coordinates
                let x = coords[0] as f32;
                let y = coords[1] as f32;
                let z = coords[2] as f32;
                Some([x, y, z])
            }
            ProjectionStrategy::Mean => {
                // Use mean of first N dimensions
                let sum: f64 = coords[..self.config.output_dim].iter().sum();
                let mean = (sum / self.config.output_dim as f64) as f32;
                Some([mean, mean, mean])
            }
            ProjectionStrategy::Pca => {
                // Future: PCA-based projection
                // For now, fall back to simple
                let x = coords[0] as f32;
                let y = coords[1] as f32;
                let z = coords[2] as f32;
                Some([x, y, z])
            }
            ProjectionStrategy::Umap => {
                // Future: UMAP projection
                // For now, fall back to simple
                let x = coords[0] as f32;
                let y = coords[1] as f32;
                let z = coords[2] as f32;
                Some([x, y, z])
            }
        }
    }

    /// Project multiple basins to Vec3.
    pub fn project_batch(&self, basins: &[GhostBasin]) -> Vec<Option<Vec3>> {
        basins.iter().map(|b| self.project_to_vec3(b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projector_default_config() {
        let projector = Projector::new();
        assert_eq!(projector.config.output_dim, 3);
        assert_eq!(projector.config.strategy, ProjectionStrategy::Simple);
    }

    #[test]
    fn test_projector_custom_config() {
        let config = ProjectionConfig {
            output_dim: 3,
            strategy: ProjectionStrategy::Mean,
        };
        let projector = Projector::with_config(config);
        assert_eq!(projector.config.output_dim, 3);
        assert_eq!(projector.config.strategy, ProjectionStrategy::Mean);
    }

    #[test]
    fn test_projector_config() {
        let config = ProjectionConfig {
            output_dim: 3,
            strategy: ProjectionStrategy::Simple,
        };
        let projector = Projector::with_config(config);
        assert_eq!(projector.config.output_dim, 3);
    }
}
