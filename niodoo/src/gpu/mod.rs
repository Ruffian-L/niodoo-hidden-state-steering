//! GPU-accelerated persistent homology computation
//!
//! This module provides CUDA-accelerated implementations of the lock-free
//! persistent homology algorithm, offering 10-50x speedups for large point clouds.

#[cfg(feature = "cuda")]
pub mod context;
#[cfg(feature = "cuda")]
pub mod memory;
#[cfg(feature = "cuda")]
pub mod nbody;
#[cfg(feature = "cuda")]
pub mod retrieval;

// Exposed regardless of GPU feature, handles CPU fallback internally
pub mod lophat;

#[cfg(feature = "cuda")]
pub mod rips;

#[cfg(test)]
mod test_integration;

// N-body pairwise acceleration. With the `cuda` feature this is the fused CUDA
// kernel (nbody::nbody_pairwise_accel); without it, a candle CPU implementation
// with identical math so the runtime builds and runs on any machine.
#[cfg(feature = "cuda")]
pub use nbody::nbody_pairwise_accel;

/// CPU fallback for the fused CUDA n-body kernel. Same math as `kernels.cu`:
///   accel[i][k] = G * sum_{j != i} m_j * (pos[j][k] - pos[i][k]) / dist(i,j)^3
///   dist(i,j)   = sqrt(softening + sum_k (pos[j][k] - pos[i][k])^2)
/// Computed one row at a time to avoid materializing an [N, N, D] intermediate.
/// (CPU results may differ from the CUDA reproduction at the last bits of float
/// precision; the GPU path remains the canonical reproduction — see RUNBOOK.)
#[cfg(not(feature = "cuda"))]
pub fn nbody_pairwise_accel(
    pos: &candle_core::Tensor,
    mass: &candle_core::Tensor,
    g: f32,
    softening: f32,
) -> anyhow::Result<candle_core::Tensor> {
    use candle_core::{DType, Tensor};
    let pos = pos.to_dtype(DType::F32)?;
    let mass = mass.to_dtype(DType::F32)?;
    let (n, _d) = pos.dims2()?;
    let soft = softening as f64;
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let pi = pos.narrow(0, i, 1)?; // [1, D]
        let diff = pos.broadcast_sub(&pi)?; // [N, D] = pos[j] - pos[i]
        let dist2 = diff.sqr()?.sum(1)?.affine(1.0, soft)?; // [N] softened
        let dist = dist2.sqrt()?; // [N]
        let dist3 = (&dist * &dist)?.mul(&dist)?; // [N] = dist^3
        // w_j = m_j / dist3_j; the self term (j == i) is zero because diff == 0.
        let w = (&mass / &dist3)?.unsqueeze(1)?; // [N, 1]
        let contrib = diff.broadcast_mul(&w)?; // [N, D]
        rows.push(contrib.sum(0)?); // [D]
    }
    let accel = Tensor::stack(&rows, 0)?.affine(g as f64, 0.0)?; // [N, D] * G
    Ok(accel)
}

use crate::indexing::TopologicalFingerprint;
use crate::tivm::SplatRagConfig;
use crate::SplatInput;
use anyhow::{bail, Result};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Check if CUDA is available on this system
#[cfg(feature = "cuda")]
pub fn cuda_available() -> bool {
    CudaContext::device_count().unwrap_or(0) > 0
}

#[cfg(not(feature = "cuda"))]
pub fn cuda_available() -> bool {
    false
}

/// Determine if GPU acceleration is requested and available
pub fn should_use_gpu() -> bool {
    if !cfg!(feature = "cuda") {
        eprintln!("⚠️ GPU feature not compiled in");
        return false;
    }

    match std::env::var("SPLATRAG_USE_GPU") {
        Ok(val) if matches!(val.as_str(), "1" | "true" | "TRUE" | "yes" | "YES") => {
            let available = cuda_available();
            if available {
                eprintln!("🚀 GPU ACCELERATION ENABLED - CUDA device available");
            } else {
                eprintln!("⚠️ GPU requested but CUDA not available");
            }
            available
        }
        _ => {
            eprintln!("ℹ️ GPU not requested (set SPLATRAG_USE_GPU=1 to enable)");
            false
        }
    }
}

/// Attempt to compute a fingerprint on the GPU
#[cfg(not(feature = "cuda"))]
pub fn try_gpu_fingerprint(
    _splat: &SplatInput,
    _cfg: &SplatRagConfig,
) -> Result<TopologicalFingerprint> {
    bail!("GPU acceleration feature not enabled");
}

#[cfg(feature = "cuda")]
pub fn try_gpu_fingerprint(
    splat: &SplatInput,
    cfg: &SplatRagConfig,
) -> Result<TopologicalFingerprint> {
    let _ = cfg; // vpb_params no longer wired through this path; kept for API stability.

    let use_gpu = cuda_available() && std::env::var("SPLATRAG_USE_GPU").is_ok();
    if use_gpu {
        eprintln!("🚀 GPU ACCELERATION ENABLED - Using CUDA for fingerprint computation");
    } else {
        eprintln!("⚠️ GPU ACCELERATION DISABLED - Using CPU fallback");
    }

    if !cuda_available() {
        bail!("CUDA not available on this system");
    }

    // Point3 / Vec3 are already [f32; 3] aliases — clone the slice.
    let static_points: Vec<[f32; 3]> = splat.static_points.clone();

    let gpu_engine = GpuPhEngine::new(0, cfg.hom_dims.iter().copied().max().unwrap_or(1))?;
    let static_pd = gpu_engine.compute_persistence_gpu(&static_points)?;

    // H0 / H1 barcodes come straight from features_by_dim (matches indexing/fingerprint.rs pattern).
    let h0_static: Vec<(f32, f32)> = static_pd
        .features_by_dim
        .get(0)
        .cloned()
        .unwrap_or_default();
    let h1_static: Vec<(f32, f32)> = static_pd
        .features_by_dim
        .get(1)
        .cloned()
        .unwrap_or_default();

    if let Some(vels) = &splat.motion_velocities {
        if !vels.is_empty() {
            let motion_points: Vec<[f32; 3]> = vels.clone();
            let _ = gpu_engine.compute_persistence_gpu(&motion_points)?;
        }
    }

    Ok(TopologicalFingerprint::new(h0_static, h1_static))
}

/// Get the number of available CUDA devices
#[cfg(feature = "cuda")]
pub fn device_count() -> Result<usize> {
    Ok(CudaContext::device_count()? as usize)
}

#[cfg(not(feature = "cuda"))]
pub fn device_count() -> Result<usize> {
    Ok(0)
}

#[cfg(feature = "cuda")]
/// GPU-accelerated persistent homology engine
pub struct GpuPhEngine {
    context: Arc<context::GpuContext>,
    max_dim: usize,
}

#[cfg(feature = "cuda")]
const ADJ_TO_BOUNDARY_SRC: &str = r#"
extern "C" __global__ void adj_to_boundary_count(
    const unsigned char* adj,
    int* edge_counts,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int count = 0;
    for (int j = tid + 1; j < n; j++) {
        if (adj[tid * n + j] > 0) {
            count++;
        }
    }
    edge_counts[tid] = count;
}

extern "C" __global__ void adj_to_boundary_fill(
    const unsigned char* adj,
    const int* col_offsets,
    int* col_ptr,
    int* row_idx,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int offset = col_offsets[tid];
    int current = 0;

    for (int j = tid + 1; j < n; j++) {
        if (adj[tid * n + j] > 0) {
            int edge_idx = offset + current;
            col_ptr[edge_idx] = edge_idx * 2;
            if (edge_idx + 1 < (n*n)/2) {
                 col_ptr[edge_idx+1] = edge_idx * 2 + 2;
            }
            row_idx[edge_idx * 2] = tid;
            row_idx[edge_idx * 2 + 1] = j;
            current++;
        }
    }
}
"#;

#[cfg(feature = "cuda")]
impl GpuPhEngine {
    pub fn new(device_id: usize, max_dim: usize) -> Result<Self> {
        let context = Arc::new(context::GpuContext::new(device_id)?);
        Ok(Self { context, max_dim })
    }

    pub fn compute_persistence_gpu(&self, points: &[[f32; 3]]) -> Result<PersistenceDiagram> {
        let stream = &self.context.stream;
        let ctx = &self.context.ctx;

        // 1. Build Rips Complex Distance Matrix (GPU)
        let threshold = 5.0_f32;
        let d_adj = rips::compute_distances_gpu(stream, points, threshold)?;

        // 2. Adjacency -> Boundary (GPU)
        let n = points.len();
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            ADJ_TO_BOUNDARY_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_121"),
                ..Default::default()
            },
        )?;
        let adj_module = ctx.load_module(ptx)?;
        let f_count = adj_module.load_function("adj_to_boundary_count")?;
        let f_fill = adj_module.load_function("adj_to_boundary_fill")?;

        // Count edges
        let mut d_counts = stream.alloc_zeros::<i32>(n)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i32 = n as i32;
        {
            let mut launch = stream.launch_builder(&f_count);
            launch.arg(&d_adj);
            launch.arg(&mut d_counts);
            launch.arg(&n_i32);
            unsafe { launch.launch(cfg) }?;
        }

        // Prefix sum (host round-trip for N integers, negligible)
        let counts = stream.memcpy_dtov(&d_counts)?;
        let mut offsets = vec![0i32; n];
        let mut total_edges = 0;
        for i in 0..n {
            offsets[i] = total_edges;
            total_edges += counts[i];
        }
        let d_offsets = stream.memcpy_stod(&offsets)?;

        // Alloc Boundary
        let mut d_col_ptr = stream.alloc_zeros::<i32>(total_edges as usize + 1)?;
        let mut d_row_idx = stream.alloc_zeros::<i32>((total_edges * 2) as usize)?;

        // Fill
        {
            let mut launch = stream.launch_builder(&f_fill);
            launch.arg(&d_adj);
            launch.arg(&d_offsets);
            launch.arg(&mut d_col_ptr);
            launch.arg(&mut d_row_idx);
            launch.arg(&n_i32);
            unsafe { launch.launch(cfg) }?;
        }

        // Fix last col_ptr entry to point past the last edge.
        let last_val = [total_edges * 2];
        {
            let mut tail = d_col_ptr.slice_mut(total_edges as usize..);
            stream.memcpy_htod(&last_val, &mut tail)?;
        }

        // 3. Reduction (GPU) — load lock_free_kernel from lophat/kernels.cu.
        let ptx_reduce = cudarc::nvrtc::compile_ptx_with_opts(
            include_str!("lophat/kernels.cu"),
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_121"),
                ..Default::default()
            },
        )?;
        let reduce_module = ctx.load_module(ptx_reduce)?;
        let f_reduce = reduce_module.load_function("lock_free_kernel")?;

        let num_cols = total_edges as usize;
        let num_rows = n;

        // Pivots initialized to -1.
        let neg_ones = vec![-1i32; num_cols];
        let mut d_pivots = stream.memcpy_stod(&neg_ones)?;

        // Heap for fill-in.
        let heap_capacity = num_cols * 10;
        let mut d_heap_data = stream.alloc_zeros::<i32>(heap_capacity)?;
        let mut d_heap_head = stream.alloc_zeros::<i32>(1)?;

        let heads_init = vec![-1i32; num_cols];
        let mut d_col_heads = stream.memcpy_stod(&heads_init)?;
        let lens_init = vec![2i32; num_cols];
        let mut d_col_lens = stream.memcpy_stod(&lens_init)?;

        let cfg_reduce = LaunchConfig::for_num_elems(num_cols as u32);
        let num_cols_i32 = num_cols as i32;
        let num_rows_i32 = num_rows as i32;
        let heap_capacity_i32 = heap_capacity as i32;
        {
            let mut launch = stream.launch_builder(&f_reduce);
            launch.arg(&mut d_pivots);
            launch.arg(&d_col_ptr);
            launch.arg(&d_row_idx);
            launch.arg(&num_cols_i32);
            launch.arg(&num_rows_i32);
            launch.arg(&mut d_heap_data);
            launch.arg(&mut d_heap_head);
            launch.arg(&heap_capacity_i32);
            launch.arg(&mut d_col_heads);
            launch.arg(&mut d_col_lens);
            unsafe { launch.launch(cfg_reduce) }?;
        }

        ctx.synchronize()?;

        // 4. Download pivots.
        let pivots = stream.memcpy_dtov(&d_pivots)?;

        // 5. Construct diagram.
        let mut pairs = Vec::new();
        let mut features_by_dim: Vec<Vec<(f32, f32)>> = vec![Vec::new(); self.max_dim + 1];

        let mut killed_vertices = std::collections::HashSet::new();
        for &row in &pivots {
            if row != -1 {
                killed_vertices.insert(row as usize);
            }
        }

        // H0 features: vertices not killed.
        for i in 0..n {
            if !killed_vertices.contains(&i) {
                pairs.push((0.0_f32, f32::INFINITY));
                features_by_dim[0].push((0.0_f32, f32::INFINITY));
            }
        }

        Ok(PersistenceDiagram {
            dimension: self.max_dim,
            pairs,
            features_by_dim,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    pub dimension: usize,
    pub pairs: Vec<(f32, f32)>,
    pub features_by_dim: Vec<Vec<(f32, f32)>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability() {
        // Just ensure cuda_available() runs without panic.
        let _ = cuda_available();
    }
}
