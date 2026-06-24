//! Batch retrieval kernels — cosine similarity over a corpus, pairwise L2.
//!
//! Replaces the pattern of looping `cosine_similarity_f32(query, corpus[i])` /
//! `euclidean_distance(a[i], b[j])` element by element on CPU.
//!
//! Backed by `cosine_batch_score` and `l2_pairwise_dist` in
//! `niodoo/src/kernels.cu` (AOT-compiled by `build.rs` for sm_121).
//!
//! Target call sites — see audit (project `audit_cpu_hot_paths`) for context:
//!   - `niodoo/src/memory_topology.rs:70-75`     pairwise distance matrix (l2)
//!   - `niodoo/src/memory_topology.rs:210-220`   cosine over all memories
//!   - `niodoo/src/retrieval/advanced.rs:172-184` par_iter dot products
//!   - `niodoo/src/runtime/secret_sauce_codec.rs:185-195` cosine helper

use anyhow::Result;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::WrapErr;
use candle_core::{CpuStorage, CudaStorage, CustomOp2, DType, Layout, Shape, Tensor};

const KERNELS_PTX: &str = include_str!(env!("KERNELS_PTX_PATH"));
const KERNEL_MODULE: &str = "niodoo_kernels";
const BLOCK_DIM: u32 = 256;

// ────────────────────────────────────────────────────────────────────────────
// COSINE BATCH SCORE — query [D] · corpus [N, D] → scores [N]
// ────────────────────────────────────────────────────────────────────────────

pub struct CosineBatchOp;

impl CustomOp2 for CosineBatchOp {
    fn name(&self) -> &'static str {
        "cosine_batch_score"
    }

    fn cpu_fwd(
        &self,
        _q: &CpuStorage,
        _ql: &Layout,
        _c: &CpuStorage,
        _cl: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "cosine_batch_score has no CPU implementation".to_string(),
        ))
    }

    fn cuda_fwd(
        &self,
        q_storage: &CudaStorage,
        q_layout: &Layout,
        c_storage: &CudaStorage,
        c_layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let q_dims = q_layout.shape().dims();
        let d_q = match q_dims {
            [d] => *d,
            other => {
                candle_core::bail!("cosine_batch_score: query must be 1-D [D], got {:?}", other)
            }
        };
        let (n, d) = c_layout.shape().dims2()?;
        if d != d_q {
            candle_core::bail!("cosine_batch_score: query dim {d_q} != corpus dim {d}");
        }

        let dev = q_storage.device().clone();
        let q_slice = q_storage.as_cuda_slice::<f32>()?;
        let q_slice = match q_layout.contiguous_offsets() {
            None => candle_core::bail!("cosine_batch_score: query must be contiguous"),
            Some((o1, o2)) => q_slice.slice(o1..o2),
        };
        let c_slice = c_storage.as_cuda_slice::<f32>()?;
        let c_slice = match c_layout.contiguous_offsets() {
            None => candle_core::bail!("cosine_batch_score: corpus must be contiguous"),
            Some((o1, o2)) => c_slice.slice(o1..o2),
        };

        let dst = unsafe { dev.alloc::<f32>(n) }?;
        let func = dev.get_or_load_custom_func("cosine_batch_score", KERNEL_MODULE, KERNELS_PTX)?;

        // 3 * 32 floats for the 3-way warp reduction (dot, q_sq, c_sq).
        let shared_mem_bytes = (3 * 32) as u32 * std::mem::size_of::<f32>() as u32;

        let n_i32 = n as i32;
        let d_i32 = d as i32;

        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes,
        };
        let mut builder = func.builder();
        builder.arg(&q_slice);
        builder.arg(&c_slice);
        builder.arg(&dst);
        candle_core::builder_arg!(builder, n_i32, d_i32);
        unsafe { builder.launch(cfg) }.w()?;

        let dst = CudaStorage::wrap_cuda_slice(dst, dev);
        Ok((dst, Shape::from((n,))))
    }
}

/// Score `query` against every row of `corpus` with cosine similarity.
///
/// `query`: `[D]` f32 on CUDA. `corpus`: `[N, D]` f32 on the same device.
/// Returns `[N]` f32 of cosine similarity scores.
pub fn cosine_batch_score(query: &Tensor, corpus: &Tensor) -> Result<Tensor> {
    if query.dtype() != DType::F32 || corpus.dtype() != DType::F32 {
        anyhow::bail!("cosine_batch_score: both tensors must be f32");
    }
    let out = query.apply_op2_no_bwd(corpus, &CosineBatchOp)?;
    Ok(out)
}

// ────────────────────────────────────────────────────────────────────────────
// PAIRWISE L2 DISTANCE — A [M, D] vs B [N, D] → dists [M, N]
// ────────────────────────────────────────────────────────────────────────────

pub struct L2PairwiseOp {
    pub softening: f32,
}

impl CustomOp2 for L2PairwiseOp {
    fn name(&self) -> &'static str {
        "l2_pairwise_dist"
    }

    fn cpu_fwd(
        &self,
        _a: &CpuStorage,
        _al: &Layout,
        _b: &CpuStorage,
        _bl: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "l2_pairwise_dist has no CPU implementation".to_string(),
        ))
    }

    fn cuda_fwd(
        &self,
        a_storage: &CudaStorage,
        a_layout: &Layout,
        b_storage: &CudaStorage,
        b_layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let (m, d_a) = a_layout.shape().dims2()?;
        let (n, d_b) = b_layout.shape().dims2()?;
        if d_a != d_b {
            candle_core::bail!("l2_pairwise_dist: A dim {d_a} != B dim {d_b}");
        }
        let d = d_a;

        let dev = a_storage.device().clone();
        let a_slice = a_storage.as_cuda_slice::<f32>()?;
        let a_slice = match a_layout.contiguous_offsets() {
            None => candle_core::bail!("l2_pairwise_dist: A must be contiguous"),
            Some((o1, o2)) => a_slice.slice(o1..o2),
        };
        let b_slice = b_storage.as_cuda_slice::<f32>()?;
        let b_slice = match b_layout.contiguous_offsets() {
            None => candle_core::bail!("l2_pairwise_dist: B must be contiguous"),
            Some((o1, o2)) => b_slice.slice(o1..o2),
        };

        let dst = unsafe { dev.alloc::<f32>(m * n) }?;
        let func = dev.get_or_load_custom_func("l2_pairwise_dist", KERNEL_MODULE, KERNELS_PTX)?;

        let shared_mem_bytes = (32) as u32 * std::mem::size_of::<f32>() as u32;

        let m_i32 = m as i32;
        let n_i32 = n as i32;
        let d_i32 = d as i32;
        let softening = self.softening;

        let cfg = LaunchConfig {
            // Grid dims: x = N (B rows), y = M (A rows). One block per (i, j) pair.
            grid_dim: (n as u32, m as u32, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes,
        };
        let mut builder = func.builder();
        builder.arg(&a_slice);
        builder.arg(&b_slice);
        builder.arg(&dst);
        candle_core::builder_arg!(builder, m_i32, n_i32, d_i32, softening);
        unsafe { builder.launch(cfg) }.w()?;

        let dst = CudaStorage::wrap_cuda_slice(dst, dev);
        Ok((dst, Shape::from((m, n))))
    }
}

/// Pairwise L2 distance between every row of `a` and every row of `b`.
///
/// `a`: `[M, D]` f32 on CUDA. `b`: `[N, D]` f32 on the same device.
/// Returns `[M, N]` f32 distances. Use `softening = 1e-6` for numerical stability.
pub fn l2_pairwise_dist(a: &Tensor, b: &Tensor, softening: f32) -> Result<Tensor> {
    if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
        anyhow::bail!("l2_pairwise_dist: both tensors must be f32");
    }
    let out = a.apply_op2_no_bwd(b, &L2PairwiseOp { softening })?;
    Ok(out)
}
