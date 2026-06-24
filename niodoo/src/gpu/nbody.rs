//! Fused N-body pairwise acceleration kernel — replaces the candle [N, N, D]
//! broadcast pyramid in `niodoo/src/simulation.rs:2640-2697`.
//!
//! Public API: [`nbody_pairwise_accel`] takes positions [N, D] and masses [N]
//! and returns accelerations [N, D] without ever materializing an [N, N, D]
//! intermediate tensor. Math:
//!
//! ```text
//!   accel[i][k] = G * sum_{j != i} m_j * (pos[j][k] - pos[i][k]) / dist(i, j)^3
//!   dist(i, j) = sqrt(softening + sum_k (pos[j][k] - pos[i][k])^2)
//! ```
//!
//! Backed by the `nbody_pairwise_accel` CUDA kernel in `kernels.cu`. The PTX
//! is AOT-compiled by `build.rs` for sm_121 (NVIDIA GB10 / Blackwell).

use anyhow::Result;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::WrapErr;
use candle_core::{CpuStorage, CudaStorage, CustomOp2, DType, Layout, Shape, Tensor};

/// Embedded PTX for all niodoo CUDA kernels. Built by `build.rs` from
/// `niodoo/src/kernels.cu` with `nvcc -arch=sm_121 -O3 --use_fast_math`.
const KERNELS_PTX: &str = include_str!(env!("KERNELS_PTX_PATH"));

const KERNEL_MODULE: &str = "niodoo_kernels";
const KERNEL_NAME: &str = "nbody_pairwise_accel";

/// Block size used inside the kernel. Tuned for D up to a few thousand;
/// each thread handles `D / BLOCK_DIM` dims.
const BLOCK_DIM: u32 = 256;

/// CustomOp2 dispatch wrapper. Stateless aside from the two scalars.
pub struct NbodyAccelOp {
    pub g: f32,
    pub softening: f32,
}

impl CustomOp2 for NbodyAccelOp {
    fn name(&self) -> &'static str {
        "nbody_pairwise_accel"
    }

    fn cpu_fwd(
        &self,
        _pos_storage: &CpuStorage,
        pos_layout: &Layout,
        _mass_storage: &CpuStorage,
        _mass_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        // CPU fallback intentionally returns zeros. The whole point of this op
        // is to avoid the candle-broadcast variant on CPU; if you hit this, you
        // shouldn't be using it. Fail loud.
        let _ = pos_layout;
        Err(candle_core::Error::Msg(
            "nbody_pairwise_accel has no CPU implementation — keep tensors on a CUDA device"
                .to_string(),
        ))
    }

    fn cuda_fwd(
        &self,
        pos_storage: &CudaStorage,
        pos_layout: &Layout,
        mass_storage: &CudaStorage,
        mass_layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        // Shape checks.
        let (n, d) = pos_layout.shape().dims2()?;
        let mass_dims = mass_layout.shape().dims();
        let mass_n = match mass_dims {
            [n] => *n,
            other => candle_core::bail!(
                "nbody_pairwise_accel: mass must be 1-D [N], got {:?}",
                other
            ),
        };
        if mass_n != n {
            candle_core::bail!("nbody_pairwise_accel: mass length {mass_n} != pos rows {n}");
        }

        let dev = pos_storage.device().clone();
        let pos_slice = pos_storage.as_cuda_slice::<f32>()?;
        let pos_slice = match pos_layout.contiguous_offsets() {
            None => candle_core::bail!("nbody_pairwise_accel: pos must be contiguous"),
            Some((o1, o2)) => pos_slice.slice(o1..o2),
        };
        let mass_slice = mass_storage.as_cuda_slice::<f32>()?;
        let mass_slice = match mass_layout.contiguous_offsets() {
            None => candle_core::bail!("nbody_pairwise_accel: mass must be contiguous"),
            Some((o1, o2)) => mass_slice.slice(o1..o2),
        };

        // Output: [N, D] zero-init not strictly needed (kernel writes every elem),
        // but using alloc_zeros keeps debugging sane if the launch silently fails.
        let dst = unsafe { dev.alloc::<f32>(n * d) }?;

        let func = dev.get_or_load_custom_func(KERNEL_NAME, KERNEL_MODULE, KERNELS_PTX)?;

        // Shared memory layout (must match kernel):
        //   D floats for s_pos_i  + 32 floats for warp_sums
        let shared_mem_bytes = ((d + 32) as u32) * std::mem::size_of::<f32>() as u32;

        let n_i32 = n as i32;
        let d_i32 = d as i32;
        let g = self.g;
        let softening = self.softening;

        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes,
        };
        let mut builder = func.builder();
        builder.arg(&pos_slice);
        builder.arg(&mass_slice);
        builder.arg(&dst);
        candle_core::builder_arg!(builder, n_i32, d_i32, g, softening);
        unsafe { builder.launch(cfg) }.w()?;

        let dst = CudaStorage::wrap_cuda_slice(dst, dev);
        Ok((dst, Shape::from((n, d))))
    }
}

/// Compute pairwise N-body accelerations on the GPU.
///
/// `pos`: `[N, D]` f32 tensor on a CUDA device. `mass`: `[N]` f32 tensor on the
/// same device. Returns `[N, D]` f32 accelerations: each row is the net
/// acceleration on that particle from all others, ignoring self-interaction.
///
/// Replaces the [N, N, D] candle pyramid in `simulation.rs:2640-2697`.
pub fn nbody_pairwise_accel(pos: &Tensor, mass: &Tensor, g: f32, softening: f32) -> Result<Tensor> {
    if pos.dtype() != DType::F32 {
        anyhow::bail!(
            "nbody_pairwise_accel: pos must be f32, got {:?}",
            pos.dtype()
        );
    }
    if mass.dtype() != DType::F32 {
        anyhow::bail!(
            "nbody_pairwise_accel: mass must be f32, got {:?}",
            mass.dtype()
        );
    }

    let out = pos.apply_op2_no_bwd(mass, &NbodyAccelOp { g, softening })?;
    Ok(out)
}
