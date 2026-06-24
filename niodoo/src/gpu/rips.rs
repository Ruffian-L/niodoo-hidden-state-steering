use anyhow::Result;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
#[cfg(feature = "cuda")]
use std::sync::Arc;

// Helper for Rips Complex structure
pub struct RipsComplex {
    pub adjacency: Vec<u8>, // N*N bitmap
    pub num_points: usize,
}

#[cfg(feature = "cuda")]
pub fn compute_distances_gpu(
    stream: &Arc<CudaStream>,
    points: &[[f32; 3]],
    threshold: f32,
) -> Result<cudarc::driver::CudaSlice<u8>> {
    let n = points.len();
    if n == 0 {
        return stream.alloc_zeros::<u8>(0).map_err(Into::into);
    }

    // 1. Upload points
    let points_flat: Vec<f32> = points.iter().flat_map(|p| p.as_slice()).cloned().collect();
    let d_points = stream.memcpy_stod(&points_flat)?;

    // 2. Allocate Edge Bitmap/List on GPU
    let mut d_adj = stream.alloc_zeros::<u8>(n * n)?;

    // 3. Compile and load distance kernel
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        include_str!("kernels/distance_matrix.cu"),
        cudarc::nvrtc::CompileOptions {
            arch: Some("sm_121"),
            ..Default::default()
        },
    )?;

    let module = stream.context().load_module(ptx)?;
    let f = module.load_function("compute_distances")?;

    let cfg = LaunchConfig::for_num_elems((n * n) as u32);
    let n_i32 = n as i32;
    let mut launch = stream.launch_builder(&f);
    launch.arg(&d_points);
    launch.arg(&mut d_adj);
    launch.arg(&n_i32);
    launch.arg(&threshold);
    unsafe { launch.launch(cfg) }?;

    Ok(d_adj)
}

#[cfg(feature = "cuda")]
pub fn build_rips_complex_gpu(
    stream: &Arc<CudaStream>,
    points: &[[f32; 3]],
    threshold: f32,
) -> Result<RipsComplex> {
    let n = points.len();
    let d_adj = compute_distances_gpu(stream, points, threshold)?;

    // 4. Download Adjacency
    let adj_host = stream.memcpy_dtov(&d_adj)?;

    Ok(RipsComplex {
        adjacency: adj_host,
        num_points: n,
    })
}

#[cfg(not(feature = "cuda"))]
pub fn build_rips_complex_gpu(
    _stream: &(),
    _points: &[[f32; 3]],
    _threshold: f32,
) -> Result<RipsComplex> {
    anyhow::bail!("GPU acceleration not enabled. Compile with --features cuda")
}
