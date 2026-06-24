use super::MatrixDecomposer;
use anyhow::Result;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

// Compressed Sparse Column (CSC) format on the GPU.
pub struct CudaDecomposer {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    reduce_module: Arc<cudarc::driver::CudaModule>,
    // Kept CPU-side for read-back queries (get_pivot, get_r_col).
    cpu_fallback_cache: Option<Vec<Vec<usize>>>,
    num_cols: usize,
    num_rows: usize,
}

impl CudaDecomposer {
    pub fn new(boundary_matrix: Vec<Vec<usize>>) -> Self {
        let ctx = CudaContext::new(0).expect("Failed to initialize CUDA context. Check drivers.");
        let stream = ctx.default_stream();

        // Compile lophat/kernels.cu via NVRTC for sm_121 — no offline reduce.ptx available.
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            include_str!("kernels.cu"),
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_121"),
                ..Default::default()
            },
        )
        .expect("Failed to NVRTC-compile lophat/kernels.cu");
        let module = ctx.load_module(ptx).expect("Failed to load reduce module");

        let cols = boundary_matrix.len();
        let rows = cols;

        Self {
            ctx,
            stream,
            reduce_module: module,
            cpu_fallback_cache: Some(boundary_matrix),
            num_cols: cols,
            num_rows: rows,
        }
    }

    /// Flatten the matrix to CSC and ship to the device.
    fn upload_matrix(&self) -> Result<(CudaSlice<usize>, CudaSlice<usize>)> {
        let matrix = self
            .cpu_fallback_cache
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CPU fallback cache not initialized"))?;

        let mut col_ptr = Vec::with_capacity(self.num_cols + 1);
        let mut row_indices = Vec::new();

        let mut current_ptr: usize = 0;
        col_ptr.push(current_ptr);

        for col in matrix {
            for &row_idx in col {
                row_indices.push(row_idx);
                current_ptr += 1;
            }
            col_ptr.push(current_ptr);
        }

        let dev_col_ptr = self.stream.memcpy_stod(&col_ptr)?;
        let dev_row_idx = self.stream.memcpy_stod(&row_indices)?;
        Ok((dev_col_ptr, dev_row_idx))
    }
}

impl MatrixDecomposer for CudaDecomposer {
    fn add_entries(&mut self, _target: usize, _source: usize) {
        // GPU path batches reduction; per-pair adds are no-ops.
    }

    fn get_pivot(&self, col_idx: usize) -> Option<usize> {
        self.cpu_fallback_cache.as_ref()?[col_idx].last().copied()
    }

    fn get_r_col(&self, col_idx: usize) -> Vec<usize> {
        self.cpu_fallback_cache
            .as_ref()
            .and_then(|cache| cache.get(col_idx))
            .cloned()
            .unwrap_or_default()
    }

    fn reduce(&mut self) {
        println!("⚡ Dispatching CUDA reduction kernel...");

        let (mut d_col_ptr, mut d_row_idx) = self.upload_matrix().unwrap();

        let mut d_pivots = self
            .stream
            .alloc_zeros::<isize>(self.num_cols)
            .expect("alloc pivots failed");

        let cfg = LaunchConfig::for_num_elems(self.num_cols as u32);
        let func = self
            .reduce_module
            .load_function("reduce_kernel")
            .expect("reduce_kernel not in module");

        let num_cols = self.num_cols;
        let mut launch = self.stream.launch_builder(&func);
        launch.arg(&mut d_col_ptr);
        launch.arg(&mut d_row_idx);
        launch.arg(&mut d_pivots);
        launch.arg(&num_cols);
        unsafe { launch.launch(cfg) }.expect("CUDA kernel launch failed");

        self.ctx.synchronize().expect("CUDA synchronize failed");
        let _ = self.num_rows;

        println!("⚡ Reduction complete.");
    }
}
