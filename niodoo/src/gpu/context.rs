//! CUDA context management and device memory allocation

use anyhow::{Context, Result};
use cudarc::driver::sys::CUdevice_attribute as Attr;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// GPU context managing device, default stream, and persistent allocations
pub struct GpuContext {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,

    pub heap: GpuHeap,

    pub kernels: KernelCache,
}

impl GpuContext {
    pub fn new(device_id: usize) -> Result<Self> {
        let ctx = CudaContext::new(device_id).context("Failed to initialize CUDA context")?;
        let stream = ctx.default_stream();

        // Pre-allocate 1GB heap for sparse matrix operations
        let heap = GpuHeap::new(Arc::clone(&stream), 1 << 30)?;

        let kernels = KernelCache::new(Arc::clone(&ctx))?;

        Ok(Self {
            ctx,
            stream,
            heap,
            kernels,
        })
    }

    pub fn device_info(&self) -> Result<DeviceInfo> {
        let device = self.ctx.cu_device();
        let name = cudarc::driver::result::device::get_name(device)?;
        let cc_major =
            self.ctx
                .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? as u32;
        let cc_minor =
            self.ctx
                .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)? as u32;
        let sm_count =
            self.ctx
                .attribute(Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as usize;
        let mem_bytes = unsafe { cudarc::driver::result::device::total_mem(device)? } as usize;

        Ok(DeviceInfo {
            name,
            compute_capability: (cc_major, cc_minor),
            memory_gb: mem_bytes / (1024 * 1024 * 1024),
            sm_count,
        })
    }
}

/// GPU memory heap for dynamic allocations
#[allow(dead_code)]
pub struct GpuHeap {
    stream: Arc<CudaStream>,
    pub data: CudaSlice<u8>,
    pub alloc_ptr: CudaSlice<u32>,
    total_size: usize,
}

impl GpuHeap {
    pub fn new(stream: Arc<CudaStream>, size: usize) -> Result<Self> {
        let data = stream.alloc_zeros::<u8>(size)?;
        let alloc_ptr = stream.alloc_zeros::<u32>(1)?;

        Ok(Self {
            stream,
            data,
            alloc_ptr,
            total_size: size,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        let zero = [0u32];
        self.stream.memcpy_htod(&zero, &mut self.alloc_ptr)?;
        Ok(())
    }
}

/// Cache of compiled CUDA kernels
#[allow(dead_code)]
pub struct KernelCache {
    ctx: Arc<CudaContext>,
    pub apparent_pairs_ptx: Option<Ptx>,
    pub lock_free_ptx: Option<Ptx>,
}

impl KernelCache {
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            ctx,
            apparent_pairs_ptx: None,
            lock_free_ptx: None,
        })
    }

    pub fn compile_apparent_pairs(&mut self) -> Result<()> {
        if self.apparent_pairs_ptx.is_some() {
            return Ok(());
        }

        let kernel_src = include_str!("kernels/apparent_pairs.cu");
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            kernel_src,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_121"),
                ..Default::default()
            },
        )?;
        self.apparent_pairs_ptx = Some(ptx);
        Ok(())
    }
}

#[derive(Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_capability: (u32, u32),
    pub memory_gb: usize,
    pub sm_count: usize,
}
