use std::env;
use std::path::PathBuf;
use std::process::Command;

const KERNELS_CU: &str = "src/kernels.cu";
// Default target GPU arch (NVIDIA GB10 / Blackwell). Override for other GPUs with
// e.g. `NIODOO_CUDA_ARCH=sm_80` (Ampere) / `sm_90` (Hopper) before building.
const DEFAULT_TARGET_ARCH: &str = "sm_121";

fn main() {
    println!("cargo:rerun-if-changed={KERNELS_CU}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=NIODOO_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=NVCC");

    // CPU-only builds (no `cuda` feature) need no CUDA toolchain at all.
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        println!("cargo:warning=niodoo: building without the `cuda` feature; skipping nvcc/PTX (CPU-only)");
        return;
    }

    let target_arch = env::var("NIODOO_CUDA_ARCH").unwrap_or_else(|_| DEFAULT_TARGET_ARCH.into());
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let ptx_out = out_dir.join("kernels.ptx");

    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".into());

    let status = Command::new(&nvcc)
        .args([
            "-ptx",
            &format!("--gpu-architecture={target_arch}"),
            "-O3",
            "--use_fast_math",
            "-diag-suppress=177",
            "-o",
        ])
        .arg(&ptx_out)
        .arg(KERNELS_CU)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-env=KERNELS_PTX_PATH={}", ptx_out.display());
            println!("cargo:warning=niodoo: kernels.cu AOT-compiled to PTX for {target_arch}");
        }
        Ok(s) => {
            println!(
                "cargo:warning=niodoo: nvcc exited {s} compiling {KERNELS_CU}; runtime NVRTC fallback will be used"
            );
        }
        Err(e) => {
            println!(
                "cargo:warning=niodoo: nvcc unavailable ({e}); skipping AOT PTX compile, runtime NVRTC fallback will be used"
            );
        }
    }
}
