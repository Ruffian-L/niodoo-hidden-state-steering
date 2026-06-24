use std::env;
use std::path::PathBuf;
use std::process::Command;

const KERNELS_CU: &str = "src/kernels.cu";
const TARGET_ARCH: &str = "sm_121";

fn main() {
    println!("cargo:rerun-if-changed={KERNELS_CU}");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let ptx_out = out_dir.join("kernels.ptx");

    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".into());

    let status = Command::new(&nvcc)
        .args([
            "-ptx",
            &format!("--gpu-architecture={TARGET_ARCH}"),
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
            println!("cargo:warning=niodoo: kernels.cu AOT-compiled to PTX for {TARGET_ARCH}");
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
