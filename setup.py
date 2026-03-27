import os
import subprocess
from paddle.utils.cpp_extension import CUDAExtension, setup

# Get CUDA compute capability
def get_sm_version():
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL
        ).decode().strip().split('\n')[0]
        major, minor = output.split('.')
        return int(major) * 10 + int(minor)
    except Exception:
        return 100  # default to sm100

sm_version = get_sm_version()
print(f"Detected SM version: sm{sm_version}")

# Paths
project_dir = os.path.dirname(os.path.abspath(__file__))
csrc_dir = os.path.join(project_dir, "csrc")
build_dir = os.path.join(project_dir, "build")
cutlass_dir = os.path.join(build_dir, "cutlass")
CUTLASS_REPO = "https://github.com/NVIDIA/cutlass.git"
CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"

# Auto-clone cutlass if not present or commit mismatch
def ensure_cutlass():
    os.makedirs(build_dir, exist_ok=True)
    if not os.path.exists(os.path.join(cutlass_dir, ".git")):
        print(f"Cloning CUTLASS to {cutlass_dir} ...")
        subprocess.check_call(
            ["git", "clone", CUTLASS_REPO, cutlass_dir],
        )
    # Checkout the pinned commit
    current_commit = subprocess.check_output(
        ["git", "-C", cutlass_dir, "rev-parse", "HEAD"],
    ).decode().strip()
    if current_commit != CUTLASS_COMMIT:
        print(f"Checking out CUTLASS commit {CUTLASS_COMMIT} ...")
        subprocess.check_call(["git", "-C", cutlass_dir, "fetch", "--all"])
        subprocess.check_call(["git", "-C", cutlass_dir, "checkout", CUTLASS_COMMIT])

ensure_cutlass()

cutlass_include = os.path.join(cutlass_dir, "include")
cutlass_util_include = os.path.join(cutlass_dir, "tools", "util", "include")

# Source files - always include entry point
sources = [
    os.path.join(csrc_dir, "scaled_mm_entry.cu"),
]

# NVCC flags
nvcc_flags = [
    "-std=c++17",
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-Xcompiler", "-fPIC",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--threads=4",
]

# Define macros based on SM version
define_macros = []

# C2X kernels for SM75-SM89
if sm_version < 90:
    define_macros.append("ENABLE_SCALED_MM_C2X=1")
    sources.append(os.path.join(csrc_dir, "scaled_mm_c2x.cu"))
    # C2X needs older SM arch flags
    if sm_version >= 75 and sm_version < 80:
        nvcc_flags.extend(["-gencode", "arch=compute_75,code=sm_75"])
    elif sm_version >= 80 and sm_version < 89:
        nvcc_flags.extend(["-gencode", "arch=compute_80,code=sm_80"])
    elif sm_version == 89:
        nvcc_flags.extend(["-gencode", "arch=compute_89,code=sm_89"])
else:
    # Still compile C2X for compatibility but with the target arch
    define_macros.append("ENABLE_SCALED_MM_C2X=1")
    sources.append(os.path.join(csrc_dir, "scaled_mm_c2x.cu"))

# SM90 kernels
if sm_version == 90:  # Always compile SM90 kernels
    define_macros.append("ENABLE_SCALED_MM_SM90=1")
    sources.extend([
        os.path.join(csrc_dir, "scaled_mm_c3x_sm90.cu"),
        os.path.join(csrc_dir, "c3x", "scaled_mm_sm90_fp8.cu"),
        os.path.join(csrc_dir, "c3x", "scaled_mm_sm90_int8.cu"),
        os.path.join(csrc_dir, "c3x", "scaled_mm_azp_sm90_int8.cu"),
        os.path.join(csrc_dir, "c3x", "scaled_mm_blockwise_sm90_fp8.cu"),
    ])

# SM100 kernels
if sm_version == 100:  # Always compile SM100 kernels
    define_macros.append("ENABLE_SCALED_MM_SM100=1")
    sources.extend([
        os.path.join(csrc_dir, "scaled_mm_c3x_sm100.cu"),
        os.path.join(csrc_dir, "c3x", "scaled_mm_sm100_fp8.cu"),
        os.path.join(csrc_dir, "c3x", "scaled_mm_blockwise_sm100_fp8.cu"),
    ])

# SM120 kernels
if sm_version >= 120:  # Always compile SM120 kernels
    define_macros.append("ENABLE_SCALED_MM_SM120=1")
    sources.extend([
        os.path.join(csrc_dir, "scaled_mm_c3x_sm120.cu"),
        os.path.join(csrc_dir, "c3x", "scaled_mm_sm120_fp8.cu"),
        os.path.join(csrc_dir, "c3x", "scaled_mm_blockwise_sm120_fp8.cu"),
    ])

# Build arch flags for NVCC
# We need to generate code for all target SMs
arch_flags = []

# Always compile for SM75 and SM80 (c2x path) with virtual arch
# arch_flags.extend(["-gencode", "arch=compute_75,code=sm_75"])
# arch_flags.extend(["-gencode", "arch=compute_80,code=sm_80"])
# arch_flags.extend(["-gencode", "arch=compute_89,code=sm_89"])
# arch_flags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
arch_flags.extend(["-gencode", "arch=compute_100a,code=sm_100a"])
arch_flags.extend(["-gencode", "arch=compute_120a,code=sm_120a"])

# Add native arch for current GPU
if sm_version >= 100:
    arch_flags.extend(["-gencode", f"arch=compute_100a,code=compute_100a"])

nvcc_flags.extend(arch_flags)

# Add define macros to nvcc flags
for macro in define_macros:
    nvcc_flags.extend(["-D", macro])

# Include directories
include_dirs = [
    csrc_dir,
    cutlass_include,
    cutlass_util_include,
]

print(f"Sources: {len(sources)} files")
print(f"Defines: {define_macros}")
print(f"Include dirs: {include_dirs}")

setup(
    name="cutlass_scaled_mm_paddle",
    ext_modules=CUDAExtension(
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "nvcc": nvcc_flags,
        },
    ),
)
