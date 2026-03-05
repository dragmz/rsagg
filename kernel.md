# Kernel Work Notes (`kernel.cl`)

## Goal
Make `kernel.cl` compile under strict OpenCL 3.0 across vendor compilers and variants:
- default
- `-DCPU`
- `-DMSIG`

## Tools Used
- `rg` for fast code search and call-site discovery.
- `git diff` / `git status` for focused change tracking.
- `clang` (LLVM 22.1.0) for local OpenCL front-end syntax checks.
- NVIDIA OpenCL runtime compiler (`clBuildProgram` via `OpenCL.dll`) for real NVIDIA platform validation.
- AMD Radeon GPU Analyzer (`rga` 2.14.1) for AMD offline OpenCL compiler validation (`gfx900`, `gfx1030`, `gfx1100`).
- `python` + `ctypes` for scripted `clBuildProgram` checks (flags and macro variants).

## Where To Get The Tools
- `rg` (ripgrep):
  - Releases: <https://github.com/BurntSushi/ripgrep/releases>
  - Windows package managers: `winget install BurntSushi.ripgrep` or `scoop install ripgrep`
- `git`:
  - Downloads: <https://git-scm.com/downloads>
  - Package manager option: `winget install Git.Git`
- `clang` / LLVM:
  - Official: <https://llvm.org>
  - Releases: <https://github.com/llvm/llvm-project/releases>
  - Package manager options: `winget install LLVM.LLVM` or `scoop install llvm`
- AMD Radeon GPU Analyzer (`rga`):
  - Official repo/releases: <https://github.com/GPUOpen-Tools/radeon_gpu_analyzer/releases>
  - GPUOpen tools portal: <https://gpuopen.com/tools/>
- NVIDIA OpenCL runtime compiler:
  - Comes with NVIDIA GPU drivers: <https://www.nvidia.com/Download/index.aspx>
- AMD OpenCL runtime compiler (runtime path, separate from RGA offline compiler):
  - Comes with AMD GPU drivers: <https://www.amd.com/en/support/download/drivers.html>
- Python (for `ctypes` scripting):
  - Downloads: <https://www.python.org/downloads/>
  - Package manager option: `winget install Python.Python.3`

## Known-Good Versions (This Session)
- `ripgrep`: `15.1.0`
- `git`: `2.45.1.windows.1`
- `Python`: `3.14.3`
- `clang` (LLVM): `22.1.0`
- `Radeon GPU Analyzer` (`rga`): `2.14.1.3`
- NVIDIA OpenCL platform:
  - Platform: `NVIDIA CUDA` (`OpenCL 3.0 CUDA 13.0.97`)
  - Device used: `NVIDIA GeForce GTX 950`
  - Driver: `582.28`
- AMD offline compiler targets validated with RGA:
  - `gfx900`, `gfx1030`, `gfx1100`

## Validation Methods
1. Static compile checks (local):
   - `clang -x cl -cl-std=CL3.0 -fsyntax-only kernel.cl`
   - repeated with `-DCPU`, `-DMSIG`.
2. AMD vendor compilation:
   - `rga -s opencl ... --OpenCLoption '-cl-std=CL3.0'`
   - repeated for all required variants and multiple ASIC targets.
3. NVIDIA vendor runtime compilation:
   - `clBuildProgram` with `-cl-std=CL3.0`
   - repeated for default / `-DCPU` / `-DMSIG`.

## Techniques Applied In Code
- Explicit OpenCL address spaces for correctness in CL3:
  - `constant` for program-scope constants.
  - `global` / `private` in function signatures and call paths.
- Address-space-specific copy helpers:
  - private<-private
  - private<-global
  - private<-constant
- Constant-memory-safe conditional-move path for `ge_precomp` table reads.
- Variant-aware helper signatures:
  - CPU path writes base32 output to `global` destination.
  - Prefix comparison takes `global const` prefix pointer.
- No algorithmic/cryptographic logic changes; fixes were type/address-space portability hardening.

## Main CL3 Failure Classes Addressed
- Program-scope variables not declared in constant address space.
- Passing `global` pointers to functions expecting `private` pointers.
- Mixed address-space usage in shared helper functions across `CPU` and non-`CPU` paths.

## Result
`kernel.cl` now compiles with strict CL3.0 on:
- AMD compiler toolchain (RGA) for tested ASICs.
- NVIDIA OpenCL runtime compiler for tested device/platform.
- all required build variants: default, `CPU`, `MSIG`.
