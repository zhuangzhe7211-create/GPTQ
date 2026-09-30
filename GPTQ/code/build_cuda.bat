@echo off
REM Build quant_cuda CUDA extension.
REM Must run inside the MSVC environment, otherwise cl.exe is not found.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo [FAIL] vcvars64.bat failed
    exit /b 1
)
cd /d C:\GPTQ\GPTQ\code
REM Build only for this machine's GPU (RTX 4080 = sm_89); otherwise it builds
REM for a dozen architectures and takes forever.
set TORCH_CUDA_ARCH_LIST=8.9
REM Tell setuptools we are already inside an activated VC env, so it does not
REM try to activate it a second time.
set DISTUTILS_USE_SDK=1
set MSSdk=1
REM CUDA 13.x ships CCCL, which refuses to compile under cl.exe's traditional
REM preprocessor. nvcc picks these up automatically and forwards them to cl.exe.
set NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor
python setup_cuda.py build_ext --inplace
