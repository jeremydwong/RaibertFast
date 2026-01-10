@echo off
REM CUDA build script for Windows
REM Requires: CUDA Toolkit, Visual Studio Build Tools

setlocal

REM Paths - adjust if needed
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
set EIGEN_PATH=C:\libs\eigen-3.4.1
set MSVC_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207

REM Add to PATH
set PATH=%CUDA_PATH%\bin;%MSVC_PATH%\bin\Hostx64\x64;%PATH%

REM Architecture: sm_86 for RTX 30xx, sm_89 for RTX 40xx, sm_75 for RTX 20xx
set CUDA_ARCH=sm_86

echo Building CUDA hopper simulation...
echo CUDA: %CUDA_PATH%
echo Eigen: %EIGEN_PATH%

REM Build test executable
nvcc -std=c++17 -O2 ^
    -arch=%CUDA_ARCH% ^
    -I"%EIGEN_PATH%" ^
    -I".." ^
    test_cuda.cu ^
    -o test_cuda.exe

if %ERRORLEVEL% EQU 0 (
    echo Build successful: test_cuda.exe
) else (
    echo Build failed!
    exit /b 1
)

endlocal
