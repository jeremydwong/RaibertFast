@echo off
REM CUDA build script for Windows
REM Requires: CUDA Toolkit, Visual Studio Build Tools

setlocal

REM Paths - adjust if needed
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
set MSVC_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207

REM Add to PATH
set PATH=%CUDA_PATH%\bin;%MSVC_PATH%\bin\Hostx64\x64;%PATH%

REM Output directory (one level up from project root)
set OUT_DIR=..\..\..\RaibertFastBuild
if not exist %OUT_DIR% mkdir %OUT_DIR%

echo Building CUDA hopper simulation...
echo CUDA: %CUDA_PATH%
echo Output: %OUT_DIR%

REM Build GPU test (FP32, semi-implicit Euler - fastest)
echo.
echo [1/2] Building GPU test (FP32)...
nvcc -O2 ^
    -ccbin "%MSVC_PATH%\bin\Hostx64\x64" ^
    -DHOPPER_USE_FLOAT32 ^
    -DHOPPER_INTEGRATOR=2 ^
    test_cuda.cu ^
    -o %OUT_DIR%\test_cuda.exe

if %ERRORLEVEL% NEQ 0 (
    echo GPU build failed!
    exit /b 1
)
echo Built: %OUT_DIR%\test_cuda.exe

REM Build CPU test (same flags for fair comparison)
echo.
echo [2/2] Building CPU test (FP32)...
nvcc -O2 ^
    -ccbin "%MSVC_PATH%\bin\Hostx64\x64" ^
    -DHOPPER_USE_FLOAT32 ^
    -DHOPPER_INTEGRATOR=2 ^
    -x cu ^
    ..\test_implicit_cpu.cpp ^
    -o %OUT_DIR%\test_cpu.exe

if %ERRORLEVEL% NEQ 0 (
    echo CPU build failed!
    exit /b 1
)
echo Built: %OUT_DIR%\test_cpu.exe

echo.
echo Build successful!
echo.
echo Run tests:
echo   %OUT_DIR%\test_cuda.exe --test
echo   %OUT_DIR%\test_cpu.exe --test
echo.
echo Run benchmark (4096 hoppers, 5s):
echo   %OUT_DIR%\test_cuda.exe --multi -n 4096 -t 5.0 --no-export
echo   %OUT_DIR%\test_cpu.exe --multi -n 4096 -t 5.0

endlocal
