@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cl /std:c++17 /O2 /EHsc /DHOPPER_INTEGRATOR=INTEGRATOR_SEMI_IMPLICIT_EULER test_implicit_cpu.cpp /Fe:test_cpu_semi_euler.exe
