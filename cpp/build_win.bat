@echo off
REM One-line build: cl /std:c++17 /O2 /EHsc /I"C:\libs\eigen-3.4.1" build.cpp /Fe:hopper.exe
REM Run from VS Developer Command Prompt, or call vcvars64.bat first
cl /std:c++17 /O2 /EHsc /I"C:\libs\eigen-3.4.1" build.cpp /Fe:hopper.exe
