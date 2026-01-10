#!/bin/bash
# One-line build: clang++ -std=c++17 -O2 -I/opt/homebrew/include/eigen3 build.cpp -o hopper
# install eigen3 on macOS with: brew install eigen; or at https://eigen.tuxfamily.org/dox/GettingStarted.html
# On windows: cl /std:c++17 /O2 /I"C:\libs\eigen-3.4.1" build.cpp /Fe:hopper.exe
clang++ -std=c++17 -O2 -I/opt/homebrew/include/eigen3 build.cpp -o hopper
