#!/bin/bash
# build.sh - Build script for Raibert Hopper C++ simulation
#
# Usage:
#   ./build.sh          # Build main simulation
#   ./build.sh test     # Build and run tests
#   ./build.sh compare  # Build and run MATLAB comparison
#   ./build.sh clean    # Remove build artifacts
#   ./build.sh all      # Build everything

set -e  # Exit on error

# Detect platform and set compiler
if [[ "$OSTYPE" == "darwin"* ]]; then
    CXX=${CXX:-clang++}
    # Try common Eigen locations on macOS
    if [ -d "/opt/homebrew/include/eigen3" ]; then
        EIGEN_PATH="/opt/homebrew/include/eigen3"
    elif [ -d "/usr/local/include/eigen3" ]; then
        EIGEN_PATH="/usr/local/include/eigen3"
    else
        echo "WARNING: Eigen not found in standard locations."
        echo "Install with: brew install eigen"
        echo "Or set EIGEN_PATH environment variable."
        EIGEN_PATH="/opt/homebrew/include/eigen3"
    fi
else
    CXX=${CXX:-g++}
    EIGEN_PATH=${EIGEN_PATH:-"/usr/include/eigen3"}
fi

# Compiler flags
CXXFLAGS="-std=c++17 -Wall -Wextra"
OPTFLAGS="-O2 -DNDEBUG"
DEBUGFLAGS="-O0 -g"
INCLUDES="-I$EIGEN_PATH"

echo "Using compiler: $CXX"
echo "Eigen path: $EIGEN_PATH"
echo ""

build_main() {
    echo "Building hopper simulation..."
    $CXX $CXXFLAGS $OPTFLAGS $INCLUDES build.cpp -o hopper
    echo "Built: ./hopper"
}

build_tests() {
    echo "Building tests..."
    $CXX $CXXFLAGS $DEBUGFLAGS $INCLUDES test_hopper.cpp -o test_hopper
    echo "Built: ./test_hopper"
    echo ""
    echo "Running tests..."
    ./test_hopper
}

build_compare() {
    echo "Building MATLAB comparison..."
    $CXX $CXXFLAGS $OPTFLAGS $INCLUDES compare_with_matlab.cpp -o compare_with_matlab
    echo "Built: ./compare_with_matlab"
    echo ""
    echo "Running comparison..."
    ./compare_with_matlab
}

clean() {
    echo "Cleaning..."
    rm -f hopper test_hopper compare_with_matlab
    rm -f trajectory.csv
    rm -rf *.dSYM
    echo "Done."
}

# Parse command
case "${1:-}" in
    test)
        build_tests
        ;;
    compare)
        build_compare
        ;;
    clean)
        clean
        ;;
    all)
        build_main
        echo ""
        build_tests
        echo ""
        build_compare
        ;;
    *)
        build_main
        ;;
esac
