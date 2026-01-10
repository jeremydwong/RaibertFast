// hopper_types.cuh - Data structures for CUDA hopper simulation
//
// Defines both Structure of Arrays (SoA) for GPU and single-hopper structs for CPU/testing
//
// This header is compatible with both CUDA (nvcc) and standard C++ compilers.
// When compiled without CUDA, the GPU-specific parts are disabled.

#ifndef HOPPER_TYPES_CUH
#define HOPPER_TYPES_CUH

// ============================================================================
// CPU/CUDA COMPATIBILITY LAYER
// ============================================================================

#ifdef __CUDACC__
    // Compiling with nvcc - use real CUDA
    #include <cuda_runtime.h>
#else
    // Compiling with standard C++ - define empty decorators
    #define __host__
    #define __device__
    #define __constant__ static
#endif

// ============================================================================
// SCALAR TYPE SELECTION
// ============================================================================
// Define HOPPER_USE_FLOAT32 to use single precision (much faster on consumer GPUs)

#ifdef HOPPER_USE_FLOAT32
    using Scalar = float;
    #define SCALAR_SUFFIX f
#else
    using Scalar = double;
    #define SCALAR_SUFFIX
#endif

// Helper macro for literal constants (e.g., S(1.0) becomes 1.0f for float32)
#define S(x) ((Scalar)(x))

// ============================================================================
// CONSTANTS
// ============================================================================

// FSM states
#define FSM_COMPRESSION 0
#define FSM_THRUST      1
#define FSM_FLIGHT      2
#define FSM_NUM_STATES  3

// Integrator selection
#define INTEGRATOR_IMPLICIT_MIDPOINT           1
#define INTEGRATOR_SEMI_IMPLICIT_EULER         2
#define INTEGRATOR_IMPLICIT_EULER              3
#define INTEGRATOR_IMPLICIT_MIDPOINT_ANALYTICAL 4  // Uses analytical Jacobian

#ifndef HOPPER_INTEGRATOR
#define HOPPER_INTEGRATOR INTEGRATOR_IMPLICIT_MIDPOINT_ANALYTICAL
#endif

// ============================================================================
// PHYSICAL PARAMETERS (constant across all hoppers)
// ============================================================================

struct PhysicsParams {
    Scalar m      = S(10.0);   // body mass [kg]
    Scalar m_l    = S(1.0);    // leg mass [kg]
    Scalar J      = S(10.0);   // body moment of inertia [kg*m^2]
    Scalar J_l    = S(1.0);    // leg moment of inertia [kg*m^2]
    Scalar g      = S(9.8);    // gravity [m/s^2]
    Scalar k_l    = S(1e3);    // leg spring constant [N/m]
    Scalar k_stop = S(1e5);    // leg stop spring constant [N/m]
    Scalar b_stop = S(1e3);    // leg stop damping [N*s/m]
    Scalar k_g    = S(1e4);    // ground spring constant [N/m]
    Scalar b_g    = S(300.0);  // ground damping [N*s/m]
    Scalar r_s0   = S(1.0);    // leg spring rest length [m]
    Scalar l_1    = S(0.5);    // foot to leg COM distance [m]
    Scalar l_2    = S(0.4);    // hip to body COM distance [m]
};

// Device-side constant memory for physics params
__constant__ PhysicsParams d_physics;

// ============================================================================
// SINGLE HOPPER STATE (for CPU reference and single-thread GPU testing)
// ============================================================================

struct HopperState {
    // Positions (5)
    Scalar x_foot;      // x position of foot
    Scalar z_foot;      // y/z position of foot (height)
    Scalar phi_leg;     // absolute angle of leg from vertical
    Scalar phi_body;    // absolute angle of body from vertical
    Scalar len_leg;     // leg length

    // Velocities (5)
    Scalar ddt_x_foot;
    Scalar ddt_z_foot;
    Scalar ddt_phi_leg;
    Scalar ddt_phi_body;
    Scalar ddt_len_leg;

    // FSM state
    int fsm_state;

    // FSM timing
    Scalar T_s;              // stance time estimate
    Scalar T_compression;    // compression duration
    Scalar t_thrust_on;      // time thrust started

    // Array access for compatibility
    __host__ __device__ Scalar& operator[](int i) {
        switch (i) {
            case 0: return x_foot;
            case 1: return z_foot;
            case 2: return phi_leg;
            case 3: return phi_body;
            case 4: return len_leg;
            case 5: return ddt_x_foot;
            case 6: return ddt_z_foot;
            case 7: return ddt_phi_leg;
            case 8: return ddt_phi_body;
            case 9: return ddt_len_leg;
            default: return x_foot;
        }
    }

    __host__ __device__ const Scalar& operator[](int i) const {
        return const_cast<HopperState*>(this)->operator[](i);
    }

    // Default constructor
    __host__ __device__ HopperState() :
        x_foot(0), z_foot(0), phi_leg(0), phi_body(0), len_leg(S(1.0)),
        ddt_x_foot(0), ddt_z_foot(0), ddt_phi_leg(0), ddt_phi_body(0), ddt_len_leg(0),
        fsm_state(FSM_FLIGHT), T_s(S(0.425)), T_compression(0), t_thrust_on(0) {}
};

// ============================================================================
// CONTROL PARAMETERS (can vary per hopper for sweeps)
// ============================================================================

struct ControlParams {
    Scalar k_fp    = S(150.0);   // foot placement gain
    Scalar b_fp    = S(15.0);    // foot placement damping
    Scalar k_att   = S(150.0);   // attitude gain
    Scalar b_att   = S(15.0);    // attitude damping
    Scalar k_xdot  = S(0.02);    // forward speed gain
    Scalar thrust_scale = S(0.035);  // thrust as fraction of k_l
    Scalar x_dot_des = S(0.0);   // desired forward velocity
    Scalar T_MAX_THRUST_DUR = S(0.425 * 0.35);  // max thrust duration
};

// ============================================================================
// STRUCTURE OF ARRAYS (SoA) for GPU - coalesced memory access
// ============================================================================

struct HopperStateArrays {
    // Positions (N each)
    Scalar* x_foot;
    Scalar* z_foot;
    Scalar* phi_leg;
    Scalar* phi_body;
    Scalar* len_leg;

    // Velocities (N each)
    Scalar* ddt_x_foot;
    Scalar* ddt_z_foot;
    Scalar* ddt_phi_leg;
    Scalar* ddt_phi_body;
    Scalar* ddt_len_leg;

    // FSM state
    int* fsm_state;

    // FSM timing
    Scalar* T_s;
    Scalar* T_compression;
    Scalar* t_thrust_on;

    int N;  // number of hoppers
};

struct ControlParamArrays {
    // Per-hopper control parameters (for sweeps)
    Scalar* k_fp;
    Scalar* b_fp;
    Scalar* k_att;
    Scalar* b_att;
    Scalar* k_xdot;
    Scalar* thrust_scale;
    Scalar* x_dot_des;

    int N;
};

// ============================================================================
// HELPER FUNCTIONS (CUDA-only)
// ============================================================================

#ifdef __CUDACC__

// Allocate SoA arrays on device
inline HopperStateArrays allocate_state_arrays_device(int N) {
    HopperStateArrays arrays;
    arrays.N = N;

    cudaMalloc(&arrays.x_foot, N * sizeof(Scalar));
    cudaMalloc(&arrays.z_foot, N * sizeof(Scalar));
    cudaMalloc(&arrays.phi_leg, N * sizeof(Scalar));
    cudaMalloc(&arrays.phi_body, N * sizeof(Scalar));
    cudaMalloc(&arrays.len_leg, N * sizeof(Scalar));

    cudaMalloc(&arrays.ddt_x_foot, N * sizeof(Scalar));
    cudaMalloc(&arrays.ddt_z_foot, N * sizeof(Scalar));
    cudaMalloc(&arrays.ddt_phi_leg, N * sizeof(Scalar));
    cudaMalloc(&arrays.ddt_phi_body, N * sizeof(Scalar));
    cudaMalloc(&arrays.ddt_len_leg, N * sizeof(Scalar));

    cudaMalloc(&arrays.fsm_state, N * sizeof(int));

    cudaMalloc(&arrays.T_s, N * sizeof(Scalar));
    cudaMalloc(&arrays.T_compression, N * sizeof(Scalar));
    cudaMalloc(&arrays.t_thrust_on, N * sizeof(Scalar));

    return arrays;
}

// Free SoA arrays on device
inline void free_state_arrays_device(HopperStateArrays& arrays) {
    cudaFree(arrays.x_foot);
    cudaFree(arrays.z_foot);
    cudaFree(arrays.phi_leg);
    cudaFree(arrays.phi_body);
    cudaFree(arrays.len_leg);

    cudaFree(arrays.ddt_x_foot);
    cudaFree(arrays.ddt_z_foot);
    cudaFree(arrays.ddt_phi_leg);
    cudaFree(arrays.ddt_phi_body);
    cudaFree(arrays.ddt_len_leg);

    cudaFree(arrays.fsm_state);

    cudaFree(arrays.T_s);
    cudaFree(arrays.T_compression);
    cudaFree(arrays.t_thrust_on);
}

// Copy single hopper state to device arrays at index
inline void copy_state_to_device(const HopperState& state, HopperStateArrays& arrays, int idx) {
    cudaMemcpy(arrays.x_foot + idx, &state.x_foot, sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(arrays.z_foot + idx, &state.z_foot, sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(arrays.phi_leg + idx, &state.phi_leg, sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(arrays.phi_body + idx, &state.phi_body, sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(arrays.len_leg + idx, &state.len_leg, sizeof(Scalar), cudaMemcpyHostToDevice);

    cudaMemcpy(arrays.ddt_x_foot + idx, &state.ddt_x_foot, sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(arrays.ddt_z_foot + idx, &state.ddt_z_foot, sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(arrays.ddt_phi_leg + idx, &state.ddt_phi_leg, sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(arrays.ddt_phi_body + idx, &state.ddt_phi_body, sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(arrays.ddt_len_leg + idx, &state.ddt_len_leg, sizeof(Scalar), cudaMemcpyHostToDevice);

    cudaMemcpy(arrays.fsm_state + idx, &state.fsm_state, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(arrays.T_s + idx, &state.T_s, sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(arrays.T_compression + idx, &state.T_compression, sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(arrays.t_thrust_on + idx, &state.t_thrust_on, sizeof(Scalar), cudaMemcpyHostToDevice);
}

// Copy single hopper state from device arrays at index
inline void copy_state_from_device(HopperState& state, const HopperStateArrays& arrays, int idx) {
    cudaMemcpy(&state.x_foot, arrays.x_foot + idx, sizeof(Scalar), cudaMemcpyDeviceToHost);
    cudaMemcpy(&state.z_foot, arrays.z_foot + idx, sizeof(Scalar), cudaMemcpyDeviceToHost);
    cudaMemcpy(&state.phi_leg, arrays.phi_leg + idx, sizeof(Scalar), cudaMemcpyDeviceToHost);
    cudaMemcpy(&state.phi_body, arrays.phi_body + idx, sizeof(Scalar), cudaMemcpyDeviceToHost);
    cudaMemcpy(&state.len_leg, arrays.len_leg + idx, sizeof(Scalar), cudaMemcpyDeviceToHost);

    cudaMemcpy(&state.ddt_x_foot, arrays.ddt_x_foot + idx, sizeof(Scalar), cudaMemcpyDeviceToHost);
    cudaMemcpy(&state.ddt_z_foot, arrays.ddt_z_foot + idx, sizeof(Scalar), cudaMemcpyDeviceToHost);
    cudaMemcpy(&state.ddt_phi_leg, arrays.ddt_phi_leg + idx, sizeof(Scalar), cudaMemcpyDeviceToHost);
    cudaMemcpy(&state.ddt_phi_body, arrays.ddt_phi_body + idx, sizeof(Scalar), cudaMemcpyDeviceToHost);
    cudaMemcpy(&state.ddt_len_leg, arrays.ddt_len_leg + idx, sizeof(Scalar), cudaMemcpyDeviceToHost);

    cudaMemcpy(&state.fsm_state, arrays.fsm_state + idx, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(&state.T_s, arrays.T_s + idx, sizeof(Scalar), cudaMemcpyDeviceToHost);
    cudaMemcpy(&state.T_compression, arrays.T_compression + idx, sizeof(Scalar), cudaMemcpyDeviceToHost);
    cudaMemcpy(&state.t_thrust_on, arrays.t_thrust_on + idx, sizeof(Scalar), cudaMemcpyDeviceToHost);
}

#endif // __CUDACC__

#endif // HOPPER_TYPES_CUH
