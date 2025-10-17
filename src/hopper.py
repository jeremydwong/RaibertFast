import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Tuple
import time
import copy


# ============================================================================
# PARAMETERS CLASS (from hopperParameters.m)
# ============================================================================
@dataclass
class Parameters:
    """Parameters class for hopper simulation"""
    # Physical parameters
    m: float = 10.0        # mass of the body
    m_l: float = 1.0       # mass of the leg
    J: float = 10.0        # moment of inertia of the body
    J_l: float = 1.0       # moment of inertia of the leg
    g: float = 9.8         # gravity
    k_l: float = 1e3       # spring constant of leg spring
    k_stop: float = 2e5    # spring constant of leg stop
    b_stop: float = 1e3    # damping constant of leg stop
    k_g: float = 1e4       # spring constant of the ground
    b_g: float = 300.0     # damping constant of the ground
    r_s0: float = 1.0      # rest length of the leg spring
    l_1: float = 0.5       # distance from the foot to the com of the leg
    l_2: float = 0.4       # distance from the hip to the com of the body

    # FSM states
    FSM_COMPRESSION: int = 0
    FSM_THRUST: int = 1
    FSM_LOADING: int = 99
    FSM_FLIGHT: int = 2
    FSM_NUM_STATES: int = 3

    # State parameters
    fsm_state: int = 2  # FSM_FLIGHT - usually we start in the air
    t_state_switch: float = 0.0
    x_dot_des: float = 0.0
    T_s: float = 0.425  # Tedrake
    T_compression: float = 0.0

    t_thrust_on: float = 0.0
    T_MAX_THRUST_DUR: float = 0.425 * 0.35  # relative to Tedrake

    toString: str = '1: x foot 2: y foot 3: abs angle leg (vert) 4: abs angle body (vert) 5:leg length'


def hopperParameters():
    """function p = hopperParameters()"""
    p = Parameters()
    p.m = 10.0
    p.m_l = 1.0
    p.J = 10.0
    p.J_l = 1.0
    p.g = 9.8
    p.k_l = 1e3
    p.k_stop = 1e5
    p.b_stop = 1e3
    p.k_g = 1e4
    p.b_g = 300.0
    p.r_s0 = 1.0
    p.l_1 = 0.5
    p.l_2 = 0.4

    p.FSM_COMPRESSION = 0
    p.FSM_THRUST = 1
    p.FSM_LOADING = 99
    p.FSM_FLIGHT = 2
    p.FSM_NUM_STATES = 3

    p.fsm_state = p.FSM_FLIGHT
    p.t_state_switch = 0.0
    p.x_dot_des = 0.0
    p.T_s = 0.425
    p.T_compression = 0
    p.t_thrust_on = 0
    p.T_MAX_THRUST_DUR = 0.425 * 0.35

    p.toString = '1: x foot 2: y foot 3: abs angle leg (vert) 4: abs angle body (vert) 5:leg length'
    return p


# ============================================================================
# STATE CONTROL (from hopperStateControl.m)
# ============================================================================
def hopperStateControl(t, q, param):
    """function [u,internalStruct] = hopperStateControl(t,q,param)"""
    THRUST = param.FSM_THRUST
    COMPRESSION = param.FSM_COMPRESSION
    FLIGHT = param.FSM_FLIGHT

    # foot placement originally k=153 and b=14.
    k_fp = 150.0
    b_fp = 15.0
    # attitude. originally k=153 and b=14.
    k_att = 150.0
    b_att = 15.0
    # forward speed. originally .01.
    k_xdot = 0.02
    # originally 0.035 m (u1 altered l_spring not Force).
    thrust = 0.035 * param.k_l
    # if we are above this height.
    thr_z_low = 0.01
    u_retract = -0.1 * param.k_l

    if param.T_s == 0:
        print('Warning: using default T_s.')
        T_s = 0.425
    else:
        T_s = param.T_s

    PASSTHROUGH_0FORCE = 0
    u = np.array([0.0, 0.0])
    a_des = 0
    if not PASSTHROUGH_0FORCE:
        d_xfoot_dt = q[5]
        y_foot = q[1]
        a = q[2]
        dadt = q[7]
        b = q[3]
        dbdt = q[8]
        l = q[4]
        dldt = q[9]
        stance_ang_des = a / 2

        if param.fsm_state == THRUST:
            u[0] = thrust
            u[1] = -k_att * (b - stance_ang_des) - b_att * dbdt
        elif param.fsm_state == COMPRESSION:
            u[1] = -k_att * (b - stance_ang_des) - b_att * dbdt
        elif param.fsm_state == FLIGHT:
            d_xbody_dt = d_xfoot_dt + dldt * np.sin(a) + l * np.cos(a) * dadt + param.l_2 * np.cos(b) * dbdt
            a_des = -np.arcsin((1 * d_xbody_dt * T_s / 2 + k_xdot * (d_xbody_dt - param.x_dot_des)) / l)
            if np.isnan(a_des):
                print('warning! nan in a_des. an error has occurred.')
            if y_foot > thr_z_low:
                u[1] = k_fp * (a - a_des) + b_fp * (dadt)
    else:
        pass

    if 1:
        if param.fsm_state == THRUST:
            if t - param.t_thrust_on > param.T_MAX_THRUST_DUR:
                u[0] = 0
    return u, a_des


# ============================================================================
# DYNAMICS FORWARD (from hopperDynamicsFwd.m)
# ============================================================================
def hopperDynamicsFwd(t, q, p_obj):
    """function structOut = hopperDynamicsFwd(t,q,p_obj)"""
    u, a_des = hopperStateControl(t, q, p_obj)
    R = q[4] - p_obj.l_1
    s1 = np.sin(q[2])
    c1 = np.cos(q[2])
    s2 = np.sin(q[3])
    c2 = np.cos(q[3])

    r_sd = p_obj.r_s0 - q[4]
    if r_sd > 0:
        F_k = p_obj.k_l * r_sd + u[0]
    else:
        F_k = p_obj.k_stop * r_sd + u[0] - p_obj.b_stop * q[9]

    if q[1] < 0:
        F_x = -p_obj.b_g * q[5]
        F_z = p_obj.k_g * (-q[1])
        F_z = F_z + max(-p_obj.b_g * q[6], 0.0)
    else:
        F_x = 0.0
        F_z = 0.0

    a = p_obj.l_1 * F_z * s1 - p_obj.l_1 * F_x * c1 - u[1]

    M = np.array([
        [-p_obj.m_l * R, 0, (p_obj.J_l - p_obj.m_l * R * p_obj.l_1) * c1, 0, 0],
        [0, p_obj.m_l * R, (p_obj.J_l - p_obj.m_l * R * p_obj.l_1) * s1, 0, 0],
        [p_obj.m * R, 0, (p_obj.J_l + p_obj.m * R * q[4]) * c1, p_obj.m * R * p_obj.l_2 * c2, p_obj.m * R * s1],
        [0, -p_obj.m * R, (p_obj.J_l + p_obj.m * R * q[4]) * s1, p_obj.m * R * p_obj.l_2 * s2, -p_obj.m * R * c1],
        [0, 0, p_obj.J_l * p_obj.l_2 * np.cos(q[2] - q[3]), -p_obj.J * R, 0]
    ])

    eta = np.array([
        a * c1 - R * (F_x - F_k * s1 - p_obj.m_l * p_obj.l_1 * q[7] * q[7] * s1),
        a * s1 + R * (p_obj.m_l * p_obj.l_1 * q[7] * q[7] * c1 + F_z - F_k * c1 - p_obj.m_l * p_obj.g),
        a * c1 + R * F_k * s1 + p_obj.m * R * (q[4] * q[7] * q[7] * s1 + p_obj.l_2 * q[8] * q[8] * s2 - 2 * q[9] * q[7] * c1),
        a * s1 - R * (F_k * c1 - p_obj.m * p_obj.g) - p_obj.m * R * (2 * q[9] * q[7] * s1 + q[4] * q[7] * q[7] * c1 + p_obj.l_2 * q[8] * q[8] * c2),
        a * p_obj.l_2 * np.cos(q[2] - q[3]) - R * (p_obj.l_2 * F_k * np.sin(q[3] - q[2]) + u[1])
    ])

    qdd = np.linalg.solve(M, eta)
    xdot = np.concatenate([q[5:10], qdd])

    structOut = {}
    structOut['stated'] = xdot
    structOut['u'] = u
    structOut['a_des'] = a_des
    structOut['r_sd'] = r_sd
    structOut['fsm_state'] = p_obj.fsm_state

    return structOut


# ============================================================================
# DYNAMICS WRAPPER (from hopperDynamics.m)
# ============================================================================
def hopperDynamics(t, q, p_obj):
    """function qdot = hopperDynamics(t,q,p_obj)"""
    structFwd = hopperDynamicsFwd(t, q, p_obj)
    qdot = structFwd['stated']
    return qdot


# ============================================================================
# EVENTS (from eventsHopperControl.m)
# ============================================================================
def eventsHopperControl(t, q, param):
    """function [value,isterminal,direction] = eventsHopperControl(t,q,param)"""
    COMPRESSION = param.FSM_COMPRESSION
    THRUST = param.FSM_THRUST
    FLIGHT = param.FSM_FLIGHT

    y_foot = q[1]
    ddt_leg = q[9]
    l = q[4]
    thresh_leg_extended = 0.0001

    if param.fsm_state == COMPRESSION:
        value = ddt_leg  # go from compression to thrust when leg stops compressing
    elif param.fsm_state == THRUST:
        value = -(param.r_s0 - l) - thresh_leg_extended  # when leg is fully extended
    elif param.fsm_state == FLIGHT:
        value = -y_foot  # touchdown
    else:
        # Should not reach here in normal FSM operation
        value = 1.0  # Return a value that won't trigger

    return value


# ============================================================================
# ENERGY CALCULATION (from hopperEnergy.m)
# ============================================================================
def hopperEnergy(t, State, P):
    """function out = hopperEnergy(t,State,P)"""
    from scipy.integrate import cumulative_trapezoid

    x_foot = State['x_foot']
    z_foot = State['z_foot']
    phi_leg = State['phi_leg']
    phi_body = State['phi_body']
    len_leg = State['len_leg']
    ddt_x_foot = State['ddt_x_foot']
    ddt_z_foot = State['ddt_z_foot']
    ddt_phi_leg = State['ddt_phi_leg']
    ddt_phi_body = State['ddt_phi_body']
    ddt_len_leg = State['ddt_len_leg']

    rs_d = P.r_s0 - len_leg

    x_leg = x_foot + P.l_1 * np.sin(phi_leg)
    z_leg = z_foot + P.l_1 * np.cos(phi_leg)

    ddt_comx_leg = ddt_x_foot + P.l_1 * np.cos(phi_leg) * ddt_phi_leg
    ddt_comz_leg = ddt_z_foot - P.l_1 * np.sin(phi_leg) * ddt_phi_leg

    ddt_comx_body = ddt_x_foot + ddt_len_leg * np.sin(phi_leg) + \
        len_leg * np.cos(phi_leg) * ddt_phi_leg + \
        P.l_2 * np.cos(phi_body) * ddt_phi_body

    z_body = z_foot + len_leg * np.cos(phi_leg) + P.l_2 * np.cos(phi_body)
    ddt_comz_body = ddt_z_foot + ddt_len_leg * np.cos(phi_leg) - \
        len_leg * np.sin(phi_leg) * ddt_phi_leg - \
        P.l_2 * np.sin(phi_body) * ddt_phi_body

    E_kin_body = 0.5 * P.m * (ddt_comx_body**2 + ddt_comz_body**2) + \
        0.5 * P.J * ddt_phi_body**2

    E_kin_leg = 0.5 * P.m_l * (ddt_comx_leg**2 + ddt_comz_leg**2) + \
        0.5 * P.J_l * ddt_phi_leg**2

    E_g_body = P.m * P.g * z_body
    E_g_leg = P.m_l * P.g * z_leg

    power_u1 = State['u'][:, 0] * ddt_len_leg
    power_u2 = State['u'][:, 1] * (ddt_phi_body - ddt_phi_leg)

    work_u1 = cumulative_trapezoid(power_u1, t, initial=0)
    work_u2 = cumulative_trapezoid(power_u2, t, initial=0)

    E_leg_spring = 0.5 * P.k_l * (rs_d > 0) * rs_d**2 + \
        0.5 * P.k_stop * (rs_d <= 0) * rs_d**2

    E_foot_spring_z = 0.5 * P.k_g * (z_foot < 0) * (z_foot)**2

    E_leg_damp = cumulative_trapezoid(-1 * P.b_stop * (rs_d <= 0) * ddt_len_leg * ddt_len_leg, t, initial=0)

    E_foot_damp_x = cumulative_trapezoid((-1 * P.b_g * (z_foot < 0) * ddt_x_foot) * ddt_x_foot, t, initial=0)
    F_foot_damp_z = np.maximum(-1 * P.b_g * (z_foot < 0) * ddt_z_foot, 0)
    E_foot_damp_z = cumulative_trapezoid(F_foot_damp_z * ddt_z_foot, t, initial=0)

    out = {}
    out['E_kin_leg'] = E_kin_leg
    out['E_kin_body'] = E_kin_body
    out['E_leg_spring'] = E_leg_spring
    out['E_leg_damp'] = E_leg_damp
    out['E_g_body'] = E_g_body
    out['E_g_leg'] = E_g_leg
    out['work_u1'] = work_u1
    out['work_u2'] = work_u2
    out['E_foot_spring_z'] = E_foot_spring_z
    out['E_foot_damp_z'] = E_foot_damp_z
    out['E_foot_damp_x'] = E_foot_damp_x
    out['E_m'] = E_kin_body + E_kin_leg + E_leg_spring + E_foot_spring_z + E_g_body + E_g_leg
    out['E_loss'] = out['E_foot_damp_z'] + out['E_foot_damp_x'] + out['E_leg_damp']
    out['E_gain'] = work_u1 + work_u2
    out['E_net_flow'] = out['E_gain'] + out['E_loss']
    out['E_delta'] = out['E_m'] - out['E_m'][0]

    fig = plt.figure()
    ms = 2
    plt.plot(t, out['E_net_flow'], linewidth=3, marker='o', markersize=ms)
    plt.plot(t, out['E_delta'], linewidth=1, marker='o', markersize=ms)
    plt.plot(t, State['fsm_state'])
    plt.plot(t, State['z_foot'] * 50)
    plt.plot(t, out['E_foot_damp_z'])
    plt.legend(['gain-loss', 'mech-diff', 'fsm', 'zfoot', 'dampz'])
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.title('Energy Accounting')
    plt.grid(True)

    return out, fig


# ============================================================================
# DRAW FUNCTION (from draw.m)
# ============================================================================
def draw(p_obj, t, q):
    """function draw(p_obj,t,q)"""
    l_1 = p_obj.l_1
    l_2 = p_obj.l_2

    if not hasattr(draw, 'hFig'):
        draw.hFig = None
        draw.xcamera = 0

    if draw.hFig is None:
        draw.hFig = plt.figure(25)
        draw.brick1 = np.array([[0.9, 1.05, 1.05, 0.9, 0.9], [0.15, 0.15, 0.0, 0.0, 0.15]])
        draw.brick2 = np.array([[-1, 0], [0, 1]]) @ draw.brick1
        draw.beam = np.array([[-1, -1, 1, 1], [0.15, 0.175, 0.175, 0.15]])
        draw.comp = np.array([[0.25, 0.25, 0.025, 0.025, -0.025, -0.025, -0.25, -0.25],
                               [0.15, 0.3, 0.3, 0.15, 0.15, 0.3, 0.3, 0.15]])
        draw.body = np.hstack([draw.brick1, draw.brick2, draw.beam, draw.comp])
        draw.body = draw.body - np.array([[0], [0.16]]) @ np.ones((1, draw.body.shape[1]))
        draw.leg = np.array([[-0.015, -0.04, -0.04, 0.04, 0.04, 0.015, -0.015],
                              [-0.25, -0.25, 0.525, 0.525, -0.25, -0.25, -0.25]])
        draw.leg = draw.leg - np.array([[0], [0.16]]) @ np.ones((1, draw.leg.shape[1]))
        draw.foot = np.array([[0.0, -0.03, -0.03, 0.015, 0.015, -0.025, -0.025, 0.025, 0.025, -0.015, -0.015, 0.03, 0.03, 0.0],
                               [0.0, 0.05, 0.14, 0.14, 1.675, 1.675, 1.725, 1.725, 1.675, 1.675, 0.14, 0.14, 0.05, 0.0]])

    hip = np.array([0, q[4]])
    rot = np.array([[np.cos(q[2]), np.sin(q[2])], [-np.sin(q[2]), np.cos(q[2])]])
    hip = rot @ hip + np.array([q[0], q[1]])
    cop = np.array([q[0], q[1]])

    Foot = rot @ draw.foot + np.array([[q[0]], [q[1]]])
    Leg = rot @ draw.leg + hip.reshape(2, 1)
    rot = np.array([[np.cos(q[3]), np.sin(q[3])], [-np.sin(q[3]), np.cos(q[3])]])
    Body = rot @ draw.body + hip.reshape(2, 1)

    plt.figure(draw.hFig.number)
    plt.clf()

    draw.xcamera = np.mean([hip[0], draw.xcamera])
    xmin = draw.xcamera - 5
    xmax = draw.xcamera + 5
    ymin = -0.25
    ymax = 3.0

    plt.plot([xmin, xmax], [0, 0], 'k-')
    plt.axis('equal')
    plt.axis([xmin, xmax, ymin, ymax])

    plt.plot(Foot[0, :], Foot[1, :], 'k-')
    plt.plot(Leg[0, :], Leg[1, :], 'k-')
    plt.plot(Body[0, :], Body[1, :], 'k-')
    c = 1 - 0.125 * t
    c = 0.75
    plt.fill(Foot[0, :], Foot[1, :], color=[c, c, c])
    plt.fill(Leg[0, :], Leg[1, :], color=[c, c, c])
    plt.fill(Body[0, :], Body[1, :], color=[c, c, c])

    plt.title(f't={t:.3f} s')


# ============================================================================
# MAIN CALL_HOPPER SCRIPT (from call_hopper.m)
# ============================================================================
def call_hopper(tstart=0, tfinal=5, x_dot_des=3.0, y0=None, sr_sim=1000, save_figures=True, figure_path='./figures'):
    """Main simulation script - transcribed from call_hopper.m

    Args:
        tstart (float): Start time for simulation (default: 0)
        tfinal (float): Final time for simulation (default: 5)
        x_dot_des (float): Desired forward velocity (default: 3.0)
        y0 (np.array): Initial state vector [x_foot, z_foot, phi_leg, phi_body, len_leg, velocities...]
                       (default: [0.0, 0.4, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        sr_sim (int): Simulation sample rate (default: 1000)
        save_figures (bool): Whether to save figures to disk (default: True)
        figure_path (str): Directory path to save figures (default: './figures')
    """
    import os

    # Set default initial conditions if not provided
    if y0 is None:
        y0 = np.array([0.0, 0.4, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Create figure directory if saving is enabled
    if save_figures and not os.path.exists(figure_path):
        os.makedirs(figure_path)

    p = hopperParameters()
    p.x_dot_des = x_dot_des
    p.t_state_switch = tstart
    tout = []
    yout = []
    teout = []
    yeout = []
    ieout = []
    cstate = []
    at_des = []
    extra_states = []
    tic = time.time()

    while tfinal-tstart > 1e-3:
        # CRITICAL: Create a deep copy of p for this solve_ivp call
        # Python lambdas capture references, not values, so we need to snapshot p
        p_snapshot = copy.deepcopy(p)

        # Create event function for solve_ivp
        def event_func(t, y):
            e = eventsHopperControl(t, y, p_snapshot)
            print(f"t={t:.6f}, Event value={e:.6e}, y[1]={y[1]:.6e}, y[9]={y[9]:.6e}")
            return e
        event_func.terminal = True
        event_func.direction = 1  # Only detect increasing crossings (like MATLAB)

        sol = solve_ivp(
            lambda t, y: hopperDynamics(t, y, p_snapshot),
            [tstart, tfinal],
            y0,
            method='Radau',  # Higher-order method, more robust than RK45 near events
            events=event_func,
            rtol=1e-8,
            atol=1e-8,
            max_step=0.05,  # MATLAB default MaxStep
            dense_output=True,
            vectorized=False  # Ensure event function is called correctly
        )

        t = np.concatenate([np.arange(tstart, sol.t[-1], 1/sr_sim), [sol.t[-1]]])
        states = sol.sol(t)
        states = states.T
        t = t[:-1]
        tout.append(t)

        if len(sol.t_events[0]) > 0:
            cstate.append(np.repeat(p.fsm_state, len(t)))
        else:
            if len(cstate) > 0:
                cstate.append(np.repeat(cstate[-1][-1], len(t)))
            else:
                cstate.append(np.repeat(p.fsm_state, len(t)))

        yout.append(states[:-1, :])

        if len(sol.t_events[0]) > 0:
            teout.append(sol.t_events[0])
        if len(sol.y_events[0]) > 0:
            yeout.append(sol.y_events[0])
        if len(sol.t_events[0]) > 0:
            ieout.append(np.ones(len(sol.t_events[0])))

        if len(sol.y_events[0]) > 0 and p.fsm_state == p.FSM_FLIGHT:
            strOut = hopperDynamicsFwd(sol.t_events[0][0], sol.y_events[0][0], p)
            at_des.append([sol.t_events[0][0], strOut['a_des']])

        int_e = []
        for iint in range(len(t)):
            int_states = hopperDynamicsFwd(t[iint], states[iint, :], p)
            int_e.append([int_states['u'][0], int_states['u'][1], int_states['fsm_state']])
        int_e = np.array(int_e)
        extra_states.append(int_e)

        if p.fsm_state == p.FSM_THRUST:
            p.T_s = (t[-1] - tstart) + p.T_compression
        elif p.fsm_state == p.FSM_COMPRESSION:
            p.T_compression = t[-1] - tstart

        if p.fsm_state == p.FSM_COMPRESSION:
            p.t_thrust_on = t[-1]

        if len(sol.t_events[0]) > 0:
            p.fsm_state = p.fsm_state + 1
            p.fsm_state = p.fsm_state % p.FSM_NUM_STATES

        y0 = yout[-1][-1, :]
        tstart = t[-1]

    toc = time.time()
    print(f'Elapsed time: {toc - tic:.2f} seconds')

    # Concatenate all the accumulated arrays
    tout = np.concatenate(tout)
    yout = np.vstack(yout)
    cstate = np.concatenate(cstate)
    extra_states = np.vstack(extra_states)
    if len(at_des) > 0:
        at_des = np.array(at_des)

    fig1 = plt.figure()
    d_xfoot_dt = yout[:, 5]
    dldt = yout[:, 9]
    l = yout[:, 4]
    dadt = yout[:, 7]
    a = yout[:, 2]
    b = yout[:, 3]
    dbdt = yout[:, 8]
    d_xbody_dt = d_xfoot_dt + dldt * np.sin(a) + l * dadt * np.cos(a) + p.l_2 * dbdt * np.cos(b)
    plt.subplot(2, 1, 1)
    plt.plot(tout, d_xbody_dt)
    xl = plt.xlim()
    plt.plot([xl[0], xl[1]], [p.x_dot_des, p.x_dot_des])
    plt.ylabel('Body velocity (m/s)')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(tout, yout[:, 2])
    if len(at_des) > 0:
        plt.plot(at_des[:, 0], at_des[:, 1], 'rx')
    plt.ylabel('Leg angle (rad)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    if save_figures:
        fig1.savefig(os.path.join(figure_path, 'velocity_and_angle.png'), dpi=300, bbox_inches='tight')
        print(f"Saved figure: {os.path.join(figure_path, 'velocity_and_angle.png')}")

    doStatePlot = 0
    if doStatePlot:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(tout, yout[:, 0:5])
        plt.subplot(2, 1, 2)
        plt.plot(tout, yout[:, 5:10])

    doEventsPlot = 0
    if doEventsPlot:
        events_mat = np.column_stack([-yout[:, 9] - 0.2, -(p.r_s0 - yout[:, 4]) - 0.0001, yout[:, 1], -yout[:, 1]])
        plt.figure()
        plt.plot(tout, events_mat)

    fig2 = plt.figure()
    ah = []
    ah.append(plt.subplot(2, 1, 1))
    plt.plot(tout, yout[:, 1:4])
    plt.legend(['yfoot', 'foot angle', 'body angle'])
    plt.ylabel('Position/Angle')
    plt.grid(True)
    ah.append(plt.subplot(2, 1, 2))
    feedback_ang = yout[:, 3] - yout[:, 2] / 2
    plt.plot(tout, feedback_ang, 'r')
    plt.ylabel('Feedback angle (rad)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    xl = plt.xlim()
    plt.plot([xl[0], xl[1]], [0, 0])
    if save_figures:
        fig2.savefig(os.path.join(figure_path, 'angles_and_position.png'), dpi=300, bbox_inches='tight')
        print(f"Saved figure: {os.path.join(figure_path, 'angles_and_position.png')}")

    # Animation setup
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    sr_video = 24
    dur_frame = 1 / sr_video
    ds = round(sr_sim / sr_video)
    tout_vid = tout[::ds]
    yout_vid = yout[::ds, 0:5]

    print(f"\nAnimating {len(yout_vid)} frames...")

    # Create animation figure
    anim_fig = plt.figure(25)

    def animate(i):
        """Animation function for each frame"""
        draw(p, tout_vid[i], yout_vid[i, 0:5])
        return []

    # Create animation
    anim = FuncAnimation(anim_fig, animate, frames=len(yout_vid),
                        interval=dur_frame*1000, blit=False, repeat=True)

    # Save animation if requested
    if save_figures:
        animation_file = os.path.join(figure_path, 'animation.mp4')
        print(f"Saving animation to {animation_file}...")
        try:
            writer = FFMpegWriter(fps=sr_video, metadata=dict(artist='HopperSim'), bitrate=1800)
            anim.save(animation_file, writer=writer)
            print(f"Animation saved successfully to {animation_file}")
        except Exception as e:
            print(f"Warning: Could not save animation. Error: {e}")
            print("Make sure ffmpeg is installed: conda install -c conda-forge ffmpeg")

    # Display animation
    plt.ion()
    for i in range(len(yout_vid)):
        tic_frame = time.time()
        draw(p, tout_vid[i], yout_vid[i, 0:5])
        dur_draw = time.time() - tic_frame
        if dur_draw < dur_frame:
            plt.pause(dur_frame - dur_draw)
        else:
            plt.pause(0.001)  # Minimum pause to update display
        plt.draw()

    print("Animation complete!")

    State = {}
    State['x_foot'] = yout[:, 0]
    State['z_foot'] = yout[:, 1]
    State['phi_leg'] = yout[:, 2]
    State['phi_body'] = yout[:, 3]
    State['len_leg'] = yout[:, 4]
    State['ddt_x_foot'] = yout[:, 5]
    State['ddt_z_foot'] = yout[:, 6]
    State['ddt_phi_leg'] = yout[:, 7]
    State['ddt_phi_body'] = yout[:, 8]
    State['ddt_len_leg'] = yout[:, 9]
    State['u'] = extra_states[:, 0:2]
    State['fsm_state'] = extra_states[:, 2]

    e, energy_fig = hopperEnergy(tout, State, p)

    if save_figures:
        energy_fig.savefig(os.path.join(figure_path, 'energy_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"Saved figure: {os.path.join(figure_path, 'energy_analysis.png')}")

    return tout, yout, State, p, anim


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    # Run the main simulation
    tout, yout, State, p = call_hopper()

    # Print final state for verification
    print("\n=== Final State ===")
    print(f"Final time: {tout[-1]:.4f} s")
    print(f"Final position (x_foot, z_foot): ({yout[-1, 0]:.4f}, {yout[-1, 1]:.4f})")
    print(f"Final angles (phi_leg, phi_body): ({yout[-1, 2]:.4f}, {yout[-1, 3]:.4f})")
    print(f"Final leg length: {yout[-1, 4]:.4f}")

    plt.show()
