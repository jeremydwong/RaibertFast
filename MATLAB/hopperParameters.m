function p = hopperParameters()
p = struct;
p.m = 10.0;        % mass of the body
p.m_l = 1.0;       % mass of the leg
p.J = 10.0;        % moment of inertia of the body
p.J_l = 1.0;       % moment of inertia of the leg
p.g = 9.8;         % gravity
p.k_l = 1e3;       % spring constant of leg spring
p.k_stop = 1e5;    % spring constant of leg stop
p.b_stop = 1e3;    % damping constant of leg stop
p.k_g = 1e4;       % spring constant of the ground
p.b_g = 300.0;     % damping constant of the ground
p.r_s0 = 1.0;      % rest length of the leg spring
p.l_1 = 0.5;       % distance from the foot to the com of the leg
p.l_2 = 0.4;       % distance from the hip to the com of the body

p.FSM_COMPRESSION = 0;
p.FSM_THRUST = 1;
p.FSM_LOADING = 99;
p.FSM_FLIGHT = 2;
p.FSM_NUM_STATES = 3;

%%state parameters
p.fsm_state = p.FSM_FLIGHT; %usually we start in the air?
p.t_state_switch = 0.0;
p.x_dot_des = 0.0;
p.T_s = 0.425;%Tedrake
p.T_compression = 0;
%
p.t_thrust_on = 0;
p.T_MAX_THRUST_DUR = .425*.35; %relative to Tedrake.

% p.

p.toString = '1: x foot 2: y foot 3: abs angle leg (vert) 4: abs angle body (vert) 5:leg length';
%   q(3) - absolute angle of leg (from vertical)
%   q(4) - absolute angle of body
%   q(5) - leg length
%   q(6:10) - derivatives of q(1:5)

% control variables:
%   u(1) - position of leg spring actuator
%   u(2) - torque at the hip

