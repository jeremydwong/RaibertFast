function [u,a_des] = hopperStateControl(t,q,param)
% function u = raibertStateControl(q,param)
%
% inputs:
% t     -> current time
% q     -> current state
% param -> parameters, 2 can be CHANGED, ARE NEEDED TO REPLICATE:
%   fsm_state
%   T_s
%
% Outputs
% u(1) = force along the leg.
% u(2) = hip actuator to change foot position.
% leg axial force, and hip torque during COMPRESSION, THRUST, and FLIGHT.
%
% control_states (struct)
% a_des     -> desired angle
% fsm_state -> updated fsm_state.
% T_s       -> pushoff duration.
%
% state variables:
%   q(1) - x position of the foot
%   q(2) - y position of the foot
%   q(3) - absolute angle of leg (from vertical)
%   q(4) - absolute angle of body
%   q(5) - leg length
%   q(6:10) - derivatives of q(1:5)

% FSM STATES. Enum values.
THRUST = param.FSM_THRUST;
COMPRESSION = param.FSM_COMPRESSION;
% LOADING = param.FSM_LOADING; % we do not need this state anymore we
% think.
FLIGHT = param.FSM_FLIGHT;
% /FSM_STATES

% CONTROL PARAMETERS. MOSTLY A BUNCH OF GAINS
k_fp    = 150.0;  b_fp = 15.0;  % flight PD leg, originally k=153 and b=14.
k_att   = 150.0; b_att = 15.0;  % ground PD leg. originally k=153 and b=14.
k_xdot = 0.02;                  % forward speed gain. originally .01.
% k_xdot_con = 1-k_xdot;
thrust = 0.035 * param.k_l;     % originally 0.035 m (u1 altered l_spring not Force).
thr_z_low = 0.01;               % if we are above this height.
u_retract = -.1 * param.k_l;    % currently we aren't retracting in flight.
% /PARAMETERS

% use an updated estimate of the contact time, or not.
if param.T_s ==0
    fprintf('Warning: using default T_s.\n');
    T_s = .425;
else
    T_s = param.T_s;
end
%
% PASSTHROUGH_0FORCE = 0;
u=[0;0];
a_des = 0;

% if ~PASSTHROUGH_0FORCE
y_foot          = q(2);
d_xfoot_dt      = q(6);
a = q(3);dadt   = q(8);
b = q(4);dbdt   = q(9);
l=q(5); dldt    = q(10);
stance_ang_des  = a/2;
% unlisted: x->q(1),d_yfoot_dt->q(7)
switch param.fsm_state
    case THRUST
        u(1) = thrust;
        u(2) = -k_att*(b-stance_ang_des) - b_att*dbdt;% previous error.
        if t-param.t_thrust_on > param.T_MAX_THRUST_DUR
            u(1) = 0;
        end

    case COMPRESSION
        %feedback control for orientation.
        u(2) = -k_att*(b-stance_ang_des) - b_att*dbdt;% NOTE: RT has b-a/2 here.
    case FLIGHT
        % position the foot either ahead or behind the COG footprint.
        d_xbody_dt  = d_xfoot_dt + dldt*sin(a) + l*cos(a)*dadt + param.l_2*cos(b)*dbdt;
        a_des       = -asin((1*d_xbody_dt*T_s/2 + k_xdot*(d_xbody_dt - param.x_dot_des))/l);
        if ~imag(a_des)==0
            fprintf('Imaginary foot angle a_des.');
        end
        if y_foot > thr_z_low
            u(2) = k_fp*(a-a_des) + b_fp*(dadt);
            %                 u(1) = u_retract;
        end
end
% end
