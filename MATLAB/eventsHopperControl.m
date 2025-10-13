function [value,isterminal,direction] = events1HopperControl(t,q,param)
% state variables:
%   q(1) - x position of the foot
%   q(2) - y position of the foot
%   q(3) - absolute angle of leg (from vertical)
%   q(4) - absolute angle of body
%   q(5) - leg length
%   q(6:10) - derivatives of q(1:5)

% control variables:
%   u(1) - position of leg spring actuator
%   u(2) - torque at the hip

isterminal = 1;
direction = 1; %only when the value is increasing.



COMPRESSION = param.FSM_COMPRESSION;
THRUST = param.FSM_THRUST;
UNLOADING = param.FSM_LOADING;
FLIGHT = param.FSM_FLIGHT;

y_foot = q(2);
ddt_leg = q(10);
% state_ctrl = param.state_ctrl;
%four state fsm.
% value(i) stores the transition condition from i to i+1.
% DDT_OFFSET= .2;
switch param.fsm_state
    case COMPRESSION
        value = ddt_leg; %-DDT_OFFSET; %go from compression to thrust when leg stops compressing.
    case THRUST
        value = -(param.r_s0 - q(5))-.0001; %go from thrust to unloading when leg elongates.
    case UNLOADING
        value = y_foot; %transition from unloading to flight
    case FLIGHT
        value = -y_foot;%from flight to compression when foot enters ground.
end;



%%%%%%%%%%%%%%%%%%%% END
% value = zeros(4,1);
% % EVENTS DOC:
% value, isterminal, and direction are vectors for which the ith element corresponds to the ith event function:
%
% value(i) is the value of the ith event function.
% isterminal(i) = 1 if the integration is to terminate at a zero of this event function, otherwise, 0.
% direction(i) = 0 if all zeros are to be located (the default), +1 if only zeros where the event function is increasing, and -1 if only zeros where the event function is decreasing.
%
% If you specify an events function and events are detected, the solver returns three additional outputs:
% TE: A column vector of times at which events occur
% YE: Solution values corresponding to these times
% IE: Indices into the vector returned by the events function. The values indicate which event the solver detected.
% If you call the solver as
% [T,Y,TE,YE,IE] = solver(odefun,tspan,y0,options)
% the solver returns these outputs as TE, YE, and IE respectively. If you call the solver as
% sol = solver(odefun,tspan,y0,options)
% the solver returns these outputs as sol.xe, sol.ye, and sol.ie, respectively.

