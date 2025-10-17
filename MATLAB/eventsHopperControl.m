function [value,isterminal,direction] = eventsHopperControl(t,q,param)
% using ONLY these state variables:
%   q(2) - y position of the foot
%   q(5) - leg length
%   q(6:10) - derivatives of q(1:5)

% as well as 
% rest spring length r_s0
% threshold for leg being extended
thresh_leg_extended = 0.0001;

isterminal = 1;
direction = 1; %only when the value is increasing.

y_foot  = q(2);
l       = q(5);
ddt_leg = q(10);
% DDT_OFFSET= .2;
switch param.fsm_state
    case param.FSM_COMPRESSION
        value = ddt_leg; %-DDT_OFFSET; %go from compression to thrust when leg stops compressing.
    case param.FSM_THRUST
        value = -(param.r_s0 - l) - thresh_leg_extended; %when leg is fully extended
    % case param.FSM_LOADING
    %     value = y_foot; % robot has left the ground
    case param.FSM_FLIGHT
        value = -y_foot;% touchdown
end



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

