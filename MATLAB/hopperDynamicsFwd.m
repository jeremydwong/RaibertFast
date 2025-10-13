function structOut = hopperDynamicsFwd(t,q,p_obj)
% function structOut = hopperDynamicsFwd()
% state variables:
%   q(1) - x position of the foot
%   q(2) - z position of the foot
%   q(3) - absolute angle of leg (from vertical)
%   q(4) - absolute angle of body
%   q(5) - leg length
%   q(6) - dxdt foot
%   q(7) - dzdt foot
%   q(8) - dphiLeg
%   q(9) - dphiBody
%   q(10) - dLeg

% control variables:
%   u(1) - position of leg spring actuator
%   u(2) - torque at the hip

[u,ctrlStruct] = hopperStateControl(t,q,p_obj);
R = q(5) - p_obj.l_1; %length of the leg minus COMlength of leg.
s1 = sin(q(3));
c1 = cos(q(3));
s2 = sin(q(4));
c2 = cos(q(4));

r_sd = p_obj.r_s0 - q(5);
if (r_sd > 0)
    F_k = p_obj.k_l * r_sd + u(1);
else 
    F_k = p_obj.k_stop * r_sd +u(1) - p_obj.b_stop*q(10);
end

if (q(2) < 0)
    F_x = -p_obj.b_g*q(6);  % don't simulate k_g in horizontal direction
    % (want autonomous dynamics so no horizontal damper)
    F_z = p_obj.k_g*(- q(2)); %spring
    F_z = F_z + max(-p_obj.b_g*q(7),0.0);%damper; damper never pulls down.
else
    F_x = 0.0;
    F_z = 0.0;
end

a = p_obj.l_1*F_z*s1 - p_obj.l_1*F_x*c1 - u(2);

M = [ -p_obj.m_l*R, 0, (p_obj.J_l-p_obj.m_l*R*p_obj.l_1)*c1, 0, 0;
    0, p_obj.m_l*R, (p_obj.J_l-p_obj.m_l*R*p_obj.l_1)*s1, 0, 0;
    p_obj.m*R, 0, (p_obj.J_l+p_obj.m*R*q(5))*c1, p_obj.m*R*p_obj.l_2*c2, p_obj.m*R*s1;
    0, -p_obj.m*R, (p_obj.J_l+p_obj.m*R*q(5))*s1, p_obj.m*R*p_obj.l_2*s2, -p_obj.m*R*c1;
    0, 0, p_obj.J_l*p_obj.l_2*cos(q(3)-q(4)), -p_obj.J*R, 0 ];

eta = [ a*c1 - R*(F_x - F_k*s1 - p_obj.m_l*p_obj.l_1*q(8)*q(8)*s1);
    a*s1 + R*(p_obj.m_l*p_obj.l_1*q(8)*q(8)*c1 + F_z - F_k*c1 - p_obj.m_l*p_obj.g);
    a*c1 + R*F_k*s1 + p_obj.m*R*(q(5)*q(8)*q(8)*s1 + p_obj.l_2*q(9)*q(9)*s2 - 2*q(10)*q(8)*c1);
    a*s1 - R*(F_k*c1 - p_obj.m*p_obj.g) - p_obj.m*R*(2*q(10)*q(8)*s1 + q(5)*q(8)*q(8)*c1 + p_obj.l_2*q(9)*q(9)*c2);
    a*p_obj.l_2*cos(q(3)-q(4)) - R*(p_obj.l_2*F_k*sin(q(4)-q(3)) + u(2)) ];

qdd = M\eta;
xdot = [q(6:10);qdd];

structOut=struct;

structOut.stated = xdot;
%we also want to know
% u, FSMstate, 
structOut.u = u;
structOut.a_des = ctrlStruct.a_des;
structOut.r_sd = r_sd;
structOut.fsm_state = p_obj.fsm_state;