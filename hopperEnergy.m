function out = hopperEnergy(t,State,P)
% state variables:
% x_foot = State.x_foot;
% z_foot = State.z_foot;
% phi_leg = State.phi_leg;
% phi_body = State.phi_body;
% len_leg = State.len_leg;
% ddt_x_foot = State.ddt_x_foot;
% ddt_z_foot = State.ddt.z_foot;
% ddt_phi_leg = State.ddt_phi_leg;
% ddt_phi_body = State.ddt_phi_body;
% ddt_len_leg = State.ddt_len_leg;
% supplementary variables:
% State.u(1)
% State.u(2)
% phewf! a few historical errors that produced the wrong energy calculations:
% 1 - did not compute the power of the dampers. computed the force of the
% dampers and then took the integral, mathematically wrong. 
% 2 - used the wrong expression for the length of the spring.
% 3 - hard to know what the correct sign for the angular velocity between
% segments is, but it's distalv-proximalv. (check energy_powerc)

x_foot = State.x_foot;
z_foot = State.z_foot;
phi_leg = State.phi_leg;
phi_body = State.phi_body;
len_leg = State.len_leg;
ddt_x_foot = State.ddt_x_foot;
ddt_z_foot = State.ddt_z_foot;
ddt_phi_leg = State.ddt_phi_leg;
ddt_phi_body = State.ddt_phi_body;
ddt_len_leg = State.ddt_len_leg;

rs_d = P.r_s0 - len_leg; % delta spring length since r_s0 is resting. 

x_leg = x_foot + P.l_1 * sin(phi_leg); %position of the leg COM.
z_leg = z_foot + P.l_1 * cos(phi_leg); %position of the leg COM.

ddt_comx_leg = ddt_x_foot + P.l_1 * cos(phi_leg) .* ddt_phi_leg;
ddt_comz_leg = ddt_z_foot - P.l_1 * sin(phi_leg) .* ddt_phi_leg;

% x_body = x_foot + len_leg .* sin(phi_leg) + P.l_2 * sin(phi_body);
ddt_comx_body = ddt_x_foot + ddt_len_leg .* sin(phi_leg) + ...
    len_leg .* cos(phi_leg) .* ddt_phi_leg + ...
    P.l_2 * cos(phi_body) .* ddt_phi_body;

z_body = z_foot + len_leg .* cos(phi_leg) + P.l_2 * cos(phi_body);
ddt_comz_body = ddt_z_foot + ddt_len_leg .* cos(phi_leg) - ...
    len_leg .* sin(phi_leg) .* ddt_phi_leg - ...
    P.l_2 * sin(phi_body) .* ddt_phi_body;

E_kin_body = .5 * P.m * (ddt_comx_body.^2 + ddt_comz_body.^2) + ...
    .5 * P.J * ddt_phi_body.^2;

E_kin_leg = .5 * P.m_l * (ddt_comx_leg.^2 + ddt_comz_leg.^2) + ...
    .5 * P.J_l * ddt_phi_leg.^2;

E_g_body = P.m * P.g * z_body;

E_g_leg = P.m_l * P.g * z_leg;

%compute the work done by u1 and u2.
power_u1 = State.u(:,1) .* ddt_len_leg;
power_u2 = State.u(:,2) .* (ddt_phi_body - ddt_phi_leg);

work_u1 = cumtrapz(t,power_u1);
work_u2 = cumtrapz(t,power_u2);

% compute the energy in the leg springs. 
E_leg_spring = 1/2 * P.k_l * (rs_d > 0) .* rs_d.^2 +...
    1/2 * P.k_stop * (rs_d <= 0) .* rs_d.^2;

% compute the energy in the foot spring.
E_foot_spring_z = 1/2*P.k_g*(z_foot<0).* (z_foot).^2; %

% compute the energy dissipated by the leg damper. 
E_leg_damp = cumtrapz(t,-1 * P.b_stop * (rs_d <=0) .* ddt_len_leg .* ddt_len_leg); %not the problem.

% compute the energy dissipated by the foot dampers.
E_foot_damp_x = cumtrapz(t,(-1 * P.b_g * (z_foot<0) .* ddt_x_foot) .* ddt_x_foot); %
F_foot_damp_z = max(-1* P.b_g * (z_foot<0) .* ddt_z_foot,0);
E_foot_damp_z = cumtrapz(t,F_foot_damp_z .* ddt_z_foot); %

out = struct;
out.E_kin_leg = E_kin_leg;
out.E_kin_body = E_kin_body;
out.E_leg_spring = E_leg_spring;
out.E_leg_damp = E_leg_damp;
out.E_g_body = E_g_body;
out.E_g_leg = E_g_leg;
out.work_u1 = work_u1;
out.work_u2 = work_u2;
out.E_leg_spring = E_leg_spring;
out.E_leg_damp = E_leg_damp;
out.E_foot_spring_z = E_foot_spring_z;
out.E_foot_damp_z = E_foot_damp_z;
out.E_foot_damp_x = E_foot_damp_x;
out.E_m = E_kin_body+E_kin_leg+E_leg_spring+E_foot_spring_z+E_g_body+E_g_leg;
out.E_loss = out.E_foot_damp_z+out.E_foot_damp_x+out.E_leg_damp;
out.E_gain = work_u1+work_u2;
out.E_net_flow = out.E_gain + out.E_loss;
out.E_delta = out.E_m-out.E_m(1);

figure;
ms = 2;
plot(t,out.E_net_flow,'linewidth',3,'marker','o','markersize',ms);hold on;plot(t,out.E_delta,'linewidth',1,'marker','o','markersize',ms);
plot(t,State.fsm_state);
plot(t,State.z_foot*50);
plot(t,out.E_foot_damp_z);
legend({'gain-loss','mech-diff','fsm','zfoot','dampz'});