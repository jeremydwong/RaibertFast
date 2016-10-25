clear all;
tstart = 0;
tfinal = 5;
%%
p = hopperParameters();
p.x_dot_des = 3; %m/s
% y0 = [0.0; 0.4; 0.01; 0.0; 1.0; zeros(5,1)];
y0 = [0.0; 0.4; 0.01; 0.0; 1.0; zeros(5,1)]; %standard start 20160501.
% y0 = [0.0; 0.4; 0.00; 0.0; 1.0; zeros(5,1)]; %vertical.
p.t_state_switch = tstart;
options = odeset('Events',@eventsHopperControl,'reltol',1e-8,'abstol',1e-8);
sr_sim = 1000;
%%
[tout,yout,teout,yeout,ieout,cstate,at_des,extra_states] = deal([]);
tic;
while tstart < tfinal
    % Solve until the first terminal event.
    [sol] = ode45(@hopperDynamics,[tstart,tfinal],y0,options,p);
    % Accumulate output.
    t = [tstart:1/sr_sim:sol.x(end),sol.x(end)];
    states = deval(sol,t);
    states = states';%now n_t x n_states.
    t = t(1:end-1);
    tout = [tout; t'];
    if ~isempty(sol.ie)
        cstate = [cstate;repmat(p.fsm_state,length(t),1,1)];
    else
        cstate = [cstate;repmat(cstate(end),length(t),1,1)];
    end;
    yout = [yout; states(1:end-1,:)];
    teout = [teout(:); sol.xe];    % Events at tstart are never reported.
    yeout = [yeout(:); sol.ye];
    ieout = [ieout(:); sol.ie];
    
    % grab control parameters.
    if ~isempty(sol.ye) && p.fsm_state == p.FSM_FLIGHT
        strOut = hopperDynamicsFwd(sol.xe,sol.ye,p);
        at_des = [at_des;sol.xe,strOut.a_des];
    end;
    
    % get internal states
    int_e = [];
    for iint =1:length(t)
        int_states = hopperDynamicsFwd(t(iint),states(iint,1:end)',p);
        int_e(iint,:) = [int_states.u',int_states.fsm_state];
    end;
    extra_states = [extra_states;int_e];
    
    % FSMvars: set p.T_s and T_compression
    switch p.fsm_state %the state that we are just leaving!
        case p.FSM_THRUST
            p.T_s = (t(end)-tstart) + p.T_compression;
            %             fprintf(['resetting T_s to ',num2str(p.T_s),'.\n']);
        case p.FSM_COMPRESSION
            p.T_compression = t(end)-tstart;
    end;
    
    % turn on thrust for a finite amount of time.     
    % this seems to make sense since otherwise we will just keep pushing
    % longer and longer. This will not guarantee that we are only
    % putting a set amount of energy in but it's kind of close.
    switch p.fsm_state
        case p.FSM_COMPRESSION % the state before thrust.
            p.t_thrust_on = t(end); 
    end;
    
    % move ahead state machine
    if (~isempty(sol.ie))
        p.fsm_state = p.fsm_state+1;
        p.fsm_state = mod(p.fsm_state,p.FSM_NUM_STATES);
    end;
    
    % Set the new initial conditions.
    y0 = yout(end,:);
    tstart = t(end);
    
    %   BEG: DONT DO THIS YET. from mathworks.
    %   % A good guess of a valid first time step is the length of
    %   % the last valid time step, so use it for faster computation.
    %   options = odeset(options,'InitialStep',t(nt)-t(nt-refine),...
    %                            'MaxStep',t(nt)-t(1));
    %   END: DON'T DO THIS.
end
toc;
%% we should be able to check that the u's are correct. 


%% VELOCITY CONTROL. check COM velocity
% and our control of velocity, which is foot placement
figure;
d_xfoot_dt = yout(:,6);dldt = yout(:,10);l = yout(:,5);dadt = yout(:,8);
a = yout(:,3);b = yout(:,4);dbdt=yout(:,9);
d_xbody_dt = d_xfoot_dt + dldt .* sin(a) + l .* dadt .* cos(a) + p.l_2 * dbdt .* cos(b);
subplot(2,1,1);
plot(tout,d_xbody_dt);
hold on;
xl = xlim;
line([xl(1),xl(2)],[p.x_dot_des,p.x_dot_des]);
% plot desired angle of the foot against actual angle at collision.
subplot(2,1,2);
plot(tout,yout(:,3))
hold on;
plot(at_des(:,1),at_des(:,2),'rx')
%%
doStatePlot = 0;
if doStatePlot
    figure;
    subplot(2,1,1);plot(tout,yout(:,1:5))
    subplot(2,1,2);plot(tout,yout(:,6:10));
end;
%%
doEventsPlot = 0;
if doEventsPlot
    events_mat = [-yout(:,10)-.2,-(p.r_s0 - yout(:,5))-.0001,yout(:,2),-yout(:,2)];
    figure;plot(tout, events_mat);
end;
%% body control during contact.
figure;
ah =[];
ah(1)=subplot(2,1,1);
plot(tout,yout(:,2:4));legend('yfoot','foot angle','body angle');
grid on;
ah(2)=subplot(2,1,2);
feedback_ang = yout(:,4)-yout(:,3)/2;
plot(tout,feedback_ang,'r'); grid on; line([xlim],[0,0]);
linkaxes(ah,'x');
%%
sr_video = 24;
dur_frame = 1/sr_video;
ds = round(sr_sim/sr_video);
tout_vid = tout(1:ds:length(tout));
yout_vid = yout(1:ds:length(yout),1:5);
for i =1:length(yout_vid)
    tic;
    draw(p,tout_vid(i),yout_vid(i,1:5));
    dur_draw = toc;
    if dur_draw<dur_frame
        pause(dur_frame-dur_draw);
    end;
    drawnow;
end;

%%
State = struct;
State.x_foot = yout(:,1);
State.z_foot = yout(:,2);
State.phi_leg = yout(:,3);
State.phi_body = yout(:,4);
State.len_leg = yout(:,5);
State.ddt_x_foot = yout(:,6);
State.ddt_z_foot = yout(:,7);
State.ddt_phi_leg = yout(:,8);
State.ddt_phi_body = yout(:,9);
State.ddt_len_leg = yout(:,10);
State.u = extra_states(:,1:2);
State.fsm_state = extra_states(:,3);
% NEAR_0 = 1e-5;
% ind_near_0 = abs(State.u(:,1)) < NEAR_0;
% 20160123
% condition u to cover numerical errors about 0? 
% cumtrapz produces improper integral of input force axial to leg.
% State.u(ind_near_0) = 0;
%%
e = hopperEnergy(tout,State,p);
%%
plot(em);