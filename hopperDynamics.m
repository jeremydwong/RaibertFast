function qdot = hopperDynamics(t,q,p_obj)
% function qdot = hopperDynamics(t,q,p_obj)
% wrapper function returning state derivative.
structFwd = hopperDynamicsFwd(t,q,p_obj);
qdot = structFwd.stated;