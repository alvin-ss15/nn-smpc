% In simulate_system.m
function [t, x, u] = simulate_system(x0, time_span, input_func, params)
    % Configure ODE solver
    options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
    
    % Anonymous function for ODE solver
    odefun = @(t, x) mass_spring_damper_dynamics(t, x, input_func(t), params);
    
    % Solve ODE
    [t, x] = ode45(odefun, time_span, x0, options);
    
    % Calculate inputs for each time point
    u = zeros(length(t), 1);
    for i = 1:length(t)
        u(i) = input_func(t(i));
    end
end