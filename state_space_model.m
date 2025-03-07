% In state_space_model.m
function [A, B, C, D] = create_state_space_model(m, k, c)
    % m: mass, k: spring constant, c: damping coefficient
    % States: [position; velocity]
    A = [0, 1; -k/m, -c/m];
    B = [0; 1/m];
    C = eye(2);  % Output both position and velocity
    D = zeros(2, 1);
end