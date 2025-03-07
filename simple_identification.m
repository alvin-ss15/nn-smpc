%% Simple Prediction Models for Mass-Spring-Damper System
% This script implements basic prediction models (linear and polynomial)
% to establish a baseline performance before neural network training

% Clear workspace and close figures
clear;
clc;
close all;

% Display header
disp('===============================================');
disp('Simple Prediction Models for MSD System');
disp('===============================================');

%% 1. Load training data
disp('Loading dataset...');
try
    % Try to load from the new standard location
    load('data/msd_dataset_train.mat');
catch
    % Fallback to the original location
    try
        load('data/training/msd_train_data.mat');
    catch
        error('Could not find training dataset. Please run the dataset generation script first.');
    end
end

%% 2. Select specific subsets of data for analysis
% We'll analyze four different configurations to explore model performance
% across different system characteristics:

% Extract parameter values into arrays for easier indexing
masses = zeros(length(train_data), 1);
spring_constants = zeros(length(train_data), 1);
damping_coeffs = zeros(length(train_data), 1);
input_types = cell(length(train_data), 1);

for i = 1:length(train_data)
    masses(i) = train_data(i).params.mass;
    spring_constants(i) = train_data(i).params.spring_constant;
    damping_coeffs(i) = train_data(i).params.damping_coefficient;
    input_types{i} = train_data(i).input_type;
end

% Configuration 1: Underdamped system with step input
config1_idx = find(strcmp(input_types, 'step') & ...
                   masses == 1 & ...
                   spring_constants == 20 & ...
                   damping_coeffs == 0.5, 1);

% Configuration 2: Overdamped system with step input
config2_idx = find(strcmp(input_types, 'step') & ...
                   masses == 1 & ...
                   spring_constants == 20 & ...
                   damping_coeffs == 15, 1);

% Configuration 3: Underdamped system with sinusoidal input
config3_idx = find(strcmp(input_types, 'sine') & ...
                   masses == 1 & ...
                   spring_constants == 20 & ...
                   damping_coeffs == 0.5, 1);

% Configuration 4: Underdamped system with random input
config4_idx = find(strcmp(input_types, 'random') & ...
                   masses == 1 & ...
                   spring_constants == 20 & ...
                   damping_coeffs == 0.5, 1);

% Check if all indices were found
if isempty(config1_idx) || isempty(config2_idx) || isempty(config3_idx) || isempty(config4_idx)
    warning('Not all configurations were found in the dataset. Using available configurations.');
    % Find some backup configurations
    indices = find([train_data.params.damping_coefficient] <= 5, 4);
    if ~isempty(indices)
        if isempty(config1_idx), config1_idx = indices(1); end
        if isempty(config2_idx), config2_idx = indices(min(2, length(indices))); end
        if isempty(config3_idx), config3_idx = indices(min(3, length(indices))); end
        if isempty(config4_idx), config4_idx = indices(min(4, length(indices))); end
    end
end

% For this tutorial, we'll focus on the first configuration (underdamped step)
config_idx = config1_idx;
current_config = train_data(config_idx);

fprintf('Selected configuration:\n');
fprintf('  Input type: %s\n', current_config.input_type);
fprintf('  Mass: %.1f kg\n', current_config.params.mass);
fprintf('  Spring constant: %.1f N/m\n', current_config.params.spring_constant);
fprintf('  Damping coefficient: %.1f N·s/m\n', current_config.params.damping_coefficient);
fprintf('  Damping ratio: %.3f\n', current_config.params.damping_ratio);

% Extract the time series data
t = current_config.time;
states = current_config.states;
inputs = current_config.inputs;

% Display the selected configuration trajectory
figure('Position', [100, 100, 900, 600]);
subplot(3, 1, 1);
plot(t, states(:, 1), 'b-', 'LineWidth', 1.5);
title('Position Trajectory', 'FontSize', 12);
ylabel('Position (m)', 'FontSize', 10);
grid on;

subplot(3, 1, 2);
plot(t, states(:, 2), 'g-', 'LineWidth', 1.5);
title('Velocity Trajectory', 'FontSize', 12);
ylabel('Velocity (m/s)', 'FontSize', 10);
grid on;

subplot(3, 1, 3);
plot(t, inputs, 'r-', 'LineWidth', 1.5);
title('Input Force', 'FontSize', 12);
xlabel('Time (s)', 'FontSize', 10);
ylabel('Force (N)', 'FontSize', 10);
grid on;

saveas(gcf, 'selected_configuration.png');
saveas(gcf, 'selected_configuration.fig');

%% 3. Prepare data for prediction models
% How many past steps to use for prediction
n_past = 3;

% Prepare data matrices
X = [];  % Features
y = [];  % Targets

% For each time point (starting after n_past)
for i = (n_past+1):length(t)-1
    % Features: past positions, velocities, and inputs
    features = [];
    for j = 0:n_past-1
        features = [features, states(i-j, 1), states(i-j, 2), inputs(i-j)];
    end
    
    % Target: next state
    target = states(i+1, :);
    
    % Add to data matrices
    X = [X; features];
    y = [y; target];
end

% Display the shape of the prepared data
fprintf('Feature matrix shape: %d rows x %d columns\n', size(X, 1), size(X, 2));
fprintf('Target matrix shape: %d rows x %d columns\n', size(y, 1), size(y, 2));

%% 4. Split into training and testing sets
train_ratio = 0.8;
train_size = floor(train_ratio * size(X, 1));

X_train = X(1:train_size, :);
y_train = y(1:train_size, :);
X_test = X(train_size+1:end, :);
y_test = y(train_size+1:end, :);

%% 5A. Linear Regression Model
disp('Training linear regression models...');

% For position prediction
pos_linear_model = fitlm(X_train, y_train(:, 1));

% For velocity prediction
vel_linear_model = fitlm(X_train, y_train(:, 2));

% Test the linear models
pos_linear_pred = predict(pos_linear_model, X_test);
vel_linear_pred = predict(vel_linear_model, X_test);

% Calculate errors
pos_linear_rmse = sqrt(mean((pos_linear_pred - y_test(:, 1)).^2));
vel_linear_rmse = sqrt(mean((vel_linear_pred - y_test(:, 2)).^2));

fprintf('Linear Model Results:\n');
fprintf('  Position prediction RMSE: %.6f\n', pos_linear_rmse);
fprintf('  Velocity prediction RMSE: %.6f\n', vel_linear_rmse);

%% 5B. Polynomial Regression Model
disp('Training polynomial regression models...');

% Create polynomial features (quadratic terms)
poly_degree = 2;

% Create polynomial features for training data
X_poly_train = X_train;
for d = 2:poly_degree
    X_poly_train = [X_poly_train, X_train.^d];
end

% Create polynomial features for test data
X_poly_test = X_test;
for d = 2:poly_degree
    X_poly_test = [X_poly_test, X_test.^d];
end

% Train polynomial models
pos_poly_model = fitlm(X_poly_train, y_train(:, 1));
vel_poly_model = fitlm(X_poly_train, y_train(:, 2));

% Test the polynomial models
pos_poly_pred = predict(pos_poly_model, X_poly_test);
vel_poly_pred = predict(vel_poly_model, X_poly_test);

% Calculate errors
pos_poly_rmse = sqrt(mean((pos_poly_pred - y_test(:, 1)).^2));
vel_poly_rmse = sqrt(mean((vel_poly_pred - y_test(:, 2)).^2));

fprintf('Polynomial Model Results:\n');
fprintf('  Position prediction RMSE: %.6f\n', pos_poly_rmse);
fprintf('  Velocity prediction RMSE: %.6f\n', vel_poly_rmse);

%% 6. Compare linear vs. polynomial predictions
% Calculate improvement percentage
pos_improvement = ((pos_linear_rmse - pos_poly_rmse) / pos_linear_rmse) * 100;
vel_improvement = ((vel_linear_rmse - vel_poly_rmse) / vel_linear_rmse) * 100;

fprintf('Polynomial vs. Linear Model Improvement:\n');
fprintf('  Position prediction: %.2f%% improvement\n', pos_improvement);
fprintf('  Velocity prediction: %.2f%% improvement\n', vel_improvement);

%% 7. Multi-step Prediction Test
% Choose a random starting point in the test set
horizon = 50;  % Number of steps to predict ahead
start_idx = randi([1, length(X_test) - horizon]);

% Initialize arrays for multi-step predictions
pos_linear_future = zeros(horizon, 1);
vel_linear_future = zeros(horizon, 1);
pos_poly_future = zeros(horizon, 1);
vel_poly_future = zeros(horizon, 1);
actual_pos = zeros(horizon, 1);
actual_vel = zeros(horizon, 1);

% Initial conditions (from test data)
current_features = X_test(start_idx, :);

% Get the actual future values for comparison
for i = 1:horizon
    if start_idx + i <= size(y_test, 1)
        actual_pos(i) = y_test(start_idx + i - 1, 1);
        actual_vel(i) = y_test(start_idx + i - 1, 2);
    end
end

% Perform multi-step prediction using both models
for i = 1:horizon
    % Predict next state with linear model
    pos_linear_future(i) = predict(pos_linear_model, current_features);
    vel_linear_future(i) = predict(vel_linear_model, current_features);
    
    % Predict next state with polynomial model
    poly_features = current_features;
    for d = 2:poly_degree
        poly_features = [poly_features, current_features.^d];
    end
    pos_poly_future(i) = predict(pos_poly_model, poly_features);
    vel_poly_future(i) = predict(vel_poly_model, poly_features);
    
    % Update features for next prediction (using linear model prediction)
    if i < horizon
        % Get the next input from test data (assuming we know future inputs)
        next_input = X_test(start_idx + i, end-2);
        
        % Create new feature vector using predictions
        new_features = [pos_linear_future(i), vel_linear_future(i), next_input];
        for j = 1:n_past-1
            new_features = [new_features, current_features(1:3)];
        end
        
        current_features = new_features;
    end
end

%% 8. Visualize Results

% 8A. Single-step Prediction Comparison
figure('Position', [100, 100, 1000, 800]);

% Position prediction comparison
subplot(2, 2, 1);
plot(y_test(1:100, 1), 'b-', 'LineWidth', 1.5);
hold on;
plot(pos_linear_pred(1:100), 'r--', 'LineWidth', 1.5);
plot(pos_poly_pred(1:100), 'g-.', 'LineWidth', 1.5);
title('Position: Single-Step Prediction (First 100 Steps)', 'FontSize', 12);
legend({'Actual', 'Linear Model', 'Polynomial Model'}, 'Location', 'best');
xlabel('Time Step', 'FontSize', 10);
ylabel('Position (m)', 'FontSize', 10);
grid on;

% Velocity prediction comparison
subplot(2, 2, 2);
plot(y_test(1:100, 2), 'b-', 'LineWidth', 1.5);
hold on;
plot(vel_linear_pred(1:100), 'r--', 'LineWidth', 1.5);
plot(vel_poly_pred(1:100), 'g-.', 'LineWidth', 1.5);
title('Velocity: Single-Step Prediction (First 100 Steps)', 'FontSize', 12);
legend({'Actual', 'Linear Model', 'Polynomial Model'}, 'Location', 'best');
xlabel('Time Step', 'FontSize', 10);
ylabel('Velocity (m/s)', 'FontSize', 10);
grid on;

% Position error comparison
subplot(2, 2, 3);
plot(abs(y_test(1:100, 1) - pos_linear_pred(1:100)), 'r-', 'LineWidth', 1.5);
hold on;
plot(abs(y_test(1:100, 1) - pos_poly_pred(1:100)), 'g-', 'LineWidth', 1.5);
title('Position: Absolute Prediction Error', 'FontSize', 12);
legend({'Linear Model Error', 'Polynomial Model Error'}, 'Location', 'best');
xlabel('Time Step', 'FontSize', 10);
ylabel('Absolute Error (m)', 'FontSize', 10);
grid on;

% Velocity error comparison
subplot(2, 2, 4);
plot(abs(y_test(1:100, 2) - vel_linear_pred(1:100)), 'r-', 'LineWidth', 1.5);
hold on;
plot(abs(y_test(1:100, 2) - vel_poly_pred(1:100)), 'g-', 'LineWidth', 1.5);
title('Velocity: Absolute Prediction Error', 'FontSize', 12);
legend({'Linear Model Error', 'Polynomial Model Error'}, 'Location', 'best');
xlabel('Time Step', 'FontSize', 10);
ylabel('Absolute Error (m/s)', 'FontSize', 10);
grid on;

saveas(gcf, 'single_step_prediction_comparison.png');
saveas(gcf, 'single_step_prediction_comparison.fig');

% 8B. Multi-step Prediction Comparison
figure('Position', [100, 100, 1000, 600]);

% Position multi-step prediction
subplot(2, 1, 1);
plot(1:horizon, actual_pos, 'b-', 'LineWidth', 1.5);
hold on;
plot(1:horizon, pos_linear_future, 'r--', 'LineWidth', 1.5);
plot(1:horizon, pos_poly_future, 'g-.', 'LineWidth', 1.5);
title('Position: Multi-Step Prediction', 'FontSize', 12);
legend({'Actual', 'Linear Model', 'Polynomial Model'}, 'Location', 'best');
xlabel('Future Time Step', 'FontSize', 10);
ylabel('Position (m)', 'FontSize', 10);
grid on;

% Velocity multi-step prediction
subplot(2, 1, 2);
plot(1:horizon, actual_vel, 'b-', 'LineWidth', 1.5);
hold on;
plot(1:horizon, vel_linear_future, 'r--', 'LineWidth', 1.5);
plot(1:horizon, vel_poly_future, 'g-.', 'LineWidth', 1.5);
title('Velocity: Multi-Step Prediction', 'FontSize', 12);
legend({'Actual', 'Linear Model', 'Polynomial Model'}, 'Location', 'best');
xlabel('Future Time Step', 'FontSize', 10);
ylabel('Velocity (m/s)', 'FontSize', 10);
grid on;

saveas(gcf, 'multi_step_prediction_comparison.png');
saveas(gcf, 'multi_step_prediction_comparison.fig');

%% 9. Model Coefficients Analysis
% Extract the most significant coefficients for interpretation
disp('Linear Model: Top 5 most significant coefficients for position prediction:');
[~, linear_coef_order] = sort(abs(pos_linear_model.Coefficients.Estimate), 'descend');
linear_coef_table = pos_linear_model.Coefficients(linear_coef_order(1:min(5, length(linear_coef_order))), :);
disp(linear_coef_table);

disp('Polynomial Model: Top 5 most significant coefficients for position prediction:');
[~, poly_coef_order] = sort(abs(pos_poly_model.Coefficients.Estimate), 'descend');
poly_coef_table = pos_poly_model.Coefficients(poly_coef_order(1:min(5, length(poly_coef_order))), :);
disp(poly_coef_table);

%% 10. Summary
fprintf('\n=============== Summary ===============\n');
fprintf('Single-Step Prediction RMSE:\n');
fprintf('  Linear Model - Position: %.6f, Velocity: %.6f\n', pos_linear_rmse, vel_linear_rmse);
fprintf('  Poly Model   - Position: %.6f, Velocity: %.6f\n', pos_poly_rmse, vel_poly_rmse);

% Calculate multi-step prediction errors
multi_step_lin_pos_rmse = sqrt(mean((actual_pos - pos_linear_future).^2));
multi_step_lin_vel_rmse = sqrt(mean((actual_vel - vel_linear_future).^2));
multi_step_poly_pos_rmse = sqrt(mean((actual_pos - pos_poly_future).^2));
multi_step_poly_vel_rmse = sqrt(mean((actual_vel - vel_poly_future).^2));

fprintf('\nMulti-Step Prediction RMSE (%d steps):\n', horizon);
fprintf('  Linear Model - Position: %.6f, Velocity: %.6f\n', multi_step_lin_pos_rmse, multi_step_lin_vel_rmse);
fprintf('  Poly Model   - Position: %.6f, Velocity: %.6f\n', multi_step_poly_pos_rmse, multi_step_poly_vel_rmse);

fprintf('\nImprovement of Polynomial over Linear Model:\n');
fprintf('  Single-Step - Position: %.2f%%, Velocity: %.2f%%\n', pos_improvement, vel_improvement);

% Multi-step improvement
multi_pos_improvement = ((multi_step_lin_pos_rmse - multi_step_poly_pos_rmse) / multi_step_lin_pos_rmse) * 100;
multi_vel_improvement = ((multi_step_lin_vel_rmse - multi_step_poly_vel_rmse) / multi_step_lin_vel_rmse) * 100;
fprintf('  Multi-Step  - Position: %.2f%%, Velocity: %.2f%%\n', multi_pos_improvement, multi_vel_improvement);

fprintf('\nNext Steps:\n');
fprintf('  1. Try other model architectures (e.g., NARX models)\n');
fprintf('  2. Implement neural network models\n');
fprintf('  3. Compare performance across different system configurations\n');
fprintf('  4. Use the models for MPC prediction\n');
fprintf('=======================================\n');