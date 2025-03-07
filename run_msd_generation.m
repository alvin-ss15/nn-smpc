%% Main Script to Run MSD Data Generation
% This script brings everything together to generate the dataset

% Clear workspace and close all figures
clear;
clc;
close all;

% Display startup message
disp('=======================================');
disp('Mass-Spring-Damper Dataset Generation');
disp('=======================================');
disp(' ');

% Make sure the data directories exist
folders = {'data/training', 'data/validation', 'data/testing', ...
           'models/system', 'utils/visualization'};
for i = 1:length(folders)
    if ~exist(folders{i}, 'dir')
        mkdir(folders{i});
    end
end

% Define the functions inline to avoid path issues
% =================== BEGIN DATA PROCESSOR FUNCTIONS ===================

function processed_data = prepare_data_for_nn(data_struct, sequence_length, prediction_horizon)
    % Prepares data from simulation for neural network training
    % 
    % Args:
    %   data_struct: Structure containing simulation results
    %   sequence_length: Number of past time steps to include
    %   prediction_horizon: Number of future time steps to predict
    %
    % Returns:
    %   processed_data: Structure with X_data and Y_data for NN training
    
    num_simulations = length(data_struct);
    all_X = [];
    all_Y = [];
    
    for sim_idx = 1:num_simulations
        % Extract data
        sim = data_struct(sim_idx);
        time = sim.time;
        states = sim.states;
        inputs = sim.inputs;
        
        % Parameter normalization (optional - can improve training)
        m_norm = sim.params.mass / 2;  % Normalize by max mass
        k_norm = sim.params.spring_constant / 50;  % Normalize by max spring constant
        c_norm = sim.params.damping_coefficient / 5;  % Normalize by max damping
        
        % Create sequences
        for i = sequence_length:length(time)-prediction_horizon
            % Input features: past states, inputs, and parameters
            X_seq = [];
            
            % Add past states and inputs
            for j = 0:sequence_length-1
                idx = i - sequence_length + j + 1;
                X_seq = [X_seq, states(idx,:), inputs(idx)];
            end
            
            % Add system parameters
            X_seq = [X_seq, m_norm, k_norm, c_norm];
            
            % Output: future states
            Y_seq = [];
            for j = 1:prediction_horizon
                Y_seq = [Y_seq, states(i+j,:)];
            end
            
            % Collect sequences
            all_X = [all_X; X_seq];
            all_Y = [all_Y; Y_seq];
        end
    end
    
    % Return prepared data
    processed_data.X = all_X;
    processed_data.Y = all_Y;
end

function [X_norm, Y_norm, norm_params] = normalize_data(X, Y)
    % Normalizes data to zero mean and unit standard deviation
    %
    % Args:
    %   X: Input features
    %   Y: Output targets
    %
    % Returns:
    %   X_norm: Normalized inputs
    %   Y_norm: Normalized outputs
    %   norm_params: Structure with mean and std for later denormalization
    
    % Calculate statistics
    X_mean = mean(X);
    X_std = std(X);
    Y_mean = mean(Y);
    Y_std = std(Y);
    
    % Handle zero std (constant features)
    X_std(X_std < 1e-8) = 1;
    Y_std(Y_std < 1e-8) = 1;
    
    % Normalize
    X_norm = (X - X_mean) ./ X_std;
    Y_norm = (Y - Y_mean) ./ Y_std;
    
    % Store normalization parameters
    norm_params.X_mean = X_mean;
    norm_params.X_std = X_std;
    norm_params.Y_mean = Y_mean;
    norm_params.Y_std = Y_std;
end

function analyze_dataset_statistics(data_struct)
    % Analyzes and reports statistics about the dataset
    %
    % Args:
    %   data_struct: Structure containing simulation results
    
    num_simulations = length(data_struct);
    
    % Extract parameters using loops to avoid comma-separated list issues
    masses = zeros(1, num_simulations);
    springs = zeros(1, num_simulations);
    dampings = zeros(1, num_simulations);
    
    for i = 1:num_simulations
        masses(i) = data_struct(i).params.mass;
        springs(i) = data_struct(i).params.spring_constant;
        dampings(i) = data_struct(i).params.damping_coefficient;
    end
    
    % Extract input types
    input_types = cell(num_simulations, 1);
    for i = 1:num_simulations
        input_types{i} = data_struct(i).input_type;
    end
    unique_input_types = unique(input_types);
    type_counts = zeros(size(unique_input_types));
    for i = 1:length(unique_input_types)
        type_counts(i) = sum(strcmp(input_types, unique_input_types{i}));
    end
    
    % Extract initial conditions
    initial_positions = zeros(num_simulations, 1);
    initial_velocities = zeros(num_simulations, 1);
    for i = 1:num_simulations
        if isfield(data_struct(i), 'initial_condition') && length(data_struct(i).initial_condition) >= 2
            initial_positions(i) = data_struct(i).initial_condition(1);
            initial_velocities(i) = data_struct(i).initial_condition(2);
        end
    end
    
    % Calculate damping ratios
    damping_ratios = zeros(num_simulations, 1);
    for i = 1:num_simulations
        m = data_struct(i).params.mass;
        k = data_struct(i).params.spring_constant;
        c = data_struct(i).params.damping_coefficient;
        damping_ratios(i) = c / (2 * sqrt(m * k));
    end
    
    % Print statistics
    fprintf('=== Dataset Statistics ===\n');
    fprintf('Total number of simulations: %d\n', num_simulations);
    fprintf('\n');
    
    fprintf('System Parameters:\n');
    fprintf('  Masses: %s\n', mat2str(unique(masses)));
    fprintf('  Spring Constants: %s\n', mat2str(unique(springs)));
    fprintf('  Damping Coefficients: %s\n', mat2str(unique(dampings)));
    fprintf('\n');
    
    fprintf('Initial Conditions:\n');
    fprintf('  Positions: %s\n', mat2str(unique(initial_positions)));
    fprintf('  Velocities: %s\n', mat2str(unique(initial_velocities)));
    fprintf('\n');
    
    fprintf('Input Types:\n');
    for i = 1:length(unique_input_types)
        fprintf('  %s: %d simulations\n', unique_input_types{i}, type_counts(i));
    end
    fprintf('\n');
    
    fprintf('Damping Ratio Distribution:\n');
    fprintf('  Min: %.2f\n', min(damping_ratios));
    fprintf('  Max: %.2f\n', max(damping_ratios));
    fprintf('  Underdamped systems (ζ < 1): %d\n', sum(damping_ratios < 1));
    fprintf('  Critically damped systems (ζ ≈ 1): %d\n', sum(abs(damping_ratios - 1) < 0.1));
    fprintf('  Overdamped systems (ζ > 1): %d\n', sum(damping_ratios > 1));
    fprintf('\n');
    
    % Create histogram of damping ratios
    figure;
    histogram(damping_ratios, 10);
    title('Distribution of Damping Ratios');
    xlabel('Damping Ratio (ζ)');
    ylabel('Number of Simulations');
    grid on;
    
    % Save the figure
    saveas(gcf, 'data/damping_ratio_distribution.png');
end

% =================== END DATA PROCESSOR FUNCTIONS ===================

% Run the main MSD model script to generate data
disp('Running mass-spring-damper model and generating dataset...');

%% 1. System Parameters
% Define the parameter sets we'll explore
masses = [1, 2];             % kg
spring_constants = [20, 50]; % N/m
damping_coeffs = [0.5, 5, 15];   % N·s/m, added 15 for overdamped examples

% Store all parameter combinations
param_combinations = [];
param_idx = 1;
for m = masses
    for k = spring_constants
        for c = damping_coeffs
            param_combinations(param_idx).mass = m;
            param_combinations(param_idx).spring_constant = k;
            param_combinations(param_idx).damping_coefficient = c;
            
            % Calculate damping ratio for reference
            param_combinations(param_idx).damping_ratio = c / (2 * sqrt(m * k));
            
            % Calculate natural frequency for reference
            param_combinations(param_idx).natural_freq = sqrt(k / m);
            
            param_idx = param_idx + 1;
        end
    end
end

%% 2. Initial Conditions
% Define the initial conditions sets
initial_positions = [0, 0.2];  % m
initial_velocities = [0, 1];   % m/s

% Store all initial condition combinations
ic_combinations = [];
ic_idx = 1;
for pos = initial_positions
    for vel = initial_velocities
        ic_combinations(ic_idx, :) = [pos, vel];
        ic_idx = ic_idx + 1;
    end
end

%% 3. Input Signal Functions
% Step Input
step_input = @(t, magnitude) magnitude * (t >= 1);  % Step at 1 second

% Sinusoidal Input
sine_input = @(t, amplitude, frequency) amplitude * sin(2 * pi * frequency * t);

% Random Input (pre-calculated)
rng(42); % Set seed for reproducibility
dt = 0.01; % 100 Hz sampling
t_end = 10; % 10 seconds simulation
t_vector = 0:dt:t_end;
random_signal = randn(size(t_vector)) * 1; % Standard deviation 1N
random_input = @(t) interp1(t_vector, random_signal, t, 'linear', 0);

%% 4. MSD System Dynamics Function
function dxdt = msd_dynamics(t, x, u_func, params)
    % Extract parameters
    m = params.mass;
    k = params.spring_constant;
    c = params.damping_coefficient;
    
    % Extract states
    position = x(1);
    velocity = x(2);
    
    % Calculate input force at time t
    force = u_func(t);
    
    % Calculate derivatives
    dxdt = zeros(2, 1);
    dxdt(1) = velocity;
    dxdt(2) = (force - k * position - c * velocity) / m;
end

%% 5. Simulation Function
function [t, x, u] = simulate_msd(params, x0, input_func, t_span)
    % Configure ODE solver
    options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
    
    % Create anonymous function for ODE solver
    odefun = @(t, x) msd_dynamics(t, x, input_func, params);
    
    % Solve ODE
    [t, x] = ode45(odefun, t_span, x0, options);
    
    % Calculate inputs for each time point
    u = zeros(length(t), 1);
    for i = 1:length(t)
        u(i) = input_func(t(i));
    end
end

%% 6. Data Generation Loop
rng(42); % Set seed for reproducibility
simulation_idx = 1;
all_simulations = struct([]);

% Define simulation time
t_span = [0 10]; % 10 seconds

% Input configurations 
step_magnitudes = [1, 5]; % N
sine_amplitudes = [2]; % N
sine_frequencies = [0.5, 2]; % Hz

fprintf('Generating datasets...\n');

% Loop through parameters
for param_idx = 1:length(param_combinations)
    params = param_combinations(param_idx);
    
    % Loop through initial conditions
    for ic_idx = 1:size(ic_combinations, 1)
        x0 = ic_combinations(ic_idx, :)';
        
        % Step inputs
        for magnitude = step_magnitudes
            input_func = @(t) step_input(t, magnitude);
            [t, x, u] = simulate_msd(params, x0, input_func, t_span);
            
            % Store simulation results
            all_simulations(simulation_idx).params = params;
            all_simulations(simulation_idx).initial_condition = x0;
            all_simulations(simulation_idx).input_type = 'step';
            all_simulations(simulation_idx).input_params = struct('magnitude', magnitude);
            all_simulations(simulation_idx).time = t;
            all_simulations(simulation_idx).states = x;
            all_simulations(simulation_idx).inputs = u;
            
            simulation_idx = simulation_idx + 1;
        end
        
        % Sinusoidal inputs
        for amplitude = sine_amplitudes
            for frequency = sine_frequencies
                input_func = @(t) sine_input(t, amplitude, frequency);
                [t, x, u] = simulate_msd(params, x0, input_func, t_span);
                
                % Store simulation results
                all_simulations(simulation_idx).params = params;
                all_simulations(simulation_idx).initial_condition = x0;
                all_simulations(simulation_idx).input_type = 'sine';
                all_simulations(simulation_idx).input_params = struct('amplitude', amplitude, 'frequency', frequency);
                all_simulations(simulation_idx).time = t;
                all_simulations(simulation_idx).states = x;
                all_simulations(simulation_idx).inputs = u;
                
                simulation_idx = simulation_idx + 1;
            end
        end
        
        % Random input
        input_func = random_input;
        [t, x, u] = simulate_msd(params, x0, input_func, t_span);
        
        % Store simulation results
        all_simulations(simulation_idx).params = params;
        all_simulations(simulation_idx).initial_condition = x0;
        all_simulations(simulation_idx).input_type = 'random';
        all_simulations(simulation_idx).input_params = struct('std_dev', 1);
        all_simulations(simulation_idx).time = t;
        all_simulations(simulation_idx).states = x;
        all_simulations(simulation_idx).inputs = u;
        
        simulation_idx = simulation_idx + 1;
    end
end

fprintf('Generated %d simulation scenarios\n', simulation_idx - 1);

%% 7. Split data into training, validation, and testing sets
num_simulations = length(all_simulations);
indices = randperm(num_simulations);

% Determine split indices (70% training, 15% validation, 15% testing)
train_end = floor(0.7 * num_simulations);
val_end = floor(0.85 * num_simulations);

train_indices = indices(1:train_end);
val_indices = indices(train_end+1:val_end);
test_indices = indices(val_end+1:end);

% Create the datasets
train_data = all_simulations(train_indices);
val_data = all_simulations(val_indices);
test_data = all_simulations(test_indices);

% Save the datasets
save('data/training/msd_train_data.mat', 'train_data');
save('data/validation/msd_val_data.mat', 'val_data');
save('data/testing/msd_test_data.mat', 'test_data');

fprintf('Data split and saved:\n');
fprintf('  Training: %d scenarios\n', length(train_data));
fprintf('  Validation: %d scenarios\n', length(val_data));
fprintf('  Testing: %d scenarios\n', length(test_data));

%% 8. Create and save visualization of example datasets
try
    % Create figure with subplots for different input types
    figure('Position', [100, 100, 1200, 800]);
    
    % Example step response
    step_idx = find(strcmp({all_simulations.input_type}, 'step'), 1);
    if ~isempty(step_idx)
        subplot(3, 2, 1);
        plot(all_simulations(step_idx).time, all_simulations(step_idx).states(:, 1), 'b-', 'LineWidth', 1.5);
        hold on;
        plot(all_simulations(step_idx).time, all_simulations(step_idx).inputs, 'r--', 'LineWidth', 1);
        hold off;
        title('Step Response - Position');
        xlabel('Time (s)');
        ylabel('Position (m) / Input (N)');
        legend({'Position', 'Input Force'});
        grid on;
        
        subplot(3, 2, 2);
        plot(all_simulations(step_idx).time, all_simulations(step_idx).states(:, 2), 'g-', 'LineWidth', 1.5);
        title('Step Response - Velocity');
        xlabel('Time (s)');
        ylabel('Velocity (m/s)');
        grid on;
    end
    
    % Example sine response
    sine_idx = find(strcmp({all_simulations.input_type}, 'sine'), 1);
    if ~isempty(sine_idx)
        subplot(3, 2, 3);
        plot(all_simulations(sine_idx).time, all_simulations(sine_idx).states(:, 1), 'b-', 'LineWidth', 1.5);
        hold on;
        plot(all_simulations(sine_idx).time, all_simulations(sine_idx).inputs, 'r--', 'LineWidth', 1);
        hold off;
        title('Sinusoidal Response - Position');
        xlabel('Time (s)');
        ylabel('Position (m) / Input (N)');
        legend({'Position', 'Input Force'});
        grid on;
        
        subplot(3, 2, 4);
        plot(all_simulations(sine_idx).time, all_simulations(sine_idx).states(:, 2), 'g-', 'LineWidth', 1.5);
        title('Sinusoidal Response - Velocity');
        xlabel('Time (s)');
        ylabel('Velocity (m/s)');
        grid on;
    end
    
    % Example random response
    random_idx = find(strcmp({all_simulations.input_type}, 'random'), 1);
    if ~isempty(random_idx)
        subplot(3, 2, 5);
        plot(all_simulations(random_idx).time, all_simulations(random_idx).states(:, 1), 'b-', 'LineWidth', 1.5);
        hold on;
        plot(all_simulations(random_idx).time, all_simulations(random_idx).inputs, 'r--', 'LineWidth', 1);
        hold off;
        title('Random Input Response - Position');
        xlabel('Time (s)');
        ylabel('Position (m) / Input (N)');
        legend({'Position', 'Input Force'});
        grid on;
        
        subplot(3, 2, 6);
        plot(all_simulations(random_idx).time, all_simulations(random_idx).states(:, 2), 'g-', 'LineWidth', 1.5);
        title('Random Input Response - Velocity');
        xlabel('Time (s)');
        ylabel('Velocity (m/s)');
        grid on;
    end
    
    % Save the figure
    saveas(gcf, 'data/example_responses.png');
    saveas(gcf, 'data/example_responses.fig');
    
    %% 9. Create Phase Portraits for Different Parameter Sets
    figure('Position', [100, 100, 1200, 600]);
    
    % Find examples with different damping ratios (underdamped vs overdamped)
    % Extract damping ratios using a loop to avoid comma-separated list issue
    damping_ratios = zeros(1, length(all_simulations));
    for i = 1:length(all_simulations)
        damping_ratios(i) = all_simulations(i).params.damping_ratio;
    end
    underdamped_idx = find(damping_ratios < 1, 1);
    overdamped_idx = find(damping_ratios > 1, 1);
    
    % Check if we found examples of both types
    if ~isempty(underdamped_idx) && ~isempty(overdamped_idx)
        % Plot underdamped phase portrait
        subplot(1, 2, 1);
        plot(all_simulations(underdamped_idx).states(:, 1), all_simulations(underdamped_idx).states(:, 2), 'b-', 'LineWidth', 1.5);
        title(sprintf('Underdamped (\\zeta = %.2f)', all_simulations(underdamped_idx).params.damping_ratio));
        xlabel('Position (m)');
        ylabel('Velocity (m/s)');
        grid on;
        
        % Plot overdamped phase portrait
        subplot(1, 2, 2);
        plot(all_simulations(overdamped_idx).states(:, 1), all_simulations(overdamped_idx).states(:, 2), 'r-', 'LineWidth', 1.5);
        title(sprintf('Overdamped (\\zeta = %.2f)', all_simulations(overdamped_idx).params.damping_ratio));
        xlabel('Position (m)');
        ylabel('Velocity (m/s)');
        grid on;
        
        % Save the figure
        saveas(gcf, 'data/phase_portraits.png');
        saveas(gcf, 'data/phase_portraits.fig');
        
        fprintf('Phase portrait visualizations saved to data folder\n');
    else
        fprintf('Warning: Could not find both underdamped and overdamped examples. Skipping phase portrait.\n');
        close(gcf);
    end
    
    fprintf('Visualization examples saved to data folder\n');
catch e
    fprintf('Warning: Error occurred during visualization: %s\n', e.message);
end

% Analyze dataset statistics
disp('Analyzing dataset statistics...');
% Combine datasets
combined_data = [train_data, val_data, test_data];
analyze_dataset_statistics(combined_data);

% Prepare sample data for neural network (for demonstration)
disp('Preparing sample data for neural network...');
sequence_length = 5;
prediction_horizon = 3;
train_processed = prepare_data_for_nn(train_data, sequence_length, prediction_horizon);

% Normalize the sample data
[X_norm, Y_norm, norm_params] = normalize_data(train_processed.X, train_processed.Y);

% Save normalization parameters for later use
save('data/norm_params.mat', 'norm_params');

% Display sample of the prepared data
fprintf('Sample of prepared data for neural network:\n');
fprintf('  Input features shape: [%d x %d]\n', size(X_norm));
fprintf('  Output targets shape: [%d x %d]\n', size(Y_norm));

% Summary stats in a figure
try
    figure('Position', [100, 100, 800, 600]);
    
    % Pie chart of input types
    subplot(2, 2, 1);
    input_types = {train_data.input_type};
    unique_types = unique(input_types);
    type_counts = zeros(size(unique_types));
    for i = 1:length(unique_types)
        type_counts(i) = sum(strcmp(input_types, unique_types{i}));
    end
    pie(type_counts);
    legend(unique_types, 'Location', 'Best');
    title('Distribution of Input Types');
    
    % Bar chart of parameter combinations
    subplot(2, 2, 2);
    params_train = [train_data.params];
    masses_train = zeros(1, length(train_data));
    springs_train = zeros(1, length(train_data));
    dampings_train = zeros(1, length(train_data));
    
    for i = 1:length(train_data)
        masses_train(i) = train_data(i).params.mass;
        springs_train(i) = train_data(i).params.spring_constant;
        dampings_train(i) = train_data(i).params.damping_coefficient;
    end
    
    unique_masses = unique(masses_train);
    unique_springs = unique(springs_train);
    unique_dampings = unique(dampings_train);
    
    num_params = length(unique_masses) + length(unique_springs) + length(unique_dampings);
    param_names = cell(num_params, 1);
    param_values = zeros(num_params, 1);
    
    for i = 1:length(unique_masses)
        param_names{i} = ['m=' num2str(unique_masses(i))];
        param_values(i) = sum(masses_train == unique_masses(i));
    end
    
    offset = length(unique_masses);
    for i = 1:length(unique_springs)
        param_names{i+offset} = ['k=' num2str(unique_springs(i))];
        param_values(i+offset) = sum(springs_train == unique_springs(i));
    end
    
    offset = offset + length(unique_springs);
    for i = 1:length(unique_dampings)
        param_names{i+offset} = ['c=' num2str(unique_dampings(i))];
        param_values(i+offset) = sum(dampings_train == unique_dampings(i));
    end
    
    bar(param_values);
    xticks(1:num_params);
    xticklabels(param_names);
    xtickangle(45);
    title('Parameter Distribution');
    ylabel('Count');
    
    % Data split visualization
    subplot(2, 2, 3);
    data_splits = [length(train_data), length(val_data), length(test_data)];
    bar(data_splits);
    xticks(1:3);
    xticklabels({'Training', 'Validation', 'Testing'});
    title('Dataset Split');
    ylabel('Number of Simulations');
    
    % Save the summary figure
    saveas(gcf, 'data/dataset_summary.png');
    saveas(gcf, 'data/dataset_summary.fig');
catch e
    fprintf('Warning: Error occurred during summary visualization: %s\n', e.message);
end

disp(' ');
disp('Dataset generation complete!');
disp('Data is ready for neural network training and MPC development.');
disp('=======================================');