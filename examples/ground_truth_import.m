% This is how ground truth data in a .mat file exported from Stone Soup 
% is read into MATLAB to generate sensor readings with suitable timestamps

clc; clear; close
load('ground_truth.mat'); %load the data file
% timestamps: array of timestamps saved as floats
% truth: cell array of vertically stacked target states
% ideal_measurements: cell array of vertically stacked ideal measurements

n_steps = length(timestamps);

for i = 1:n_steps

    timestamp_i = timestamps(i); %get current timestamp 
    target_states_i = truth{i}; %get current target states
    ideal_measurements_i = ideal_measurements{i}; %get current target states 

    if ~isempty(truth{i})
        % if targets are present, individual state vectors are accessed via

        n_targets = size(target_states_i,2); %get n of targets/measurements

        for t=1:n_targets
            % [x, x_dot, y, y_dot, z, z_dot]
            target_state_t = target_states_i(t,:);

            % [elevation(=theta), bearing(=phi), range, rangerate]
            ideal_measurement_t = ideal_measurements_i(t,:);
        end
    end
end