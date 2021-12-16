% inputs: pulse - 1-D array of pulse waveform
%             z - matrix of 3-D Cartesian coordinates of each sensor position
%            fs - sampling frequency in Hz
%             T - length of signal in seconds
%
% outputs: signal_out - array containing the simulated signals for each of
%                       the hydrophones
%               truth - DoAs for each of the targets
%                   M - number of targets

function [signal_out,truth,M] = sonar_sim_active(pulse,z,fs,T)

Lambda = 1; % expected number of targets (for Poisson sampling)

%M = random('Poisson',Lambda);
M = 1;

%range = random('Uniform',10,100) / fs;
range = 2000; % range in samples - divide by fs and multiply by wave speed
              % to get range as distance

N = T*fs+1;

signal_out = zeros(9,N);

truth = zeros(2,M);

for n=1:M
    % simulate angles for one target (away from borders)
    %theta = 0.1+(pi/2-0.2)*rand(1);
    theta = 0.8;
    phi = 0.2;
    signal_out = signal_out + sonar_sim2d(pulse,z,fs,T,theta,phi,range);
    truth(1,n) = theta;
    truth(2,n) = phi;
end

end

% Simulated sonar data for an evenly-spaced linear array of 10 hydrophones
%
% Input: pulse - transmitted sonar pulse
%        d     - spacing between hydrophones (m)
%        theta - elevation angle (rad)
%
% Output: signal_out - matrix containing the signals measured at each
%                      hydrophone
function [signal_out] = sonar_sim2d(pulse,z,fs,T,theta,phi,range)

c = 1481; % speed of sound (m/s)

N = T*fs+1;

L_pulse = length(pulse);

% calculate phase offsets based on positions of hydrophones
a = [cos(theta) * sin(phi),...
     sin(theta) * sin(phi),...
     cos(phi)];

% calculate time delays
time_delays = round(range + fs*a * z' / c);

% simulated return pulses
signal_out=zeros(length(z), N);

for n=1:length(z)
    if time_delays(n) > 0
        signal_out(n,time_delays(n):L_pulse+time_delays(n)-1) = pulse;
    else
        error('something went wrong');
    end
end

end