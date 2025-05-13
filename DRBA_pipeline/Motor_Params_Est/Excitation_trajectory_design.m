% Design the excitation trajectory
t_max = 50.0;
t_now = 0.0;

% Define the parameters
num_freqs = 10;
freqs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5];
amplitude = 90.0;
fs = 200;
timestep = 1/fs;

current_freq_index = 1;
sample_index = 0;
total_samples = 0;

freq = freqs(current_freq_index);
samples_per_cycle = fs/freq;

% Initiate trajectory matrix
traj = zeros(1,t_max*fs);

while t_now < t_max
    t_now = total_samples * timestep;
    t_cycle = sample_index * timestep;
    signal = amplitude * sin(2.0*pi*freq*t_cycle);

    sample_index = sample_index+1;
    total_samples = total_samples+1;

    traj(total_samples) = signal;
    
    if sample_index >= samples_per_cycle
        sample_index = 0;
        current_freq_index = current_freq_index+1;
        if current_freq_index >= num_freqs
            current_freq_index = 1;
        end
        freq = freqs(current_freq_index);
        samples_per_cycle = fs/freq;
    end
end

plot(traj);

