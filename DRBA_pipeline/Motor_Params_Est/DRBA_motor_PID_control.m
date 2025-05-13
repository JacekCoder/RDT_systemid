%% Identified motor params
J = 0.05;
L = 0.00069;
R = 0.0755;
d = 0.35757;
f = 0.53378;
Kb = 20.91;
Kt = 0.7;
dt = 0.005;
%% set_3_has_1 for estimation
sysid_data = importdata("set3_has_1.csv").data;
len = size(sysid_data,1);
input_time = (0:0.005:(len-1)*0.005)';
output_speed = sysid_data(:,2);
output_pos = sysid_data(:,3);
input_current = sysid_data(:,4);
input_pwm = sysid_data(:,5)*10;
input_voltage = sysid_data(:,6);

eq_voltage = input_pwm .* input_voltage/100;

v = timeseries(eq_voltage,input_time);
simout = sim('BLDC_full_sysid_sim.slx','StartTime','0','StopTime','50');

plot(simout.pos);
hold on
plot(output_pos);