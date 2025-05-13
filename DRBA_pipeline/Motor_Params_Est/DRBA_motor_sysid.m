%% Inital value for motor params
Kt = 0.72;
Kb = 22;
d = 0.1;
f = 0.05;
J = 0.03;
L = 0.00067;
R = 0.0725;

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

%% set_4_has_1 for validation
sysid_data2 = importdata("set4_has_1.csv").data;
len2 = size(sysid_data2,1);
input_time2 = (0:0.005:(len2-1)*0.005)';
output_speed2 = sysid_data2(:,2);
output_pos2 = sysid_data2(:,3);
input_current2 = sysid_data2(:,4);
input_pwm2 = sysid_data2(:,5);
input_voltage2 = sysid_data2(:,6);

eq_voltage2 = input_pwm2 .* input_voltage2/100;