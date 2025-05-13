# Motor Parameters Estimation
## Model
A simplified Brushless DC (BLDC) motor model is used to estimate the motor parameters. The input to the model is the applied voltage ($U$), and the output is the motor acceleration ($\ddot{\theta}$). The model can be divided into two parts: the electrical model and the mechanical model. The electrical model describes the relationship between the input voltage and the motor current ($I$), while the mechanical model describes the relationship between the motor current and the motor acceleration.

The electrical model is given by:
```math
\begin{equation}
U = R \cdot I + L \cdot \frac{dI}{dt} + K_b \cdot \dot{\theta}
\end{equation}
```
where $U$ is the input voltage, $R$ is the resistance, $L$ is the inductance, $K_b$ is the back EMF constant, and $\dot{\theta}$ is the motor speed.

The mechanical model is given by:
```math
\begin{equation}
J \cdot \ddot{\theta} = K_t \cdot I - d \cdot \dot{\theta} - sgn(\dot{\theta})\cdot f
\end{equation}  
```
where $J$ is the moment of inertia, $K_t$ is the torque constant, $d$ is the damping coefficient, and $f$ is the dry friction coefficient. To be noted that the dry friction modeled in MuJoCo is a constraint which is much more complicated. But for parameter estimation, we simplify the model to a constant dry friction loss. Moreover, the moment of inertia consists of two parts: the inertia of the motor rotor (which is `armature` in MuJoCo xml) and the inertia of the wheel(which the `diag` in MuJoCo xml).

## Collected Data
The data is collected from real-world experiments using a BLDC motor with sampling rate of $200 Hz$. The data consists of the following variables:
- `time`: The time ($s$).
- `voltage`: The rated voltage ($V$).
- `pwm`: The PWM signal applied to the motor ($\%$).
- `position`: The motor position ($rad$).

Need to specify that the applied voltage is the PWM signal multiplied by the rated voltage. Also, the position is collected instead of the speed since the motor is not equipped with an encoder. The position of the motor is measured using hall sensors with equivalent resolution of 0.66 degrees. The resolution of hall sensors can be represented as:
```math
\begin{equation}
\text{resolution} = \frac{2\pi}{3N\cdot \text{gear ratio}}
\end{equation}
```
where $N$ is the number of poles of the motor which is $8$ and gear ratio is $22.67$. As a result, the calculated speed is not smooth especially at low speed. So we use the position data as the output to estimate the motor parameters instead of the speed data.

## Model Parameter Estimation
### Parameters to be estimated
- `R`: Resistance ($\Omega$).
- `L`: Inductance ($H$).
- `K_b`: Back EMF constant ($V/rpm$).
- `K_t`: Torque constant ($N\cdot m/A$).
- `J`: Moment of inertia ($kg\cdot m^2$).
- `d`: Damping coefficient ($N\cdot m\cdot s/rad$).
- `f`: Dry friction coefficient ($N\cdot m$).

### Initial guess and bounds
- `R`: Initial guess: 0.0725 Bounds: [0.0695, 0.0755]
- `L`: Initial guess: 0.00067, Bounds: [0.00065, 0.00069]
- `K_b`: Initial guess: 22, Bounds: [20, 24]
- `K_t`: Initial guess: 0.72, Bounds: [0.70, 0.74]
- `J`: Initial guess: 0.03, Bounds: [0.01, 0.05]
- `d`: Initial guess: 0.1, Bounds: [0.01, 1.0]
- `f`: Initial guess: 0.05, Bounds: [0.01, 1.0]

Electrical parameters are provided by the manufacturer and the initial guess and bounds are set accordingly. The inertial parameters are set based on the geometry and mass of the motor and wheel. The damping coefficient and dry friction coefficient are set empirically based on the motor performance. The bounds are set to ensure that the estimated parameters are within a reasonable range.

### Optimization
All these parameters ($\mu$) can be estimated by minimizing the output differences between the model and the real data given the same input. The nonlinear least square optimization problem can be formulated as:
```math
\begin{align}
\mu^* &= arg\min_{\mu} \sum_{i=1}^{N} \left( \theta_{model}(t_i, \mu) - \theta_{real}(t_i) \right)^2\\
s.t. &\quad\mu_{initial}=\mu_0\\
&\quad \mu_{lower} \leq \mu \leq \mu_{upper}
\end{align}
```
where $\theta_{model}$ is the output of the model, $\theta_{real}$ is the output of the real data, $t_i$ is the time, and $N$ is the number of data points. 

The optimization method used is the Levenberg-Marquardt algorithm. Alternative methods such as the trust region reflective algorithm and gradient descent algorithm can also be selected.

## Extended Kalman Filter-based Position and Velocity Estimation (Optional)
With identified motor model, we can use the Extended Kalman Filter (EKF) to estimate the motor position to improve the smoothness of the calculated velocity. The accuracy of the estimated velocity is critical for the estimation of human velocity and speed-adaptive control of DRBA.

Let the state vector be:
```math
\begin{equation}
x = \begin{bmatrix}
\theta\\
\omega\\
I
\end{bmatrix}
\end{equation}
```
where $\theta$ is the motor position, $\omega$ is the motor speed and $I$ is the motor current. 

The state transition model can be formulated as:

```math
\begin{equation}
\begin{bmatrix}
\dot{\theta}\\
\dot{\omega}\\
\dot{I}
\end{bmatrix} = 
\begin{bmatrix}
\dot{\theta} = \omega\\
\frac{1}{J} \left( K_t \cdot I  - d \cdot \omega - sgn(\omega) \cdot f \right)\\
\frac{1}{L} \left( U - R \cdot I - K_b \cdot \omega \right)
\end{bmatrix}
\end{equation}
```

The discrete-time state transition model can be formulated as:
```math
\begin{equation}
\begin{bmatrix}
\theta_{k}\\
\omega_{k}\\
I_{k}
\end{bmatrix} =
f(x_{k-1}, u_{k-1}) =
\begin{bmatrix}
\theta_{k-1} + \omega_{k-1} \cdot dt\\
\omega_{k-1} + \frac{1}{J} \left( K_t \cdot I_{k-1}  - d \cdot \omega_{k-1} - sgn(\omega_{k-1}) \cdot f \right) \cdot dt\\
I_{k} = I_{k-1} + \frac{1}{L} \left( U - R \cdot I_{k-1} - K_b \cdot \omega_{k-1} \right) \cdot dt
\end{bmatrix}
\end{equation}
```
where $dt$ is the time step.

Then the Jacobian matrix $F$ of the state transition model can be determined. Since $sgn(\omega)$ is not differentiable, we typically approximate it with a smooth function for EKF linearization:
```math
\begin{equation}
sng(\omega) \approx tanh(\frac{\omega}{\epsilon})
\end{equation}
```
where $\epsilon$ is a very small constant. Then:
```math
\begin{equation}
\frac{d}{dt}sgn(\omega) \approx \frac{1}{\epsilon}(1 - tanh^2(\frac{\omega}{\epsilon}))
\end{equation}
```
The Jacobian matrix of the state transition model can be formulated as:
```math
\begin{equation}
F_k = \begin{bmatrix}
1 & dt & 0\\
0 & 1 - \frac{dt}{J} \cdot (d+\frac{f}{\epsilon}(1 - tanh^2(\frac{\omega_k}{\epsilon}))) & \frac{dt}{J} \cdot K_t\\
0 & -\frac{dt}{L} \cdot K_b & 1 - \frac{dt}{L} \cdot R
\end{bmatrix}
\end{equation}
```

The measurement model can be formulated as:
```math
\begin{equation}
z_k =  h(\theta_k)=\theta_k
\end{equation}
```
where $z_k$ is the measurement vector. The Jacobian matrix $H$ of the measurement model can be formulated as:
```math
\begin{equation}
H_k = \begin{bmatrix}
1 & 0 & 0
\end{bmatrix}
\end{equation}
```

The predicted state and covariance estimation can be represented as:
```math
\begin{align}
\hat{x}_{k|k-1} = f(\hat{x}_{k-1|k-1}, u_{k-1})\\
P_{k|k-1} = F_k \cdot P_{k-1|k-1} \cdot F_k^T + Q
\end{align}
```
where $\hat{x}_{k|k-1}$ is the predicted state, $P_{k|k-1}$ is the predicted covariance, and $Q$ is the process noise covariance.

The updated state and covariance estimation can be represented as:
```math
\begin{align}
\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k \cdot (z_k - h(\hat{x}_{k|k-1}))\\
P_{k|k} = (I - K_k \cdot H_k) \cdot P_{k|k-1}
\end{align}
```
where $K_k$ is the near-optimal Kalman gain, $I$ is the identity matrix, and $P_{k|k}$ is the updated covariance.
The Kalman gain can be calculated as:
```math
\begin{equation}
K_k = P_{k|k-1} \cdot H_k^T \cdot (H_k \cdot P_{k|k-1} \cdot H_k^T + R)^{-1}
\end{equation}
```
where $R$ is the measurement noise covariance.


## PID Speed Control and Parameter Estimation
MuJoCo only simulates the torque of the motor instead of the voltage so we cannot replicate the speed control parameters of the real motor directly. So we need to design a PID-based speed controller in MuJoCo and search for the optimal PID parameters which enables the controller to capture the performances of the real controller.

One option is to use the `motor` actuator in MuJoCo which applies the torque to the motor. With given reference speed ($\omega_{ref}$) and actual speed ($\omega_{real}$), the PID controller can be formulated as:
```math
\begin{align}
\omega_{error} &= \omega_{ref} - \omega_{real}\\
\omega_{control} &= K_p \cdot \omega_{error} + K_i \cdot \int_0^t \omega_{error} dt + K_d \cdot \frac{d}{dt} \omega_{error}
\end{align}
```
where $K_p$, $K_i$, and $K_d$ are the proportional, integral, and derivative gains, respectively.

Another option is to use the `intvelocity` actuator in MuJoCo with a embedded speed controller. Instead of correcting the speed error, the `intvelocity` actuator corrects the accumulated position error by coupling an integrator with a position-feedback controller. The prinicple is similar to the real DRBA motor speed controller("时间-位置闭环控制"). The tunable parameters are `kp` and `kv` which are the position feedback gain and actuator damping gain respectively.

Current parameter search is based on the `intvelocity` actuator.