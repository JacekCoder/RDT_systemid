# Mechanical Arm Parameter Estimation

## Model

A simplified mechanical arm (DRBA model) is used to estimate damping and friction parameters using the MuJoCo physics engine. The input to the model is the initial joint configuration, and the output is the joint trajectory. The model dynamics include friction and damping at the joints.

The mechanical model implicitly used within MuJoCo is expressed as:

$$
\tau = -d \cdot \dot{\theta} - \text{sgn}(\dot{\theta}) \cdot f
$$

where:

* \$\tau\$ is the joint torque.
* \$d\$ is the damping coefficient.
* \$f\$ is the friction coefficient.
* \$\dot{\theta}\$ is the joint angular velocity.

## Collected Data

Simulation data is collected by running the model from three different initial joint configurations:

* **Set 1**: `[0.0, 0.0, 0.0, 0.0]`
* **Set 2**: `[0.2, 0.2, -0.2, -0.2]`
* **Set 3**: `[-0.2, -0.2, 0.2, 0.2]`

Each simulation runs for 500 steps, recording angles of six joints:

* Left distal joint
* Left forearm joint
* Left interface joint
* Right distal joint
* Right forearm joint
* Right interface joint

## Model Parameter Estimation

### Parameters to be estimated

Due to symmetry, parameters for the left and right joints are shared. Hence, the parameters to be estimated are:

* \$d\_1, f\_1\$: Distal joints
* \$d\_2, f\_2\$: Forearm joints
* \$d\_3, f\_3\$: Interface joints

### Initial guess and bounds

* All initial guesses are set to 0.1.
* Parameter bounds: `[0, 1]` for all parameters.

The bounds for damping and friction coefficients are empirically set to maintain physically meaningful values.

### Optimization

Parameters (\$\mu\$) are estimated by minimizing the mean squared error (MSE) between simulated and real trajectories:

$$
\begin{aligned}
\mu^* &= \arg\min_{\mu} \sum_{i=1}^{N} \left( \theta_{simulated}(t_i, \mu) - \theta_{real}(t_i) \right)^2 \\
\text{s.t.} &\quad \mu_{initial}=\mu_0 \\
&\quad \mu_{lower} \leq \mu \leq \mu_{upper}
\end{aligned}
$$

where \$\theta\_{simulated}\$ and \$\theta\_{real}\$ are joint angles from simulation and real data, respectively. The optimization algorithm used is L-BFGS-B.

## Visualization

Optimized parameters are validated by comparing simulated trajectories against collected real trajectories, visualized through plots to assess fitting accuracy.
