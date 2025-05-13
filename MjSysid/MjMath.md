# MuJoCo Dynamics

## Constraints
### Dry friction (frictionloss)
Dry friction ($f_s$), also called static friction, is modeled as load-indepedant force in MuJoCo bounded with a box constraint $\lvert f_s \rvert \leq \mu_s$, where $\mu_s$ is a positive value defined by `mjModel.dof_frictionloss` for each joint in MuJoCo.

Finally, the actual static friction can be presented as 