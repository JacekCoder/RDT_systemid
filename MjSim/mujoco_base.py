import mujoco as mj
import mujoco.viewer as mjv
from mujoco.glfw import glfw
class MuJoCoBase():
    def __init__(self, xml_path,
                 sim_timestep = None):

        # MuJoCo data structures
        self._model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
        self._data = mj.MjData(self._model)                # MuJoCo data
        self._sim_timestep = sim_timestep
        
        # MuJoco simulation settings
        
        # 1. Simulation timestep
        if self._sim_timestep is not None:
            self._model.opt.timestep = sim_timestep
            
        # 2. Simulation integration method
        self._model.opt.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
        
        # 3. Interative GUI opt
        self._GUI_opt = mj.MjvOption()
        self._GUI_opt.flags[mj.mjtVisFlag.mjVIS_CONVEXHULL] = False # Show convex hulls of geoms
        self._GUI_opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = False # Show contact points in collision detections
        self._GUI_opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = False # Show contact forces in collision detections
        self._sim_pause = False
        
        # Init MuJoCo viewer (GUI)
        self._viewer = mjv.launch_passive(self._model, self._data,key_callback=self.key_callback)
        self._viewer.opt.flags = self._GUI_opt.flags
    
    def step(self): 
        """
        Advance the simulation by one step
        """
        if not self._sim_pause:
            mj.mj_step(self._model, self._data)
            self._viewer.sync()
    def forward(self): 
        """
        Forward dynamics without integration in time
        """
        mj.mj_forward(self._model, self._data)
    
    def inverse(self): 
        """
        Inverse dynamics: qacc must be set before calling
        """
        mj.mj_inverse(self._model, self._data)

    def reset(self, *args, **kwargs): 
        """
        Reset the simulation
        """
        mj.mj_resetData(self._model, self._data)
        
    def key_callback(self, shortcuts): 
        """
        Keyboard shortcuts for interactive GUI
        
        Args:
            shortcuts (unicode): Keyboard shortcuts
        """
        if chr(shortcuts) == ' ':
            self._sim_pause = not self._sim_pause
        if chr(shortcuts) == 'H':
            self._GUI_opt.flags[mj.mjtVisFlag.mjVIS_CONVEXHULL] = not self._GUI_opt.flags[mj.mjtVisFlag.mjVIS_CONVEXHULL]
            self._viewer.opt.flags = self._GUI_opt.flags
        if chr(shortcuts) == 'C':
            self._GUI_opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = not self._GUI_opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT]
            self._viewer.opt.flags = self._GUI_opt.flags
        if chr(shortcuts) == 'F':
            self._GUI_opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = not self._GUI_opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE]
            self._viewer.opt.flags = self._GUI_opt.flags

    def controller(self, *args, **kwargs):
        """
        Controller for the robot
        """
        pass

    def close(self):
        """ 
        Close the GUI
        """
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        pass
    
    def animate(self): 
        """
        Motion replay (kinemaics only)
        """
        pass
    
    def rollout(self): 
        """
        Generate a rollout
        """
        pass
    
    def simulate(self,*args,**kwargs): 
        """
        Main simulation loop
        """
        pass
    
    @property
    def model(self):
        """
        Return the MuJoCo model
        """
        return self._model
    @property
    def data(self):
        """
        Return the MuJoCo data
        """
        return self._data
    
    # data-related features
    @property
    def qpos(self):
        """
        Return the joint positions
        """
        return self._data.qpos
    
    
    @property
    def cinert(self):
        """
        Return the inertial properties of a body in the c-frame
        """
        return self._data.cinert
    
    @property
    def qM(self):
        """
        Return the joint space mass matrix
        """
        return self._data.qM
    
    # model-related features
    # size
    @property
    def nq(self):
        """
        Return the number of generalized coordinates = dim(qpos)
        """
        return self._model.nq
    
    @property
    def nv(self):
        """
        Return the number of degrees of freedom = dim(qvel)
        """
        return self._model.nv
    
    @property
    def nM(self):
        """
        Return the number of non-zeros in sparse inertial matrix
        """
        return self._model.nM
    
    # bodies
    @property
    def body_ipos(self):
        """
        Return the local position of center of mass
        """
        return self._model.body_ipos
    @property
    def body_iquat(self):
        """
        Return the local orientation of inertial ellipsoid
        """
        return self._model.body_iquat
    @property
    def body_mass(self):
        """
        Return the mass of the body
        """
        return self._model.body_mass
    @property
    def body_subtreemass(self):
        """
        Return the mass of subtree starting at this body
        """
        return self._model.body_subtreemass
    @property
    def body_inertia(self):
        """
        Return the diagonal inertia in ipos/iquat frame
        """
        return self._model.body_inertia

    # dofs
    
    @property
    def dof_M0(self):
        """
        Return the diag. inertial in qpos0
        """
        return self._model.dof_M0
    
    @property
    def dof_Madr(self):
        """
        Return the dof address in M-diagonal (diagonal address in the sparse inertial matrix )
        """
        return self._model.dof_Madr
    
    @property
    def dof_simplenum(self):
        """
        Return the number of consecutive simple dofs
        """
        return self._model.dof_simplenum
    
    