import gym
import numpy as np
from gym import utils
import time

from .jinja_mujoco_env import MujocoEnv


class UnitreeEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, parametric=True, init_task=None):
        self.original_lengths = np.array([ 0.2, 0.2, 0.2, 0.2, 0.05])
        self.current_lengths = np.array(self.original_lengths)
        self.model_args = {"size": list(self.original_lengths)}

        self.markers = ["thigh", "leg", "foottip"]
        self.legs = ["fr_", "fl_", "rr_", "rl_"]
        self.origin = "torso"

        MujocoEnv.__init__(self, 'unitree.xml', 10)
        utils.EzPickle.__init__(self)

        #self.min_task = np.ones_like(self.original_lengths)*0.035
        self.min_task = self.original_lengths*0.2
        self.max_task = self.original_lengths*2.0
        self.min_task[4] = 0.01
        self.max_task[4] = 0.8
        

        self.parametric = parametric

        if init_task:
            task = self.get_test_tasks()[init_task]
            self.set_task(*task)

    def get_test_tasks(self):
        return {"normal": np.array( [*self.original_lengths] ),
                "short": np.array( [*(self.original_lengths*0.5)] ),
                "long": np.array( [*(self.original_lengths*2)] )}

    def set_random_task(self):
        self.set_task(*self.sample_task())

    def sample_task(self):
        task = np.random.uniform(self.min_task, self.max_task, self.min_task.shape)
        task[1] = np.clip(task[0] + np.random.uniform(-0.05, 0.05), self.min_task[1], self.max_task[1])
        task[3] = np.clip(task[2] + np.random.uniform(-0.05, 0.05), self.min_task[3], self.max_task[3])
        return task

    def sample_tasks(self, num_tasks=1):
        return np.stack([self.sample_task() for _ in range(num_tasks)])

    def get_task(self):
        return np.copy(self.current_lengths)

    @property
    def limb_segment_lengths(self):
        return np.concatenate((np.ones((4,1)) * 0.213, self.current_lengths[:4].reshape(4, 1)), axis=-1)
    
    @property
    def morpho_params(self):
        assert self.current_lengths.flatten().shape == (4, )
        return self.current_lengths.flatten()

    def set_task(self, *task):
        if len(task) == len(self.current_lengths):
            self.current_lengths[:] = task
        else:
            raise ValueError("Incorrect task shape")
        self.model_args = {"size": list(self.current_lengths)}
        self.build_model()
        
    def reset(self):
        #time.sleep(5)
        self.sim.reset()
        self.reset_model()

        self.initial_pos = self.sim.data.get_site_xpos(f"{self.origin}_track").copy()
        self.init_height = 0
        if self.fall_on_reset:
            ob = self.simulate_to_stop(max_steps=3000, gravity=-1, vel_threshold=-100.0, freeze_qpos_idx=[0] + list(range(2,2+12+1)))
        else:
            ob = self._get_obs()
        
        qpos_now = np.array(self.sim.data.qpos.flat)
        qvel_now = np.array(self.sim.data.qvel.flat)
        qpos = qpos_now
        qpos[1] = qpos[1] + 0.02
        qvel = qvel_now * 0.0 # + self.np_random.standard_normal(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.init_height = 0
        ob = self._get_obs()
        
        self.init_height = ob[0]
        #ob[0] = ob[0] - self.init_height
        return ob, {}
    
    def simulate_to_stop(self, max_steps=1000, vel_threshold=1e-2, gravity=None,
            freeze_qpos_idx=[], render=False):
        frozen_qpos = [self.sim.data.qpos[i] for i in freeze_qpos_idx]
        frozen_qvel = [self.sim.data.qvel[i] for i in freeze_qpos_idx]
        if gravity:
            assert gravity < 0.  # :D
            org_grav = self.sim.model.opt.gravity[2]
            self.sim.model.opt.gravity[2] = gravity

        for sstep in range(max_steps):
            self.sim.data.ctrl[:] = 0.
            self.sim.step()
            for fi, fp, fv in zip(freeze_qpos_idx, frozen_qpos, frozen_qvel):
                self.sim.data.qpos[fi] = fp
                self.sim.data.qvel[fi] = fv
            #print(self.sim.data.qpos[1])
            if self.sim.data.qvel[1] < 0:
              self.sim.data.qvel[1] = np.amax([-1.0, self.sim.data.qvel[1]])
            else:
              self.sim.data.qvel[1] = np.amin([1.0, self.sim.data.qvel[1]])

            if render:
                self.render()

        if "ant" in self.model_path.lower() or "humanoid.xml" in self.model_path.lower():
            self.init_height = self.sim.data.qpos[2]
        else:
            self.init_height = self.sim.data.qpos[1]
        self.sim.model.opt.gravity[2] = org_grav

        #self.sim.step()
        #time.sleep(5)
        return self._get_obs()
        

    def step(self, action):
        #xposbefore = self.sim.data.qpos[0]
        #self.do_simulation(action, self.frame_skip)
        #xposafter = self.sim.data.qpos[0]
        #ob = self._get_obs()
        #reward_ctrl = - 0.1 * np.square(action).sum()
        #reward_run = 1.25 * (xposafter - xposbefore)/self.dt
        #reward = np.amax([(reward_ctrl + reward_run), 0.0])
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        posafter, height, ang, angx = self.sim.data.qpos[0:4]
        alive_bonus = 1.0
        reward_run = 3.0 * ((posafter - posbefore) / self.dt)
        reward_run = np.amax([(reward_run), 0.0])
        upright = -(( 1+ np.abs(ang))**2 + np.abs(angx) - 1.0)* 0.1 * 0.5
        control_cost = -np.linalg.norm(action)*0.001
        reward = (.5 + float(height > (self.init_height-0.2)) + float(height > (self.init_height-0.1))*0.25+ float(height > (self.init_height)*0.25)) * (reward_run + .1) + upright + control_cost + (float(np.abs(ang) > 1.0) * -0.5)
        terminated = np.abs(ang) > 1.8 or height <= (self.init_height-0.25)
        truncated = False #np.abs(ang) > 2.8
        if height > 2.0:
          print("WARNING WARNING: CRAZY FLYING ROBOT DETECTED!!! WARNING WARNING!")
          #exit(0)
          terminated = True
          reward = -30.0

        # Get pos/vel of the feet
        track_info = self.get_track_dict()


        info = {"reward_run": reward_run, 'reward_sum':reward,
                **track_info}
        return ob, reward, terminated, truncated, info

    def _get_obs(self):
        qpos = self.sim.data.qpos.flat[1:]
        qvel = self.sim.data.qvel.flat
        #qpos[0] -= self.init_height
        #qpos[0] = qpos[0] #/ self.init_height

        #return np.concatenate([
        #    qpos,
        #    qvel,
        #])
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.init_qpos = np.array(
         [0., 0.5, 0,0, 0, 0.2, -0.5, 0, 0.2, -0.5, 0, 0.2, -0.5, 0, 0.2, -0.5]
         )
        qpos = self.init_qpos 
        qpos[4:] = qpos[4:] + self.np_random.uniform(low=-.1, high=.1, size=12)
        qvel = self.init_qvel * 0.0
        qvel[1] = -0.01 #+ self.np_random.standard_normal() * .1
        qpos[1] = np.amax(self.current_lengths) + 0.2
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_sim_state(self, state):
        return self.sim.set_state(state)

    def get_sim_state(self):
        return self.sim.get_state()


def register_unitree(env_name):
    if env_name == "GaitTrackUnitreeEnv-v0":
        kwargs = {"parametric": True}
    else:
        raise ValueError("Unknown env name")

    gym.envs.register(
            id=env_name,
            entry_point="%s:UnitreeEnv" % __name__,
            max_episode_steps=600,
            kwargs=kwargs,
    )
