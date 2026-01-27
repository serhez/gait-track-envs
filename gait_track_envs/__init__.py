from . import ant
from . import walker2d
from . import half_cheetah
from . import half_cheetah_36
from . import half_cheetah_36_motor
from . import half_cheetah_2seg
from . import hopper
from . import humanoid
from . import unitree
from . import template_renderer


def register_env(env_name):
    if "Ant" in env_name:
        ant.register_ant(env_name)
    elif "Walker2d" in env_name:
        walker2d.register_walker2d()
    elif "2segHalfCheetah" in env_name:
        half_cheetah_2seg.register_half_cheetah_2seg(env_name)
    elif "HalfCheetah36Motor" in env_name:
    	half_cheetah_36_motor.register_half_cheetah_36_motor(env_name)
    elif "HalfCheetah36" in env_name:    
        half_cheetah_36.register_half_cheetah(env_name)
    elif "HalfCheetah" in env_name:
        half_cheetah.register_half_cheetah(env_name)
    elif "Hopper" in env_name:
        hopper.register_hopper()
    elif "Humanoid" in env_name:
        humanoid.register_humanoid(env_name)
    elif "Unitree" in env_name:
        unitree.register_unitree(env_name)    
    else:
        raise ValueError("Unknown env name: {}".format(env_name))
