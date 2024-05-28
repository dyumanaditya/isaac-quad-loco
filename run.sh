#!/bin/bash

#git clone https://github.com/dyumanaditya/quadruped-locomotion
git clone https://github.com/dyumanaditya/isaac-ppo

mv isaac-ppo/isaac_ppo quadruped-locomotion/

/isaac-sim/python.sh main.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 1024 --video --headless --offscreen_render