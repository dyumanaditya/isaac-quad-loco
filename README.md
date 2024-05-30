# Quadruped Locomotion
Quadruped Locomotion learning through Reinforcement Learning and Model Predictive Control

## Installation
[Isaac-PPO](https://github.com/dyumanaditya/isaac-ppo) and [Isaac-Sim Orbit](https://isaac-orbit.github.io/) are needed for this repository.
Please follow the installation instructions on both repositories. When installing Orbit, make sure to create a virtual environment
called `orbit` using the instructions provided.

## Tasks
Below are a list of tasks that have been tested. They come from the default orbit tasks [listed here](https://isaac-orbit.github.io/orbit/source/features/environments.html#locomotion).
1. `Isaac-Velocity-Flat-Unitree-A1-v0`
2. `Isaac-Velocity-Rough-Unitree-A1-v0`

## Usage
### Modes
There are several modes available to run the package listed below:
1. `rl` - Runs pure RL PPO on the specified task
2. `mpc` - Runs pure MPC control on the specified task
3. `mpc-rl` - Runs RL + MPC on the specified task

**Note:** For `rl` and `rl-mpc` modes, the default run trains the models. To play the learnt policy, use the
command line argument `--play_mode` while running.

### RL
**Training**

Example with flat Unitree-A1 environment with video recordings. **Note:** Hyperparameters can be changed in [`hyperparameters.py`](controllers/rl/hyperparameters.py).
```bash
python main.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 4096 --headless --video --offscreen_render
```
**Playing**

Example with flat Unitree-A1 environment. This will automatically load the latest trained model from the logs unless a
`--model_path` is specified.
```bash
python main.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 10 --play_mode
```

### MPC
TODO

### MPC-RL
TODO