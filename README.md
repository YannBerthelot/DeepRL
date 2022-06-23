# DeepRL

DeepRL is a Python repository aiming at implementing deepRL agents using PyTorch. It is compatible with Tensorboard and wandb
Implemented agents are:
--A2C (TD and MC, n-steps is WIP) including LSTM extension
--REINFORCE

## Installation

You need to have swig installed for it to work.

```bash
git clone https://github.com/YannBerthelot/deeprlyb.git
cd deeprlyb
poetry install
```
or

```bash
pip install -i https://test.pypi.org/simple/ deeprlyb
```


## Usage

To run the example
```bash
poetry run python src/deeprlyb.py -s tests/config.ini
```
otherwise you can import agents and feed them the config like:
```python
import gym
from deeprlyb.agents import A2C

config_file = "path/to/config.ini"
config = read_config(config_file)

env = gym.make('CartPole-v1')
agent = A2C(env, config)
agent.train_MC(env, nb_timesteps)
agent.test(env, nb_episodes_test, render=True)

```
## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## To-do

- [x] Add normalization and standardization of states and rewards
- [x] Add normalization and standardization of target
- [x] Add entropy
- [x] Add schedules for parameters (e.g. learning rate)
- [x] Add LSTM (multi-layers included and batch support included)
- [x] Add rollout buffer and switch to batch learning
- [x] Package the code into a Python lib

### Priority

- [ ] Correct agent testing
- [ ] Add N-steps

### Optionnal

- [ ] Re-add n-step A2C (that works with batch)
- [ ] Add testing during training to select agent to save
- [ ] Rework action selection logging
- [ ] Finish adding tests (cf : https://andyljones.com/posts/rl-debugging.html)
- [ ] Add explained variance

