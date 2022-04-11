# DeepRL

DeepRL is a Python repository aiming at implementing deepRL agents using PyTorch. It is compatible with Tensorboard and wandb
Implemented agents are:
--A2C (vanilla and n-steps) including LSTM extension
--REINFORCE

## Installation

Not really packaged at the moment, just clone the repo and use the things. Install poetry if necessary, otherwise deal with package install.

```bash
git clone https://github.com/YannBerthelot/DeepRL.git
cd DeepRL
poetry install
```

## Usage

```bash
poetry run python main.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## To-do

- [x] Merge n_steps A2C and LSTM A2C
- [x] Merge A2C and n_steps A2C
- [x] Add normalization and standardization of states and rewards
- [x] Add normalization and standardization of target
- [x] Add entropy
- [ ] Add units tests
- [x] Add policy and target networks
- [x] Add explained variance
- [ ] Add testing during training to select agent to save
- [ ] Rework action selection logging
- [x] Add schedules for parameters (e.g. learning rate)
