# DeepRL

DeepRL is a Python repository aiming at implementing deepRL agents using PyTorch. It is compatible with Tensorboard and wandb
Implemented agents are:
--A2C (vanilla and n-steps) including LSTM extension
--REINFORCE

## Installation
Not really packaged at the moment, just clone the repo and use the things. Install poetry if necessary, otherwise deal with package install.

```bash
git clone https://github.com/YannBerthelot/DeepRL.git
git checkout LSTM_N_STEP_A2C
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
- [ ] Merge A2C, n_steps A2C and LSTM A2C
- [ ] Add normalization/standardization of states and rewards
- [ ] Add tests
