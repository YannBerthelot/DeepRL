import wandb
from n_step_A2C import A2C
from config import config
from pymgrid_utils import get_environments

config["name"] = "pymgrid"

for experiment in range(1, config["N_EXPERIMENTS"] + 1):
    run = wandb.init(
        project="Pymgrid tests",
        entity="yann-berthelot",
        name=f'{config["name"]} {experiment}/{config["N_EXPERIMENTS"]}',
        reinit=True,
        config=config,
    )
    mg_env_train, mg_env_eval = get_environments()
    agent = A2C(
        mg_env_train, config=config, comment=f"baseline_pymgrid_{experiment}", run=run
    )

    # Train the agent
    # agent.train(mg_env_train, config["NB_TIMESTEPS_TRAIN"])

    # Load best agent from training
    agent.env = mg_env_eval
    agent.load(f"config_pymgrid_{experiment}_best")

    # Evaluate and render the policy
    agent.test(
        mg_env_eval,
        nb_episodes=config["NB_EPISODES_TEST"],
        render=False,
        scaler_file=f"data/baseline_pymgrid_{experiment}_obs_scaler.pkl",
    )
    if config["logging"] == "wandb":
        run.finish()
