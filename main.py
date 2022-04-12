import wandb
import os
import gym
from A2C import A2C
from config import config
from copy import copy
import cProfile

if __name__ == "__main__":
    # Init folder for model saves

    os.makedirs(config["TENSORBOARD_PATH"], exist_ok=True)

    # Init Gym env
    env = gym.make(config["ENVIRONMENT"])
    config_0 = copy(config)
    config_0["name"] = "MLP"

    for i, config in enumerate([config_0]):
        for experiment in range(1, config["N_EXPERIMENTS"] + 1):
            if config["logging"] == "wandb":
                run = wandb.init(
                    project="LunarLander A2C RNN discrete",
                    entity="yann-berthelot",
                    name=f'{config["name"]} {experiment}/{config["N_EXPERIMENTS"]}',
                    reinit=True,
                    config=config,
                )
            else:
                run = None
            # Init agent
            comment = f"config_{i}_{experiment}"
            agent = A2C(env, config=config, comment=comment, run=run)
            # Train the agent
            agent.train(env, config["NB_TIMESTEPS_TRAIN"])

            # Load best agent from training
            agent.load(f"{comment}_best")

            # Evaluate and render the policy
            agent.test(
                env,
                nb_episodes=config["NB_EPISODES_TEST"],
                render=True,
                scaler_file=f"data/{comment}_obs_scaler.pkl",
            )

            if config["logging"] == "wandb":
                run.finish()
