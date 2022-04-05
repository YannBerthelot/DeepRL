import wandb
import os
import gym
from n_step_A2C import A2C
from config import config
from copy import copy

if __name__ == "__main__":
    # Init folder for model saves

    os.makedirs(config["TENSORBOARD_PATH"], exist_ok=True)

    # Init Gym env
    env = gym.make(config["ENVIRONMENT"])

    config_0 = copy(config)
    config_0["name"] = "No target network"
    config_0["TARGET_UPDATE"] = 1

    config_1 = copy(config)
    config_1["name"] = "Target network"
    config_1["TARGET_UPDATE"] = 16

    config_2 = copy(config)
    config_2["name"] = "Target network"
    config_2["TARGET_UPDATE"] = 128

    config_3 = copy(config)
    config_3["name"] = "Target network"
    config_3["TARGET_UPDATE"] = 1024

    for i, config in enumerate([config_0, config_1, config_2, config_3]):
        for experiment in range(1, config["N_EXPERIMENTS"] + 1):
            if config["logging"] == "wandb":
                run = wandb.init(
                    project="LunarLander-v2 A2C RNN normalized tests-10",
                    entity="yann-berthelot",
                    name=f'{config["name"]} {experiment}/{config["N_EXPERIMENTS"]}',
                    reinit=True,
                    config=config,
                )
            # Init agent
            agent = A2C(
                env,
                config=config,
                comment=f"config {i} {experiment}",
            )

            # Train the agent
            agent.train(env, config["NB_TIMESTEPS_TRAIN"])

            # Load best agent from training
            agent.load(f"config {i} {experiment}_best")

            # Evaluate and render the policy
            agent.test(env, nb_episodes=config["NB_EPISODES_TEST"], render=True)
            if config["logging"] == "wandb":
                run.finish()
