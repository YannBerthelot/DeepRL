import os
import gym
import wandb
from deeprlyb.agents.A2C import A2C
from deeprlyb.utils.config import read_config


if __name__ == "__main__":
    config = read_config()
    os.makedirs(config["PATHS"]["tensorboard_path"], exist_ok=True)
    env = gym.make(config["GLOBAL"]["environment"])
    config["GLOBAL"]["name"] = "MLP"

    for experiment in range(1, config["GLOBAL"].getint("n_experiments") + 1):
        if config["GLOBAL"]["logging"] == "wandb":
            run = wandb.init(
                project="Debugging",
                entity="yann-berthelot",
                name=f'{config["name"]} {experiment}/{config["GLOBAL"]["n_experiments"]}',
                reinit=True,
                config=config,
            )
        else:
            run = None
        comment = f"config_{experiment}"
        agent = A2C(env, config=config, comment=comment, run=run)
        agent.train(env, config["GLOBAL"].getfloat("nb_timesteps_train"))
        agent.load(f"{comment}_best")
        agent.test(
            env,
            nb_episodes=config["GLOBAL"].getint("nb_episodes_test"),
            render=config["GLOBAL"].getboolean("render"),
        )
        if config["GLOBAL"]["logging"] == "wandb":
            run.finish()
