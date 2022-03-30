import wandb
import os
import gym
from n_step_A2C import A2C
from config import config
from copy import copy

if __name__ == "__main__":
    # Init folder for model saves
    os.makedirs(config["MODEL_PATH"], exist_ok=True)
    os.makedirs(config["TENSORBOARD_PATH"], exist_ok=True)

    # Init Gym env
    env = gym.make(config["ENVIRONMENT"])
    if config["logging"] == "wandb":
        wandb.tensorboard.patch(root_logdir="logs")

    config_0 = copy(config)
    config_0["name"] = "FC"
    config_0["RECURRENT"] = False
    config_0["COMMON_NN_ARCHITECTURE"] = "[64]"
    config_0["ACTOR_NN_ARCHITECTURE"] = "[32]"
    config_0["CRITIC_NN_ARCHITECTURE"] = "[32]"

    config_1 = copy(config)
    config_1["name"] = "LSTM"
    config_1["RECURRENT"] = True
    config_1["HIDDEN_SIZE"] = 32
    config_1["COMMON_NN_ARCHITECTURE"] = "[64]"
    config_1["ACTOR_NN_ARCHITECTURE"] = "[]"
    config_1["CRITIC_NN_ARCHITECTURE"] = "[]"

    for i, config in enumerate([config_0, config_1]):
        for experiment in range(1, config["N_EXPERIMENTS"] + 1):
            if config["logging"] == "wandb":
                run = wandb.init(
                    project="CartPole-v1 A2C RNN tests",
                    entity="yann-berthelot",
                    name=f'{config["name"]} {experiment}/{config["N_EXPERIMENTS"]}',
                    # sync_tensorboard=True,
                    reinit=True,
                    config=config,
                )
            # Init agent
            agent = A2C(
                env,
                config=config,
                comment=f'config {i} {experiment}/{config["N_EXPERIMENTS"]}',
            )

            # Train the agent
            agent.train(env, config["NB_TIMESTEPS_TRAIN"])

            # Load best agent from training
            agent.load("best")

            # Evaluate and render the policy
            agent.test(env, nb_episodes=config["NB_EPISODES_TEST"], render=False)
            if config["logging"] == "wandb":
                run.finish()
