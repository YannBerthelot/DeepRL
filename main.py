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
    config["name"] = "rl-zoo"

    config_0 = copy(config)
    config_0["name"] = "FC"
    config_0["RECURRENT"] = False
    config_0["COMMON_NN_ARCHITECTURE"] = "[64]"
    config_0["ACTOR_NN_ARCHITECTURE"] = "[32]"
    config_0["CRITIC_NN_ARCHITECTURE"] = "[32]"
    config_0["NORMALIZE"] = False

    config_1 = copy(config)
    config_1["name"] = "LSTM"
    config_1["RECURRENT"] = True
    config_1["HIDDEN_SIZE"] = 32
    config_1["COMMON_NN_ARCHITECTURE"] = "[64]"
    config_1["ACTOR_NN_ARCHITECTURE"] = "[]"
    config_1["CRITIC_NN_ARCHITECTURE"] = "[]"
    config_1["NORMALIZE"] = False

    config_2 = copy(config)
    config_2["name"] = "FC"
    config_2["RECURRENT"] = False
    config_2["COMMON_NN_ARCHITECTURE"] = "[64]"
    config_2["ACTOR_NN_ARCHITECTURE"] = "[32]"
    config_2["CRITIC_NN_ARCHITECTURE"] = "[32]"
    config_2["NORMALIZE"] = True

    config_3 = copy(config)
    config_3["name"] = "LSTM"
    config_3["RECURRENT"] = True
    config_3["HIDDEN_SIZE"] = 32
    config_3["COMMON_NN_ARCHITECTURE"] = "[64]"
    config_3["ACTOR_NN_ARCHITECTURE"] = "[]"
    config_3["CRITIC_NN_ARCHITECTURE"] = "[]"
    config_3["NORMALIZE"] = True

    for i, config in enumerate([config]):
        for experiment in range(1, config["N_EXPERIMENTS"] + 1):
            if config["logging"] == "wandb":
                run = wandb.init(
                    project="LunarLander-v2 A2C RNN normalized tests-7",
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
