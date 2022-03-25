import wandb
import os
import gym
from n_step_A2C import A2C
from config import config

if __name__ == "__main__":
    # Init folder for model saves
    os.makedirs(config["MODEL_PATH"], exist_ok=True)
    os.makedirs(config["TENSORBOARD_PATH"], exist_ok=True)

    # Init Gym env
    env = gym.make(config["ENVIRONMENT"])
    wandb.tensorboard.patch(root_logdir="logs")
    wandb.init(
        project="my-test-project", entity="yann-berthelot", sync_tensorboard=True
    )

    for n_step in range(1, 10):

        config["N_STEPS"] = n_step
        wandb.config = config
        # Init agent
        comment = f"n_step : {n_step}"
        agent = A2C(env, config=config, comment=comment)
        # Train the agent
        agent.train(env, config["NB_TIMESTEPS_TRAIN"])

        # Load best agent from training
        agent.load("best")

        # Evaluate and render the policy
        agent.test(env, nb_episodes=config["NB_EPISODES_TEST"], render=False)
