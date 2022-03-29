import wandb
import os
import gym
from n_step_A2C import A2C, A2CRecurrent
from config import config

if __name__ == "__main__":
    # Init folder for model saves
    os.makedirs(config["MODEL_PATH"], exist_ok=True)
    os.makedirs(config["TENSORBOARD_PATH"], exist_ok=True)

    # Init Gym env
    env = gym.make(config["ENVIRONMENT"])
    if config["logging"] == "wandb":
        wandb.tensorboard.patch(root_logdir="logs")

    for lr in range(3, 4 + 1):
        for experiment in range(1, config["N_EXPERIMENTS"] + 1):
            config["ACTOR_LEARNING_RATE"] = 10 ** (-lr)
            config["CRITIC_LEARNING_RATE"] = 10 ** (-lr)
            config["LEARNING_RATE"] = 10 ** (-lr)
            if config["logging"] == "wandb":
                run = wandb.init(
                    project="CartPole-v1 A2C RNN tests",
                    entity="yann-berthelot",
                    name=f'lr : 1e-{lr} {experiment}/{config["N_EXPERIMENTS"]}',
                    # sync_tensorboard=True,
                    reinit=True,
                    config=config,
                )
            # Init agent
            if config["NETWORK_TYPE"] == "rnn":
                agent = A2CRecurrent(
                    env,
                    config=config,
                    comment=f'lr : 1e-{lr} {experiment}/{config["N_EXPERIMENTS"]}',
                )
            else:
                agent = A2C(env, config=config)

            # Train the agent
            agent.train(env, config["NB_TIMESTEPS_TRAIN"])

            # Load best agent from training
            agent.load("best")

            # Evaluate and render the policy
            agent.test(env, nb_episodes=config["NB_EPISODES_TEST"], render=False)
            if config["logging"] == "wandb":
                run.finish()
