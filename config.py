import platform
import torch

if torch.cuda.is_available():
    GPU_NAME = torch.cuda.get_device_name(0)
else:
    GPU_NAME = "No GPU"
config = {
    # DEVICES
    "DEVICE": "CPU",
    "CPU": platform.processor(),
    "GPU": GPU_NAME,
    # GLOBAL INFO
    "ENVIRONMENT": "LunarLander-v2",
    # AGENT INFO
    # General
    "AGENT": "n-steps A2C",
    "GAMMA": 0.99,
    "NB_TIMESTEPS_TRAIN": 5e4,
    "NB_EPISODES_TEST": 50,
    "VALUE_FACTOR": 0.0,
    "ENTROPY_FACTOR": 0.0,
    # Specific
    "N_STEPS": 1,
    # NETWORKS
    "Network type":"RNN",
    # Actor
    "ACTOR_NN_ARCHITECTURE": "[64, 32]",
    "ACTOR_LEARNING_RATE": 1e-3,
    "ACTOR_DROPOUT": 0.0,
    "ACTOR_ACTIVATION_FUNCTION": "tanh",
    # Critic
    "CRITIC_NN_ARCHITECTURE": "[64, 32]",
    "CRITIC_LEARNING_RATE": 1e-3,
    "CRITIC_DROPOUT": 0.0,
    "CRITIC_ACTIVATION_FUNCTION": "relu",
    # PATHS
    "TENSORBOARD_PATH": "logs",
    "MODEL_PATH": "models",
    # Experiments
    "N_EXPERIMENTS": 10,
}
