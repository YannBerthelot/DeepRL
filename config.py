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
    "GAMMA": 1.0,
    "NB_TIMESTEPS_TRAIN": 4e5,
    "NB_EPISODES_TEST": 50,
    "VALUE_FACTOR": 0.5,
    "ENTROPY_FACTOR": 1e-4,
    # Specific
    "N_STEPS": 5,
    # NETWORKS
    "RECURRENT": False,
    "LEARNING_RATE": 1e-5,
    "TARGET_UPDATE": 1,
    # RNN
    "HIDDEN_SIZE": 64,
    "HIDDEN_LAYERS": 1,
    "COMMON_NN_ARCHITECTURE": "[64,64]",
    "COMMON_ACTIVATION_FUNCTION": "relu",
    # Actor
    "ACTOR_NN_ARCHITECTURE": "[32,16]",
    "ACTOR_DROPOUT": 0.0,
    "ACTOR_ACTIVATION_FUNCTION": "tanh",
    # Critic
    "CRITIC_NN_ARCHITECTURE": "[32,16]",
    "CRITIC_DROPOUT": 0.0,
    "CRITIC_ACTIVATION_FUNCTION": "relu",
    # PATHS
    "TENSORBOARD_PATH": "logs",
    "MODEL_PATH": "models",
    # Experiments
    "N_EXPERIMENTS": 3,
    "EARLY_STOPPING_STEPS": 10000,
    # Logging
    "logging": "tensorboard",
    # Normalization
    "NORMALIZE": "standardize",
    "LEARNING_START": 1000,
}
