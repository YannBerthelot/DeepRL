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
    "GAMMA": 0.995,
    "NB_TIMESTEPS_TRAIN": 2e5,
    "NB_EPISODES_TEST": 10,
    "VALUE_FACTOR": 0.3,
    "ENTROPY_FACTOR": 0.01,
    "KL_FACTOR": 0.1,
    "LEARNING_START": 5e3,
    # Specific
    "N_STEPS": 5,
    # NETWORKS
    "RECURRENT": False,
    "GRADIENT_CLIPPING": 500,
    "LEARNING_RATE": 5e-4,
    "LEARNING_RATE_END": 5e-5,
    # RNN
    "HIDDEN_SIZE": 64,
    "HIDDEN_LAYERS": 1,
    "COMMON_NN_ARCHITECTURE": "[64,64]",
    "COMMON_ACTIVATION_FUNCTION": "relu",
    # Actor
    "ACTOR_NN_ARCHITECTURE": "[64,32,8]",
    "ACTOR_ACTIVATION_FUNCTION": "tanh",
    # Critic
    "CRITIC_NN_ARCHITECTURE": "[32,8]",
    "CRITIC_ACTIVATION_FUNCTION": "relu",
    # PATHS
    "TENSORBOARD_PATH": "logs",
    "MODEL_PATH": "models",
    # Experiments
    "N_EXPERIMENTS": 10,
    "EARLY_STOPPING_STEPS": 10000,
    # Logging
    "logging": "tensorboard",
    # Normalization
    "SCALING": True,
    "SCALING_METHOD": "standardize",
    "TARGET_SCALING": True,
    # Continuous
    "CONTINUOUS": False,
    "LAW": "normal",
}
