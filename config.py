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
    "ENVIRONMENT": "CartPole-v1",
    # AGENT INFO
    # General
    "AGENT": "n-steps A2C",
    "GAMMA": 0.99,
    "NB_TIMESTEPS_TRAIN": 2e5,
    "NB_EPISODES_TEST": 10,
    "VALUE_FACTOR": 0.5,
    "ENTROPY_FACTOR": 0.0,
    "KL_FACTOR": 0.0000,
    "LEARNING_START": 1e2,
    # Specific
    "N_STEPS": 1,
    # NETWORKS
    "RECURRENT": True,
    "GRADIENT_CLIPPING": 5e2,
    "LEARNING_RATE": 1e-3,
    "LEARNING_RATE_END": 1e-5,
    # RNN
    "HIDDEN_SIZE": 64,
    "HIDDEN_LAYERS": 2,
    "COMMON_NN_ARCHITECTURE": "[64]",
    "COMMON_ACTIVATION_FUNCTION": "relu",
    # Actor
    "ACTOR_NN_ARCHITECTURE": "[32]",
    "ACTOR_ACTIVATION_FUNCTION": "tanh",
    # Critic
    "CRITIC_NN_ARCHITECTURE": "[32]",
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
    "SCALING": True,
    "SCALING_METHOD": "standardize",
    "TARGET_SCALING": False,
    # Continuous
    "CONTINUOUS": False,
    "LAW": "normal",
    "BUFFER_SIZE": 5,
    "NORMALIZE_ADVANTAGES": False,
}
