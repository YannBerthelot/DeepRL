class Config:
    DEVICE = "CPU"

    ## Networks
    # Actor
    ACTOR_NN_ARCHITECTURE = [64, 32]
    ACTOR_LEARNING_RATE = 1e-3
    ACTOR_DROPOUT = 0.0
    ACTOR_ACTIVATION_FUNCTION = "tanh"

    # Critic
    # Actor
    CRITIC_NN_ARCHITECTURE = [64, 32]
    CRITIC_LEARNING_RATE = 1e-3
    CRITIC_DROPOUT = 0.0
    CRITIC_ACTIVATION_FUNCTION = "relu"

    # AGENT
    GAMMA = 0.99
    NB_TIMESTEPS_TRAIN = 50000
    NB_EPISODES_TEST = 10
    MODEL_PATH = "models"
    VALUE_FACTOR = 0.5
    ENTROPY_FACTOR = 0.01
    # Number of steps to run before running updates on them
    BATCH_SIZE = 1
    N_STEPS = 2

    # Tensorboard
    TENSORBOARD_PATH = "logs"
