class Config:
    DEVICE = "CPU"

    ## Networks
    # Policy
    POLICY_NN_ARCHITECTURE = [64, 32, 16]
    POLICY_LEARNING_RATE = 1e-3
    ENTROPY_FACTOR = 0.02
    DROPOUT = 0.0

    # AGENT
    GAMMA = 0.99
    NB_EPISODES_PER_EPOCH = 1
    NB_EPOCH = 2000
    NB_EPISODES_TEST = 10
    MODEL_PATH = "models"

    # Used for normalization purpose in the update.
    BATCH_SIZE = 10
