"""configs stored for model building"""

import os

# Supress tensorflow warnings and GPU informations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Config Variables
IMAGE_SIZE: tuple[int, int] = (256, 256)  # Size of each image
BATCH_SIZE: int = 32  # Batch size for training purpose
DATA_DIR: str = 'data/PlantVillage'  # data directory for training
EPOCHS: int = 1  # Number of training epochs
CHANNELS: int = 3  # Number of channels in image(RGB)
TRIAN_SIZE: float = 0.8  # Training data out of complete data
VALIDATION_SIZE: float = 0.1  # Validation data out of complete data
TEST_SIZE: float = 0.1  # Test data out of complete data
