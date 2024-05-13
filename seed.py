import random
import numpy as np
import torch


def set_seeds(seed_value=2023):
    """
    Set seeds for different random number generators to `seed_value`.

    Parameters:
    - seed_value (int): The value of the seed to set.
    """
    # Python's random
    random.seed(seed_value)

    # NumPy
    np.random.seed(seed_value)

    # PyTorch
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seeds set to {seed_value}")
