import os
import numpy as np
import torch.nn as nn
import torch

def create_folder(path: str) -> None:
    """ Creates a folder if it does not exist

    Parameters:
    -----------
    path : str
        Path to the folder to be created

    Returns:
    --------
    None

    """

    if not os.path.exists(path):
        os.mkdir(path)

def mae(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Returns Mean Absolute Error

    Parameters:
    -----------
    predicted : np.ndarray
        Predicted values
    actual : np.ndarray
        Actual values
        
    Returns:
    --------
    absolute_error : np.ndarray
        Mean Absolute Error

    """
    
    absolute_error = np.abs(np.subtract(predicted, actual))

    if absolute_error.shape[0] == 1:
        return absolute_error
    else:
        return np.mean(absolute_error)

def my_KLDivLoss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem

    Parameters:
    -----------
    prediction : torch.Tensor
        Predicted values
    target : torch.Tensor
        Actual values

    Returns:
    --------    
    loss : torch.Tensor
        K-L Divergence loss
        
    """

    prediction.cpu()
    target.cpu()

    loss_func = nn.KLDivLoss(reduction='sum')
    target += 1e-16
    target_shape = target.shape[0]
    loss = loss_func(prediction, target) / target_shape
    return loss