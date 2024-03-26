from .base_layer import BaseLayer
import numpy as np


class CrossEntropyLoss(BaseLayer):
    def __init__(self):
        self.cache = None
        pass

    def forward(self, input_x: np.ndarray, target_y: np.ndarray):
        """
        TODO: Implement the forward pass for cross entropy loss function

        """
        N, _ = input_x.shape
        # Calculate the sum of losses for each example, loss for one example -log(e_i/sum(e_j)) where i is the
        # correct class according to the label target_y and j is sum over all classes
        single_loss = -np.log(input_x[np.arange(N), target_y] + 3e-13)
        
        loss = np.sum(single_loss)
        # Normalize the loss by dividing by the total number of samples N
        loss /= N
        # Store your loss output and input and targets in cache
        self.cache = [loss.copy(), input_x.copy(), target_y.copy()]
        return loss

    def backward(self):
        """
        TODO: Compute gradients given the true labels
        """
        # Retrieve data from cache to calculate gradients
        loss_temp, x_temp, y_temp = self.cache
        N, _ = x_temp.shape

        # Use the formula for the gradient of Cross entropy loss to calculate the gradients
        # Gradient matrix will be NxD matrix, with N rows for all the samples in the minibatch, and D=3072
        dx = np.zeros((N, _))
        dx[np.arange(N), y_temp] = (x_temp[np.arange(N), y_temp]) - 1

        assert x_temp.shape == dx.shape, "Mismatch in shape"
        # Normalize the gradient by dividing with the total number of samples N
        dx /= N

        return dx

    def zero_grad(self):
        pass
