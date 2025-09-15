# --------------------------------
# api.py
# This file contains the API for the MLOpsDemo project.
# It defines the LinearNet class and the fit_predict function.
# It's only for the demonstration purpose for the MLOpsDemo project.
# --------------------------------

import torch
import torch.nn as nn
import torch.optim as optim

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(1, 1)
    def forward(self, x):
        return self.fc1(x)
    
class Trainer:
    """
    A class for training a linear model.
    """
    def __init__(self):
        self.model = LinearNet()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        

    def fit(self, X, y, epochs=100):
        """
        Train the model for a given number of epochs.
        """
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(X)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
        return loss.item()

    def predict(self, X):
        """
        Predict the output for a given input.
        """
        with torch.no_grad():
            return self.model(X)

def fit_predict(X, y, epochs=100):
    """
    Fit the model and predict the output for a given input.
    """
    trainer = Trainer()
    loss = trainer.fit(X, y, epochs)
    print(f"Final Loss: {loss}")
    return trainer.predict(X), loss