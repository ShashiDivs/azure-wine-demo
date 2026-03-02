import torch
import torch.nn as nn
from torchinfo import summary



class WineClassification(nn.Module):

    """
    - Linear Layers 
    - ReLu non-linearity, max(0, x) faster and compuationally better
    - BatchNorm
    - Dropout reduce overfiiting 
    """

    def __init__(self, input_size, hidden_sizes=[64,32],num_classes=3, dropout_rate=0.3):

        """
        input_size = Number of input features
        hidden_features two layers
        num_classes = number of output classes
        dropoutrates = dropping neirons for reduce overfit
        """
        super(WineClassification, self).__init__()

        layers = []

        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Dropout(dropout_rate))

        for i in range(len(hidden_sizes) -1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        self.network = nn.Sequential(*layers) # 5 times ---forward --bacward 1 epoch 

    def forward(self, x):
        return self.network(x)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
        return predicted
    
    def predicted_proba(self, x):

        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probabilites = torch.softmax(outputs, dim=1)
        return probabilites


class SimpleANN(nn.Module):
    """Lightweight ANN without BatchNorm -- faster training, fewer parameters

    WHY HAVE TWO MODELS?
        This demonstrates a common pattern: start simple, add complexity only
        if needed. SimpleANN trains faster and is less likely to overfit on
        small datasets. WineClassifierANN is more powerful but needs more data.
    """

    def __init__(self, input_size, num_classes=3):
        super(SimpleANN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
        return predicted

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities

if __name__ == "__main__":

    input_size = 13
    num_classes =3

    model = WineClassification(input_size, hidden_sizes=[64,32], num_classes=num_classes)
    print(summary(model))