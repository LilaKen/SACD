import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_size=2048, num_classes=5):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),  # Assuming 32 filters, which can be changed.
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * (input_size // 2), 128),
            # 32 is the number of filters, and input_size//2 because of maxpooling
            nn.LeakyReLU(),
            nn.Linear(128, num_classes),
            # Remove the nn.Sigmoid() here if you're using nn.CrossEntropyLoss in training
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x


class CNN_CWT(nn.Module):
    def __init__(self, input_size=4096, num_classes=5):
        super(CNN_CWT, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, stride=1, padding=2),  # Assuming 32 filters, which can be changed.
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * (input_size // 2), 128),
            # 32 is the number of filters, and input_size//2 because of maxpooling
            nn.LeakyReLU(),
            nn.Linear(128, num_classes)
            # Remove the nn.Sigmoid() here if you're using nn.CrossEntropyLoss in training
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x