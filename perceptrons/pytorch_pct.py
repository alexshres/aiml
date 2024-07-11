import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Load and preprocess the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, 
                               download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, 
                              download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=64, shuffle=False)

# Using inheritance to grab all attributes of nn.Module class
class PyTorchPCT(nn.Module):
    def __init__(self, input_size, num_classes, learning_rate=0.01, epochs=10):
        super(PyTorchPCT, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)  # Fully connected layer
        self.learning_rate = learning_rate
        self.epochs = epochs

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        out = self.fc(x)
        return out

    def train_model(self, train_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            for images, labels in train_loader:
                outputs = self(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')

    def evaluate_model(self, test_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')
