import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import random

# Define the model architecture


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to train the model on local data and calculate accuracy


def train(model, train_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        print(
            f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):0.2f}, Accuracy = {accuracy:0.2f}%")

# Simulate federated learning using the MNIST dataset


def federated_learning(num_clients, num_epochs, learning_rate):
    # Create the global model
    global_model = Net()
    criterion = nn.CrossEntropyLoss()

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(
        root='./data', train=True, transform=transforms.ToTensor(), download=True)

    # Split the dataset into client data
    client_data = torch.utils.data.random_split(
        train_dataset, [len(train_dataset) // len(num_clients)] * len(num_clients))

    # Perform federated learning
    print(f'The random selected users are {num_clients}')
    for i in range(len(num_clients)):
        # Create a local copy of the global model
        print(f"This is user {num_clients[i]}")
        local_model = Net()
        local_model.load_state_dict(global_model.state_dict())

        # Get the local client data
        train_data = client_data[i]
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=10, shuffle=True)

        # Create an optimizer for the local model
        optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)

        # Train the local model
        train(local_model, train_loader, optimizer, criterion, num_epochs)

        # Update the global model with the local model's parameters
        global_model.load_state_dict(local_model.state_dict())

    return global_model


# Set the parameters for federated learning
num_clients = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
num_epochs = 5
learning_rate = 0.01
selected_users = 5


def user_selection(num_clients, subset_size):
    if subset_size >= len(num_clients):
        return num_clients

    subset = random.sample(num_clients, selected_users)
    return subset


# Run federated learning
global_model = federated_learning(user_selection(
    num_clients, selected_users), num_epochs, learning_rate)
