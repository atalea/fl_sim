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

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        print(
            f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader)}, Accuracy = {accuracy}%")

# Function to randomly select a subset of users


def select_users(pool_users, num_selected_users, min_samples, train_dataset):
    selected_users = []

    while len(selected_users) < num_selected_users:
        user_index = random.choice(pool_users)
        train_indices = torch.where(train_dataset.targets == user_index)[0]

        if len(train_indices) >= min_samples:
            selected_users.append(user_index)

    return selected_users

# Function for model aggregation


def aggregate_models(global_model, local_models):
    for global_param, local_params in zip(global_model.parameters(), zip(*[local_model.parameters() for local_model in local_models])):
        global_param.data = torch.mean(torch.stack(
            [local_param.data for local_param in local_params]), dim=0)

# Simulate federated learning using the MNIST dataset


def federated_learning(num_users, num_selected_users, num_epochs, learning_rate, min_samples):
    # Create the global model
    global_model = Net()
    criterion = nn.CrossEntropyLoss()

    # Load the MNIST dataset
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root='./data', train=True, transform=transform, download=True)

    # Create a pool of user indices
    pool_users = list(range(num_users))

    # Select a subset of users
    selected_users = select_users(
        pool_users, num_selected_users, min_samples, train_dataset)

    # Perform federated learning
    local_models = []
    for user_index in selected_users:
        # Create a local model
        local_model = Net()
        local_model.load_state_dict(global_model.state_dict())

        # Get the local client data
        train_indices = torch.where(train_dataset.targets == user_index)[0]

        # Check if the selected user has enough samples
        if len(train_indices) >= min_samples:
            train_data = torch.utils.data.Subset(train_dataset, train_indices)
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=10, shuffle=True)

            # Train the local model
            train(local_model, train_loader, optim.SGD(
                local_model.parameters(), lr=learning_rate), criterion, num_epochs)

            # Add the local model to the list
            local_models.append(local_model)

    # Aggregate local models to update the global model
    aggregate_models(global_model, local_models)

    return global_model


# Set the parameters for federated learning
num_users = 100
num_selected_users = 10
num_epochs = 5
learning_rate = 0.01
min_samples = 100

# Run federated learning
global_model = federated_learning(
    num_users, num_selected_users, num_epochs, learning_rate, min_samples)
