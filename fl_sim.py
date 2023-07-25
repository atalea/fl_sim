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


def train(model, train_loader, optimizer, criterion, local_epochs):
    model.train()
    for epoch in range(local_epochs):
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
        # print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):0.2f}, Accuracy = {accuracy:0.2f}%")
        return accuracy, float(loss)

# Simulate federated learning using the MNIST dataset


def federated_learning(clients, local_epochs, global_epochs,  learning_rate):
    # Create the global model
    global_model = Net()
    criterion = nn.CrossEntropyLoss()

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(
        root='./data', train=True, transform=transforms.ToTensor(), download=True)

    # Split the dataset into client data
    client_data = torch.utils.data.random_split(
        train_dataset, [len(train_dataset) // len(clients)] * len(clients))

    print(f'The random selected users are {clients}')
    # Perform federated learning
    for i in range(global_epochs):
        print(f'Global epoch # {i}')

        for i in range(len(clients)):
            # select active users
            # active = active_users(num_clients, active_users)
            # Create a local copy of the global model
            # print(f"This is user {i+1}, their ID is: {clients[i]}")
            local_model = Net()
            local_model.load_state_dict(global_model.state_dict())

            # Get the local client data
            train_data = client_data[i]
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=10, shuffle=True)

            # Create an optimizer for the local model
            optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)

            # Train the local model
            acc, loss = train(local_model, train_loader,
                              optimizer, criterion, local_epochs)

        # Update the global model with the local model's parameters
        global_model.load_state_dict(local_model.state_dict())
        print(f'Global Accuracy {acc: .2f}, global loss {loss: .2f}')

    return global_model


# Set the parameters for federated learning
def clients_pool(num_clients):
    clients = []
    for i in range(num_clients):
        clients.append(i)
    return clients


num_clients = int(input('Please Enter the number of clients: '))
clients = clients_pool(num_clients)
global_epochs = 2
local_epochs = 5
learning_rate = 0.01
selected_users = 10
active_users = 5


# slect a random number of active users
def user_selection_fedAvg(clients, selected_users):
    if selected_users >= len(clients):
        return clients

    subset = random.sample(clients, selected_users)
    return subset


def active_user(clients, active_users):
    if active_users >= len(clients):
        return clients
    active = random.sample(clients, active_users)
    return active


# Run federated learning
global_model = federated_learning(user_selection_fedAvg(
    clients, selected_users), local_epochs, global_epochs, learning_rate)
