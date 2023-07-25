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

    print(f'The random selected users for FedAvg are {clients}')

    # Perform federated learning
    for i in range(global_epochs):
        print(f'Global epoch # {i}')
        # Select active clients for fedAvg
        active = active_user(clients, top_k)
        transition_prob = wireless_channel_transition_probability(active)
        successfull_users = []
        for i in range(len(active)):
            if transition_prob[i] in state_1:
                successfull_users.append(active[i])

            elif transition_prob[i] in state_0:
                pass

        print(f'Successfull clients for FedAvg are: {successfull_users}')

        # This is for the fedAvg training
        for i in range(len(successfull_users)):
            # print(f"This is user {i+1}, their ID is: {active[i]}")
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

    # select top K clients after indexing
    transition_prob = wireless_channel_transition_probability(active)
    # print(transition_prob)

    return global_model


# Set the parameters for federated learning
def clients_pool(num_clients):
    clients = []
    for i in range(num_clients):
        clients.append(i)
    return clients

# assign power to clients


def power(clients):
    clients_power = []
    for i in range(len(clients)):
        rand = random.randint(1, 100)
        clients_power.append(rand)
    return clients_power


num_clients = int(input('Please Enter the number of clients: '))
clients = clients_pool(num_clients)
clients_power = power(clients)
global_epochs = 3
local_epochs = 5
learning_rate = 0.01
selected_users = 10
top_k = 5
state_0 = [0.9449, 0.0087, 0.9913]
state_1 = [0.0551, 0.8509, 0.1491]

# slect a random number of active users


def user_selection_fedAvg(clients, selected_users):
    if selected_users >= len(clients):
        return clients

    subset = random.sample(clients, selected_users)
    return subset

# select subset from the random selected users


def active_user(clients, top_k):
    if top_k >= len(clients):
        return clients
    active = random.sample(clients, top_k)
    return active

# transition probability calculation


def wireless_channel_transition_probability(clients):
    states = [0.9449, 0.0087, 0.9913, 0.0551, 0.8509, 0.1491]
    clients_state = []
    for i in range(len(clients)):
        temp = random.choice(states)
        clients_state.append(temp)

    return clients_state


# Run federated learning
global_model = federated_learning(user_selection_fedAvg(
    clients, selected_users), local_epochs, global_epochs, learning_rate)
