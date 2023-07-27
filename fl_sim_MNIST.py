import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import random
import heapq
import numpy as np

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
# This is FedAvg algorithm


def federated_learningFedAvg(clients, local_epochs, global_epochs,  learning_rate):
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
    print('This is FedAvg algorithm')
    for i in range(global_epochs):
        print(f'Global epoch # {i}')
        # Select active clients for fedAvg
        active = active_user(clients, top_k)
        wireless_channel_transition_probability(active)
        print(clients_state)
        successfull_users = []
        for i in range(len(active)):
            if clients_state[i] == 1:
                successfull_users.append(active[i])

            else:
                pass

        print(f'Successfull clients for FedAvg are: {successfull_users}')

        # This is for the fedAvg training
        if (successfull_users == []):
            print('No user is able to transmit')
            print(f'Global Accuracy {acc: .2f}, global loss {loss: .2f}')
        else:

            for i in range(len(successfull_users)):
                # print(f"This is user {i+1}, their ID is: {active[i]}")
                local_model = Net()
                local_model.load_state_dict(global_model.state_dict())

                # Get the local client data
                train_data = client_data[i]
                train_loader = torch.utils.data.DataLoader(
                    train_data, batch_size=10, shuffle=True)

                # Create an optimizer for the local model
                optimizer = optim.SGD(
                    local_model.parameters(), lr=learning_rate)

                # Train the local model
                acc, loss = train(local_model, train_loader,
                                  optimizer, criterion, local_epochs)

            # Update the global model with the local model's parameters
            global_model.load_state_dict(local_model.state_dict())
            print(f'Global Accuracy {acc: .2f}, global loss {loss: .2f}')

    return global_model


def federated_learningIBCS(clients, local_epochs, global_epochs,  learning_rate):
    # Create the global model
    global_model = Net()
    criterion = nn.CrossEntropyLoss()

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(
        root='./data', train=True, transform=transforms.ToTensor(), download=True)

    # Split the dataset into client data
    client_data = torch.utils.data.random_split(
        train_dataset, [len(train_dataset) // len(clients)] * len(clients))

    # select top K clients after indexing
    print('\n')
    print('This is IBCS algorithm')
    # Perform federated learning
    for i in range(global_epochs):
        print(f'Global epoch # {i}')
        # Select active clients for top_k indexing algorithm
        wireless_channel_transition_probability(clients)
        active = clients_indexing(clients, clients_power)
        print(clients_state)
        successfull_users = []
        for i in range(len(active)):
            if clients_state[i] == 1:
                successfull_users.append(active[i])

            else:
                pass

        print(f'Successfull clients for IBCS are: {successfull_users}')

        # This is for the fedAvg training
        if (successfull_users == []):
            print('No user is able to transmit')
            print(f'Global Accuracy {acc: .2f}, global loss {loss: .2f}')
        else:

            for i in range(len(successfull_users)):
                # print(f"This is user {i+1}, their ID is: {active[i]}")
                local_model = Net()
                local_model.load_state_dict(global_model.state_dict())

                # Get the local client data
                train_data = client_data[i]
                train_loader = torch.utils.data.DataLoader(
                    train_data, batch_size=10, shuffle=True)

                # Create an optimizer for the local model
                optimizer = optim.SGD(
                    local_model.parameters(), lr=learning_rate)

                # Train the local model
                acc, loss = train(local_model, train_loader,
                                  optimizer, criterion, local_epochs)

            # Update the global model with the local model's parameters
            global_model.load_state_dict(local_model.state_dict())
            print(f'Global Accuracy {acc: .2f}, global loss {loss: .2f}')
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
top_k = 3
state_0 = [0.9449, 0.0087, 0.9913]
state_1 = [0.0551, 0.8509, 0.1491]
clients_state = []

# slect a random number of active users


def user_selection_fedAvg(clients, selected_users):
    if selected_users >= len(clients):
        return clients

    subset = random.sample(clients, selected_users)
    return subset

# select subset from the random selected users for fedAvg


def active_user(clients, top_k):
    if top_k >= len(clients):
        return clients
    active = random.sample(clients, top_k)
    return active

# select top_k users


def clients_indexing(clients, clients_power):
    user_indices = []
    for i in range(len(clients)):
        if clients_state[i] == 1:
            v_i_t = -(state_1[1]/len(clients)) - \
                (((state_1[0]*clients_power[i])/100))
            user_indices.append(v_i_t)
            # print(f'client {clients[i]}, is in state {clients_state[i]}')
        elif clients_state[i] == 0:
            v_i_t = -(state_0[1]/len(clients)) - \
                (((state_0[0]*clients_power[i])/100))
            user_indices.append(v_i_t)
    # print('Indices are', user_indices)
    # this prints the top k values
    top_k_users = heapq.nlargest(top_k, user_indices)
    # print(f'the top {top_k} users who can transmit are: {top_k_users}')
    # this prints the top k indices
    user_indices = np.argsort(user_indices)
    top_k_users = user_indices[-top_k:]
    # print(f'the top {top_k} users who can transmit are: {top_k_users}')
    # print(f'client {clients[i]}, is in state {clients_state[i]}')
    return top_k_users

# transition probability calculation


def wireless_channel_transition_probability(clients):
    temp = []
    if clients_state == []:
        print('This is time 0')
        for i in range(len(clients)):
            rand_transision = random.random()
            temp.append(rand_transision)
        # print(f'This is temp trans{temp}')
        for i in range(len(temp)):
            if temp[i] <= state_0[0] and temp[i] > state_1[0]:
                clients_state.append(0)
            else:
                clients_state.append(1)
    else:
        print('This is Not time 0')
        for i in range(len(clients)):
            rand_transision = random.random()
            temp.append(rand_transision)
        # print(f'This is temp trans{temp}')
        for i in range(len(temp)):
            if clients_state[i] == 0 and temp[i] >= state_0[1] and temp[i] < state_1[1]:
                clients_state[i] = 1
            elif clients_state[i] == 1 and temp[i] >= state_1[1]:
                clients_state[i] = 1
            elif clients_state[i] == 0 and temp[i] <= state_0[2] and temp[i] > state_1[2]:
                clients_state[i] = 0
            else:
                clients_state[i] = 0


# Run federated learning
global_model = federated_learningFedAvg(user_selection_fedAvg(
    clients, selected_users), local_epochs, global_epochs, learning_rate)

# reset states before you start the second algorithm
clients_state = []

global_model = federated_learningIBCS(
    clients, local_epochs, global_epochs, learning_rate)
