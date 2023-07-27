import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import random
import heapq
import numpy as np

# Define the model architecture


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = nn.functional.relu(self.fc1(x))
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

# Function to test the global model


def test(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation during testing
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    average_loss = running_loss / len(test_loader)
    return accuracy, average_loss
# This is FedAvg algorithm


def federated_learningFedAvg(clients, local_epochs, global_epochs,  learning_rate):
    temp_power = []
    # Create the global model
    global_model = CNNModel()
    criterion = nn.CrossEntropyLoss()

    # Load the MNIST dataset
    trans_mnist = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=trans_mnist)
    # print(f'trainf data leng is {len(train_dataset)}')

    test_dataset = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=False, download=True,
                       transform=transforms.ToTensor()),
        shuffle=False
    )
    # print(f'test data leng is {len(test_dataset)}')

    # Split the dataset into client data
    client_data = torch.utils.data.random_split(
        train_dataset, [len(train_dataset) // len(clients)] * len(clients))

    # print(f'The random selected users for FedAvg are {clients}')

    # Perform federated learning
    print('This is FedAvg algorithm')
    for i in range(global_epochs):
        print(f'Global epoch # {i}')
        # Select active clients for fedAvg
        active = active_user(clients, top_k)
        wireless_channel_transition_probability(active)
        # print(f'client status{clients_state}')
        successfull_users = []
        for i in range(len(active)):
            if clients_state[active[i]] == 0:
                successfull_users.append(active[i])
            # calculate non-successfull users power
            else:
                temp_power.append(clients_power[active[i]])
        # print(f'temp power {temp_power}')
        # print(f'Successfull clients for FedAvg are: {successfull_users}')

        # This is for the fedAvg training
        if (successfull_users == []):
            print('No user is able to transmit')
            # print(f'Global Accuracy {acc: .2f}, global loss {loss: .2f}')
        else:

            for i in range(len(successfull_users)):
                # print(f"This is user {i+1}, their ID is: {active[i]}")
                local_model = CNNModel()
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
            # print(f'Global Accuracy {acc: .2f}, global loss {loss: .2f}')
            test_accuracy, test_loss = test(
                global_model, test_dataset, criterion)
            # print(f"Testing Accuracy: {test_accuracy:.2f}%")
            # print(f"Testing Loss: {test_loss:.2f}")
            fedavg_accu.append(test_accuracy)
            fedavg_loss.append(test_loss)
            fedavg_power.append(temp_power)

    return global_model


def federated_learningIBCS(clients, local_epochs, global_epochs,  learning_rate):
    # Create the global model
    temp_power = []
    global_model = CNNModel()
    criterion = nn.CrossEntropyLoss()

    # Load the MNIST dataset
    trans_mnist = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=trans_mnist)
    # print(f'trainf data leng is {len(train_dataset)}')

    test_dataset = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=False, download=True,
                       transform=transforms.ToTensor()),
        shuffle=False
    )
    # print(f'test data leng is {len(test_dataset)}')

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
        # print(f'active are {active}')
        # print(f'status {clients_state}')
        successfull_users = []
        for i in range(len(active)):
            if clients_state[active[i]] == 0:
                successfull_users.append(active[i])
            else:
                temp_power.append(clients_power[active[i]])
        # print(f'temp power {temp_power}')
        # print(f'Successfull clients for ibcs are: {successfull_users}')

        # This is for the fedAvg training
        if (successfull_users == []):
            print('No user is able to transmit')
            # print(f'Global Accuracy {acc: .2f}, global loss {loss: .2f}')
        else:

            for i in range(len(successfull_users)):
                # print(f"This is user {i+1}, their ID is: {active[i]}")
                local_model = CNNModel()
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
            # print(f'Global Accuracy {acc: .2f}, global loss {loss: .2f}')
            test_accuracy, test_loss = test(
                global_model, test_dataset, criterion)
            # print(f"Testing Accuracy: {test_accuracy:.2f}%")
            # print(f"Testing Loss: {test_loss:.2f}")
            ibcs_accu.append(test_accuracy)
            ibcs_loss.append(test_loss)
            ibcs_power.append(temp_power)
    # print(transition_prob)
    # collect the data (power, loss, acc) --> plot()

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
global_epochs = 2
local_epochs = 2
learning_rate = 0.01
selected_users = 10
top_k = 3
state_0 = [0.9449, 0.0087, 0.9913]
state_1 = [0.0551, 0.8509, 0.1491]
clients_state = []

fedavg_accu = []
fedavg_loss = []
fedavg_power = []
ibcs_accu = []
ibcs_loss = []
ibcs_power = []

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
            if rand_transision <= state_0[0]:
                # print(f'random here is {rand_transision}')
                clients_state.append(0)
            else:
                # print(f'random here is {rand_transision}')
                clients_state.append(1)
    else:
        # print('This is Not time 0')
        for i in range(len(clients)):
            rand_transision = random.random()
            # print(f'random here is {rand_transision}')
            if clients_state[i] == 0:
                if rand_transision <= state_0[1]:
                    clients_state[i] = 1
                else:
                    clients_state[i] = 0
            else:
                if rand_transision <= state_0[2]:
                    clients_state[i] = 0
                else:
                    clients_state[i] = 1


# Run federated learning
global_model = federated_learningFedAvg(user_selection_fedAvg(
    clients, selected_users), local_epochs, global_epochs, learning_rate)

print(f'fedavg accuracy {fedavg_accu: .2f}')
print(f'fedavg loss {fedavg_loss: .2f}')
print(f'fedavg power {fedavg_power}')


# reset states before you start the second algorithm
clients_state = []

global_model = federated_learningIBCS(
    clients, local_epochs, global_epochs, learning_rate)

print(f'ibcs accuracy {ibcs_accu}')
print(f'ibcs loss {ibcs_loss}')
print(f'ibcs power {ibcs_power}')
