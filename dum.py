import random
import heapq
import numpy as np


def clients_pool(num_clients):
    clients = []
    for i in range(num_clients):
        clients.append(i)
    return clients


def power(clients):
    clients_power = []
    for i in range(len(clients)):
        rand = random.randint(1, 100)
        clients_power.append(rand)
    return clients_power


def wireless_channel_transition_probability(clients):
    temp = []
    if clients_state == []:
        print('This is time 0')
        for i in range(len(clients)):
            rand_transision = random.random()
            temp.append(rand_transision)
        print(f'This is temp trans{temp}')
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
        print(f'This is temp trans{temp}')
        for i in range(len(temp)):
            if clients_state[i] == 0 and temp[i] >= state_0[1] and temp[i] < state_1[1]:
                clients_state[i] = 1
            elif clients_state[i] == 1 and temp[i] >= state_1[1]:
                clients_state[i] = 1
            elif clients_state[i] == 0 and temp[i] <= state_0[2] and temp[i] > state_1[2]:
                clients_state[i] = 0
            else:
                clients_state[i] = 0

        # print(clients_state)


def clients_indexing(clients, transition_prob, clients_power):
    state_0 = [0.9449, 0.0087, 0.9913]
    state_1 = [0.0551, 0.8509, 0.1491]
    user_indices = []
    for i in range(len(clients)):
        if transition_prob[i] in state_0:
            v_i_t = -(state_1[1]/len(clients)) - \
                (((state_1[0]*clients_power[i])/100))
            user_indices.append(v_i_t)
            # print(
            #     f'client {clients[i]}, is in state_0 with tratransition probability of {transition_prob[i]}')
        elif transition_prob[i] in state_1:
            v_i_t = -(state_0[1]/len(clients)) - \
                (((state_0[0]*clients_power[i])/100))
            user_indices.append(v_i_t)
    print('Indices are', user_indices)
    # this prints the top k values
    successfull_users = heapq.nlargest(top_k, user_indices)
    print(f'the top {top_k} users who can transmit are: {successfull_users}')
    # this prints the top k indices
    user_indices = np.argsort(user_indices)
    successfull_users = user_indices[-top_k:]
    print(f'the top {top_k} users who can transmit are: {successfull_users}')
    # print(
    #     f'client {clients[i]}, is in state_1 with tratransition probability of {transition_prob[i]}')


state_0 = [0.9449, 0.0087, 0.9913]
state_1 = [0.0551, 0.8509, 0.1491]
global_epochs = 2
top_k = 2
clients_state = []
num_clients = int(input('Please Enter the number of clients: '))
clients = clients_pool(num_clients)
clients_power = power(clients)
for i in range(global_epochs):
    print(f'Prev state {clients_state}')
    transition_prob = wireless_channel_transition_probability(clients)
    print(f'current state {clients_state}')
    print('\n')
# clients_indexing(clients, transition_prob, clients_power)
# print((clients))
# print(clients_power)
# print(transition_prob)
