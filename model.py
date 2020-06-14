import copy
import time
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from dqn import make_dqn
from interface import Interface


BATCHSIZE = 64
ITERATIONS = 2500
MAX_MEMORY = 1000
GAMMA = 0.95
INIT_EPSILON = 1
MIN_EPSILON = 0.01


memory = []
DQN_p = make_dqn(2, (4, 50, 100))
DQN_t = make_dqn(2, (4, 50, 100))
DQN_p.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00005, 
        beta_1=0.9, beta_2=0.999, 
        epsilon=1e-07
        ), 
    loss='mse'
    )
DQN_t.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-07
        ), 
    loss='mse'
    )
interface = Interface()
time.sleep(1)


def add_step(memory, state_t, action_t, reward_t, q_values, state_t1, terminal):
    """ Add the features of one step to the specified replay memory """
    memory.append((state_t, action_t, reward_t, q_values, state_t1, terminal))
    if len(memory) > MAX_MEMORY:
        del memory[0]

def step(interface, pred_dqn, memory, state_t, score_t, epsilon):
    """ 
    Perform one action (exploring or epsilon-greedy) and return subsequent state
    Stepwise reward is obtained by subtracting the previous step score_p from the current score_t
    """
    q_values = pred_dqn.predict(state_t[np.newaxis, :])
    print(np.round(q_values[0], 4))
    if random.random() < epsilon:
        action_t = np.random.choice([0, 1])
        print("Exploration - action:", action_t)
    else:
        action_t = np.argmax(q_values)
    state, score_t1, terminal = interface.action(action_t)
    state_t1 = np.append(state, state_t[1:], axis=0)
    reward_t = score_t1 - score_t
    add_step(memory, state_t, action_t, reward_t, q_values, state_t1, terminal)
    return state_t1, score_t1, terminal

def play_episode(interface, pred_dqn, memory, epsilon):
    """ Restart and play through one dino run until the dino crashes"""
    state_t = np.zeros((4, 50, 100))
    state_t[0] = interface.re_start()
    score_t = 0
    time.sleep(1)
    while True:
        state_t, score_t, terminal = step(interface, pred_dqn, memory, state_t, score_t, epsilon)
        if terminal:
            break

def update(pred_dqn, target_dqn, batch):
    """ Update the DQNN based on samples from the experience memory """
    states = np.zeros((BATCHSIZE, 4, 50, 100))
    targets = np.zeros((BATCHSIZE, 2))
    for i, step in enumerate(batch):
        state_t = step[0]
        action_t = step[1]
        reward_t = step[2] 
        q_values = step[3]
        state_t1 = step[4]
        terminal = step[5]
        states[i] = state_t
        targets[i] = q_values
        if terminal:
            targets[i, action_t] = -100
        else:
            action_t1 = np.argmax(pred_dqn.predict(state_t1[np.newaxis, :]))
            q_values_t1 = target_dqn.predict(state_t1[np.newaxis, :])[0]
            targets[i, action_t] = 1 + GAMMA * q_values_t1[action_t1]
    return pred_dqn.train_on_batch(states, targets)


score_history = []
loss_history = []
epsilon = INIT_EPSILON


for i in range(ITERATIONS):
    print("Episode " +  str(i))
    play_episode(interface, DQN_p, memory, epsilon)
    score_history.append(interface.get_score())
    epsilon = np.max([MIN_EPSILON, epsilon - epsilon / 100])
    print("Epsilon:", epsilon)
    if len(memory) < 100:
        continue
    batch = random.sample(memory, BATCHSIZE)
    loss = update(DQN_p, DQN_t, batch)
    print("Loss:", loss)
    loss_history.append(loss)
    if (i + 1) % 10 == 0:
        print('Update Target NN')
        DQN_t.set_weights(DQN_p.get_weights())
    avg = np.sum(score_history[-10:]) / 10
    print("Rolling 10-Runs average:", avg)
    if i % 500 == 0:
        interface.close()
        time.sleep(5)
        interface = Interface()
        time.sleep(5)
    