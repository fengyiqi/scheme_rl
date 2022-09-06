import os
import json
import gym
from gym import spaces
import numpy as np
import torch
import xml.etree.ElementTree as ET
from mytools.objective.sodshocktube import SodShockTube
from mytools.postprocessing.smoothness import symmetry, periodic, do_weno5_si
from mytools.stencils.teno5 import TENO5
import matplotlib.pyplot as plt
import glob
from collections import deque

import torch.nn as nn
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
device

def compute_td_loss(agent, target_network, states, actions, rewards, next_states, done_flags,
                    gamma=0.99, device=device):

    # convert numpy array to torch tensors
    states = torch.tensor(states, dtype=torch.float64)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float64)
    next_states = torch.tensor(next_states, dtype=torch.float64)
    done_flags = torch.tensor(done_flags.astype('float32'), dtype=torch.float)

    # get q-values for all actions in current states
    # use agent network
#     print(states)
    predicted_qvalues = agent(states)

    # compute q-values for all actions in next states
    # use target network
    predicted_next_qvalues= target_network(next_states)
    
    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(
        len(actions)), actions]

    # compute Qmax(next_states, actions) using predicted next q-values
    next_state_values,_ = torch.max(predicted_next_qvalues, dim=1)

    # compute "target q-values" 
    target_qvalues_for_actions = rewards + gamma * next_state_values * (1-done_flags)

    # mean squared error loss to minimize
#     loss = torch.mean((predicted_qvalues_for_actions -
#                        target_qvalues_for_actions.detach()) ** 2)
    criterion = nn.SmoothL1Loss()
    loss = criterion(predicted_qvalues_for_actions, target_qvalues_for_actions)

    return loss


def compute_td_loss_priority_replay(agent, target_network, replay_buffer,
                                    states, actions, rewards, next_states, done_flags, weights, buffer_idxs,
                                    gamma=0.99, device=device):

    # convert numpy array to torch tensors
    states = torch.tensor(states, device=device, dtype=torch.float64)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float64)
    next_states = torch.tensor(next_states, device=device, dtype=torch.float64)
    done_flags = torch.tensor(done_flags.astype('float32'),device=device,dtype=torch.float)
    weights = torch.tensor(weights, device=device, dtype=torch.float)

    # get q-values for all actions in current states
    # use agent network
    predicted_qvalues = agent(states)

    # compute q-values for all actions in next states
    # use target network
    predicted_next_qvalues = target_network(next_states)
    
    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(
        len(actions)), actions]

    # compute Qmax(next_states, actions) using predicted next q-values
    next_state_values,_ = torch.max(predicted_next_qvalues, dim=1)

    # compute "target q-values" 
    target_qvalues_for_actions = rewards + gamma * next_state_values * (1-done_flags)
    
    #compute each sample TD error
    criterion = nn.SmoothL1Loss()
    
    loss = criterion(predicted_qvalues_for_actions, target_qvalues_for_actions.detach()) * weights
    
    # mean squared error loss to minimize
    loss = loss.mean()
    
    # calculate new priorities and update buffer
    with torch.no_grad():
        new_priorities = predicted_qvalues_for_actions.detach() - target_qvalues_for_actions.detach()
        new_priorities = np.absolute(new_priorities.detach().cpu().numpy())
        replay_buffer.update_priorities(buffer_idxs, new_priorities)
        
    return loss


def td_loss_ddqn(agent, target_network, states, actions, rewards, next_states, done_flags,
                    gamma=0.99, device=device):

    # convert numpy array to torch tensors
    states = torch.tensor(states, device=device, dtype=torch.float64)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)
    next_states = torch.tensor(next_states, device=device, dtype=torch.float64)
    done_flags = torch.tensor(done_flags.astype('float32'),device=device,dtype=torch.float)

    # get q-values for all actions in current states
    # use agent network
    q_s = agent(states)

    # select q-values for chosen actions
    q_s_a = q_s[range(
        len(actions)), actions]

    # compute q-values for all actions in next states
    # use agent network (online network)
    q_s1 = agent(next_states).detach()

    # compute Q argmax(next_states, actions) using predicted next q-values
    _,a1max = torch.max(q_s1, dim=1)

    #use target network to calclaute the q value for best action chosen above
    q_s1_target = target_network(next_states)

    q_s1_a1max = q_s1_target[range(len(a1max)), a1max]

    # compute "target q-values" 
    target_q = rewards + gamma * q_s1_a1max * (1-done_flags)

    # mean squared error loss to minimize
#     loss = torch.mean((q_s_a - target_q).pow(2))
    criterion = nn.SmoothL1Loss()
    loss = criterion(q_s_a, target_q)

    return loss