import time
import pickle   #打开pkl包
import numpy as np
from ismember import ismember
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import copy
import os
gamma = 0.99
device='cpu'



class DistributionalDQN(nn.Module):
    def __init__(self, state_dim, n_actions, N_ATOMS):
        super(DistributionalDQN, self).__init__()

        self.input_layer = nn.Linear(state_dim, 128)
        self.hiddens = nn.ModuleList([nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()) for _ in range(7)])

        self.out = nn.Linear(128, n_actions)

    def forward(self, state):
        batch_size = state.size()[0]
        out = self.input_layer(state)
        for layer in self.hiddens:
            out = layer(out)

        out = self.out(out)
        return out


class dist_DQN(object):
    def __init__(self,
                 state_dim=37,
                 num_actions=25,
                 v_max=20,
                 v_min=-20,
                 device='cpu',
                 gamma=0.999,
                 tau=0.005,
                 n_atoms=51
                 ):
        self.device=device

        self.Q = DistributionalDQN(state_dim, num_actions, n_atoms).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.000005)  #0.00001
        self.tau = tau
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max

        self.num_actions = num_actions
        self.atoms = n_atoms

    def train(self,batchs,epoch):

        (state, next_state, action, next_action, reward, done,bloc_num)=batchs
        states_num = state.shape[0]
        batch_s = 128
        uids = np.unique(bloc_num)
        num_batch = uids.shape[0] // batch_s  # 分批次
        record_loss_num = 0
        record_loss = []

        sum_q_loss = 0
        Batch = 0
        for batch_idx in range(num_batch):
            batch_uids = uids[batch_idx * batch_s: (batch_idx + 1) * batch_s]
            batch_user = np.isin(bloc_num, batch_uids)
            state_user = state[batch_user, :]
            next_state_user = next_state[batch_user, :]
            action_user = action[batch_user]
            next_action_user = next_action[batch_user]
            reward_user = reward[batch_user]
            done_user = done[batch_user]
            batch = (state_user, next_state_user, action_user, next_action_user,reward_user, done_user)
            loss = self.compute_loss(batch)
            sum_q_loss += loss.item()
            if Batch % 25 == 0:
                print('Epoch :', epoch, 'Batch :', Batch, 'Average Loss :', sum_q_loss / (Batch + 1))
                record_loss1 = sum_q_loss / (Batch + 1)
                record_loss.append(record_loss1)

            self.optimizer.zero_grad()  #梯度清零
            loss.backward()             #反向传播
            self.optimizer.step()       #更新

            #更新Q_target网络参数
            if num_batch%50==0:
                self.polyak_target_update()

            Batch += 1

        return record_loss


    def polyak_target_update(self):    #更新网络
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def compute_loss(self, batch):
        state, next_state, action, next_action, reward, done = batch
        batch_size = state.shape[0]
        range_batch = torch.arange(batch_size).long().to(device)

        #利用神经网络输出动作
        log_Q_dist_prediction = self.Q(state)
        log_Q_dist_prediction1 = log_Q_dist_prediction[range_batch, action]   #  1810  一个一维的数

        with torch.no_grad():
            Q_dist_target= self.Q_target(next_state)

        #求最大值
        a_star = torch.argmax(Q_dist_target, dim=1)

        log_Q_experience = Q_dist_target[range_batch, next_action.squeeze(1)]

        #最大的Q值
        Q_dist_star = Q_dist_target[range_batch, a_star]

        # 更新 targetQ Q值=============================
        end_multiplier = 1 - done
        eplion=0.2

        targetQ = reward + (gamma * end_multiplier* (Q_dist_star+eplion*(log_Q_experience-Q_dist_star)))

        td_error = torch.square(targetQ - log_Q_dist_prediction1)
        old_loss = torch.mean(td_error)

        return old_loss

    def get_action(self, state):
        with torch.no_grad():
            batch_size = state.shape[0]
            Q_dist= self.Q(state)
            a_star = torch.argmax(Q_dist, dim=1)
            return a_star
