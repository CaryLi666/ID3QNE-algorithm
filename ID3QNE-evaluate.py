import torch
import numpy as np
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")
device = 'cpu'


def do_eval(model, batchs, batch_size=128):
    (state, next_state, action, next_action, reward, done) = batchs
    Q_value = model.Q(state)
    agent_actions = torch.argmax(Q_value, dim=1)
    phy_actions = action
    Q_value_pro1 = F.softmax(Q_value)
    Q_value_pro_ind = torch.argmax(Q_value_pro1, dim=1)
    Q_value_pro_ind1 = range(len(Q_value_pro_ind))
    Q_value_pro = Q_value_pro1[Q_value_pro_ind1, Q_value_pro_ind]
    return Q_value, agent_actions, phy_actions, Q_value_pro


def do_test(model, Xtest, actionbloctest, bloctest, Y90, SOFA, reward_value, beat):
    bloc_max = max(bloctest)  # 最大才20个阶段
    r = np.array([reward_value, -reward_value]).reshape(1, -1)
    r2 = r * (2 * (1 - Y90.reshape(-1, 1)) - 1)
    R3 = r2[:, 0]
    R4 = (R3 + reward_value) / (2 * reward_value)
    RNNstate = Xtest
    print('####  生成测试集轨迹  ####')
    statesize = int(RNNstate.shape[1])
    states = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), statesize))
    actions = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), 1), dtype=int)
    next_actions = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), 1), dtype=int)
    rewards = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), 1))
    next_states = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), statesize))
    done_flags = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), 1))
    bloc_num = np.zeros((np.floor(RNNstate.shape[0]).astype('int64'), 1))
    blocnum1 = 1
    c = 0

    bloc_num_reward = 0
    for i in range(RNNstate.shape[0] - 1):  # 每一行循环
        states[c] = RNNstate[i, :]
        actions[c] = actionbloctest[i]
        bloc_num[c] = blocnum1
        if (bloctest[i + 1] == 1):  # end of trace for this patient
            next_states1 = np.zeros(statesize)
            next_actions1 = -1
            done_flags1 = 1
            blocnum1 = blocnum1 + 1
            bloc_num_reward += 1
            reward1 = -beat[0] * (SOFA[i]) + R3[i]
            bloc_num_reward = 0
        else:
            next_states1 = RNNstate[i + 1, :]
            next_actions1 = actionbloctest[i + 1]
            done_flags1 = 0
            blocnum1 = blocnum1
            reward1 = - beat[1] * (SOFA[i + 1] - SOFA[i])
            bloc_num_reward += 1
        next_states[c] = next_states1
        next_actions[c] = next_actions1
        rewards[c] = reward1
        done_flags[c] = done_flags1
        c = c + 1  # 从0开始
    states[c] = RNNstate[c, :]
    actions[c] = actionbloctest[c]
    bloc_num[c] = blocnum1

    next_states1 = np.zeros(statesize)
    next_actions1 = -1
    done_flags1 = 1
    blocnum1 = blocnum1 + 1
    bloc_num_reward += 1
    reward1 = -beat[0] * (SOFA[c]) + R3[c]

    bloc_num_reward = 0
    next_states[c] = next_states1
    next_actions[c] = next_actions1
    rewards[c] = reward1
    done_flags[c] = done_flags1
    c = c + 1  # 从0开始
    bloc_num = bloc_num[:c, :]
    states = states[: c, :]
    next_states = next_states[: c, :]
    actions = actions[: c, :]
    next_actions = next_actions[: c, :]
    rewards = rewards[: c, :]
    done_flags = done_flags[: c, :]

    bloc_num = np.squeeze(bloc_num)
    actions = np.squeeze(actions)
    rewards = np.squeeze(rewards)
    done_flags = np.squeeze(done_flags)

    # numpy形式转化为tensor形式
    batch_size = states.shape[0]
    state = torch.FloatTensor(states).to(device)
    next_state = torch.FloatTensor(next_states).to(device)
    action = torch.LongTensor(actions).to(device)
    next_action = torch.LongTensor(next_actions).to(device)
    reward = torch.FloatTensor(rewards).to(device)
    done = torch.FloatTensor(done_flags).to(device)
    batchs = (state, next_state, action, next_action, reward, done, bloc_num)

    rec_phys_q = []
    rec_agent_q = []
    rec_agent_q_pro = []
    rec_phys_a = []
    rec_agent_a = []
    rec_sur = []
    rec_reward_user = []
    batch_s = 128
    uids = np.unique(bloc_num)
    num_batch = uids.shape[0] // batch_s  # 分批次
    for batch_idx in range(num_batch + 1):
        batch_uids = uids[batch_idx * batch_s: (batch_idx + 1) * batch_s]
        batch_user = np.isin(bloc_num, batch_uids)
        state_user = state[batch_user, :]
        next_state_user = next_state[batch_user, :]
        action_user = action[batch_user]
        next_action_user = next_action[batch_user]
        reward_user = reward[batch_user]
        done_user = done[batch_user]
        sur_Y90 = Y90[batch_user]

        batch = (state_user, next_state_user, action_user, next_action_user, reward_user, done_user)
        q_output, agent_actions, phys_actions, Q_value_pro = do_eval(model, batch)

        q_output_len = range(len(q_output))
        agent_q = q_output[q_output_len, agent_actions]
        phys_q = q_output[q_output_len, phys_actions]

        rec_agent_q.extend(agent_q.detach().numpy())
        rec_agent_q_pro.extend(Q_value_pro.detach().numpy())

        rec_phys_q.extend(phys_q.detach().numpy())
        rec_agent_a.extend(agent_actions.detach().numpy())
        rec_phys_a.extend(phys_actions.detach().numpy())
        rec_sur.extend(sur_Y90)
        rec_reward_user.extend(reward_user.detach().numpy())

    np.save('Q值/shencunlv.npy', rec_sur)
    np.save('Q值/agent_bQ.npy', rec_agent_q)
    np.save('Q值/phys_bQ.npy', rec_phys_q)
    np.save('Q值/reward.npy', rec_reward_user)

    np.save('Q值/agent_actionsb.npy', rec_agent_a)
    np.save('Q值/phys_actionsb.npy', rec_phys_a)

    np.save('Q值/rec_agent_q_pro.npy', rec_agent_q_pro)
