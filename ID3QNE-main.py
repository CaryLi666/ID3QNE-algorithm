import time
import pickle
import numpy as np

import torch.optim
from deepQnet import Dist_DQN
from evaluate import do_eval, do_test
import matplotlib.pyplot as plt

device = 'cpu'

# ===========================小函数===============================
from scipy.stats import zscore, rankdata


def my_zscore(x):
    return zscore(x, ddof=1), np.mean(x, axis=0), np.std(x, axis=0, ddof=1)


if __name__ == '__main__':
    start = time.perf_counter()
    with open('患者总数据.pkl', 'rb') as file:
        MIMICtable = pickle.load(file)

    #####################模型参数设置##############################
    num_epoch = 101  # 训练循环次数
    gamma = 0.99
    beat1 = 0
    beat2 = 0.6
    beta3 = 0.3
    ncv = 5  # nr of crossvalidation runs (each is 80% training / 20% test)交叉验证运行的Nr(每次为80%训练/ 20%测试)
    nra = 5
    lr = 1e-5
    reward_value = 24
    beta = [beat1, beat2, beta3]
    icustayidlist = MIMICtable['icustayid']
    icuuniqueids = np.unique(icustayidlist)  # list of unique icustayids from MIMIC唯一的icustayid列表
    reformat5 = MIMICtable.values.copy()
    print('####  生成状态  ####')

    # -----------------------筛选后的特征=37个--------------------------------
    colnorm = ['SOFA', 'age', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'Temp_C',
               'Sodium', 'Chloride', 'Glucose', 'Calcium', 'Hb', 'WBC_count', 'Platelets_count',
               'PTT', 'PT', 'Arterial_pH', 'paO2', 'paCO2', 'HCO3', 'Arterial_lactate', 'Shock_Index',
               'PaO2_FiO2', 'cumulated_balance', 'CO2_mEqL', 'Ionised_Ca']
    ##8个指标
    collog = ['SpO2', 'BUN', 'Creatinine', 'SGOT', 'Total_bili', 'INR', 'input_total', 'output_total']

    colnorm = np.where(np.isin(MIMICtable.columns, colnorm))[0]
    collog = np.where(np.isin(MIMICtable.columns, collog))[0]

    scaleMIMIC = np.concatenate([zscore(reformat5[:, colnorm], ddof=1),
                                 zscore(np.log(0.1 + reformat5[:, collog]), ddof=1)], axis=1)

    train = np.load('数据集/train.npy')
    validat = np.load('数据集/validation.npy')
    test = np.load('数据集/test.npy')

    Xvalidat = scaleMIMIC[validat, :]
    blocsvalidat = reformat5[validat, 0]
    ptidvalidat = reformat5[validat, 1]

    Xtrain = scaleMIMIC[train, :]
    Xtest = scaleMIMIC[test, :]
    blocstrain = reformat5[train, 0]  # 序列号
    bloctest = reformat5[test, 0]
    ptidtrain = reformat5[train, 1]  # 患者编号
    ptidtest = reformat5[test, 1]

    # *************************
    RNNstate = Xtrain  # ***

    print('####  生成动作  ####')
    nact = nra * nra  # 5*5=25
    iol = MIMICtable.columns.get_loc('input_4hourly')  # 输入的列
    vcl = MIMICtable.columns.get_loc('max_dose_vaso')  # 最大使用加压药量的列

    a = reformat5[:, iol].copy()  # IV fluid  静脉输液复苏
    a = rankdata(a[a > 0]) / a[a > 0].shape[0]  # excludes zero fluid (will be action 1)不包括零液体 将是动作1
    iof = np.floor((a + 0.2499999999) * 4)  # converts iv volume in 4 actions 在4个动作中转换静脉输液量
    a = reformat5[:, iol].copy()
    a = np.where(a > 0)[0]  # location of non-zero fluid in big matrix
    io = np.ones((reformat5.shape[0], 1))  # array of ones, by default
    io[a] = (iof + 1).reshape(-1, 1)  # where more than zero fluid given: save actual action
    io = io.ravel()  # 两者的本质都是想把多维的数组降为1维  注射有5个动作，已经通过秩进行判断

    vc = reformat5[:, vcl].copy()
    vcr = rankdata(vc[vc != 0]) / vc[vc != 0].size
    vcr = np.floor((vcr + 0.249999999999) * 4)  # converts to 4 bins
    vcr[vcr == 0] = 1
    vc[vc != 0] = vcr + 1
    vc[vc == 0] = 1

    ma1 = np.array(
        [np.median(reformat5[io == 1, iol]), np.median(reformat5[io == 2, iol]), np.median(reformat5[io == 3, iol]),
         np.median(reformat5[io == 4, iol]), np.median(reformat5[io == 5, iol])])  # median dose of drug in all bins
    ma2 = np.array(
        [np.median(reformat5[vc == 1, vcl]), np.median(reformat5[vc == 2, vcl]), np.median(reformat5[vc == 3, vcl]),
         np.median(reformat5[vc == 4, vcl]), np.median(reformat5[vc == 5, vcl])])

    med = np.concatenate([io.reshape(-1, 1), vc.reshape(-1, 1)], axis=1)
    uniqueValues, actionbloc = np.unique(med, axis=0, return_inverse=True)

    actionbloctrain = actionbloc[train]  # ***
    actionblocvalidat = actionbloc[validat]  # ***
    actionbloctest = actionbloc[test]

    ma2Values = ma2[uniqueValues[:, 1].astype('int64') - 1].reshape(-1, 1)
    ma1Values = ma1[uniqueValues[:, 0].astype('int64') - 1].reshape(-1, 1)
    uniqueValuesdose = np.concatenate([ma2Values, ma1Values], axis=1)  # median dose of each bin for all 25 actions

    # =================奖励============================
    print('####  生成奖励  ####')
    outcome = 9
    Y90 = reformat5[train, outcome]
    r = np.array([reward_value, -reward_value]).reshape(1, -1)
    r2 = r * (2 * (1 - Y90.reshape(-1, 1)) - 1)

    # -----奖励函数准备-----------------------------
    SOFA = reformat5[train, 57]  # ***
    R3 = r2[:, 0]
    R4 = (R3 + reward_value) / (2 * reward_value)
    c = 0
    bloc_max = max(blocstrain)

    # ================构建状态&&下一个状态序列表  生成策略轨迹=================================
    print(RNNstate.shape[0])

    print('####  生成轨迹  ####')
    statesize = int(RNNstate.shape[1])
    states = np.zeros((np.floor(RNNstate.shape[0] * 1.2).astype('int64'), statesize))
    actions = np.zeros((np.floor(RNNstate.shape[0] * 1.2).astype('int64'), 1), dtype=int)
    next_actions = np.zeros((np.floor(RNNstate.shape[0] * 1.2).astype('int64'), 1), dtype=int)
    rewards = np.zeros((np.floor(RNNstate.shape[0] * 1.2).astype('int64'), 1))
    next_states = np.zeros((np.floor(RNNstate.shape[0] * 1.2).astype('int64'), statesize))
    done_flags = np.zeros((np.floor(RNNstate.shape[0] * 1.2).astype('int64'), 1))
    bloc_num = np.zeros((np.floor(RNNstate.shape[0] * 1.2).astype('int64'), 1))
    blocnum1 = 1

    bloc_num_reward = 0
    for i in range(RNNstate.shape[0] - 1):  # 每一行循环
        states[c] = RNNstate[i, :]
        actions[c] = actionbloctrain[i]
        bloc_num[c] = blocnum1
        if (blocstrain[i + 1] == 1):  # end of trace for this patient
            next_states1 = np.zeros(statesize)
            next_actions1 = -1
            done_flags1 = 1
            blocnum1 = blocnum1 + 1
            bloc_num_reward += 1
            reward1 = -beat1 * (SOFA[i]) + R3[i]
            bloc_num_reward = 0
        else:
            next_states1 = RNNstate[i + 1, :]
            next_actions1 = actionbloctrain[i + 1]
            done_flags1 = 0
            blocnum1 = blocnum1
            reward1 = - beat2 * (SOFA[i + 1] - SOFA[i])
            bloc_num_reward += 1
        next_states[c] = next_states1
        next_actions[c] = next_actions1
        rewards[c] = reward1
        done_flags[c] = done_flags1
        c = c + 1
    states[c] = RNNstate[c, :]
    actions[c] = actionbloctrain[c]
    bloc_num[c] = blocnum1

    next_states1 = np.zeros(statesize)
    next_actions1 = -1
    done_flags1 = 1
    blocnum1 = blocnum1 + 1
    bloc_num_reward += 1
    reward1 = -beat1 * (SOFA[c]) + R3[c]

    bloc_num_reward = 0
    next_states[c] = next_states1
    next_actions[c] = next_actions1
    rewards[c] = reward1
    done_flags[c] = done_flags1
    c = c + 1

    bloc_num[c] = blocnum1
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
    batch_size = states.shape[0]
    state = torch.FloatTensor(states).to(device)
    next_state = torch.FloatTensor(next_states).to(device)
    action = torch.LongTensor(actions).to(device)
    next_action = torch.LongTensor(next_actions).to(device)
    reward = torch.FloatTensor(rewards).to(device)
    done = torch.FloatTensor(done_flags).to(device)
    SOFAS = torch.LongTensor(SOFA).to(device)
    batchs = (state, next_state, action, next_action, reward, done, bloc_num, SOFAS)
    # =================训练模型，主循环==================
    Y90_validat = reformat5[validat, outcome]
    SOFA_validat = reformat5[validat, 57]
    model = Dist_DQN()  # 实例化网络模型
    record_loss_z = []
    record_phys_q = []
    record_agent_q = []
    for epoch in range(num_epoch):
        record = model.train(batchs, epoch)
        record_loss_z.append(record)
        if epoch % 50 == 0:
            torch.save({
                'Q_state_dict': model.Q.state_dict(),
                'Q_target_state_dict': model.Q_target.state_dict(),
            }, 'model\dist_noW{}.pt'.format(epoch))
        record_a = np.array(record_loss_z)
        record_b = np.sum(record_a, axis=1)
        # -------------验证集，评估------------------------------
        batch_s =ptidvalidat
        uids = np.unique(bloc_num)
        batch_uids = range(1, batch_s)
        batch_user = np.isin(bloc_num, batch_uids)
        state_user = state[batch_user, :]
        next_state_user = next_state[batch_user, :]
        action_user = action[batch_user]
        next_action_user = next_action[batch_user]
        reward_user = reward[batch_user]
        done_user = done[batch_user]
        batch = (state_user, next_state_user, action_user, next_action_user, reward_user, done_user)

        q_output, agent_actions, phys_actions, Q_value_pro = do_eval(model, batch)

        q_output_len = range(len(q_output))
        agent_q = q_output[:, agent_actions]
        phys_q = q_output[:, phys_actions]

        print('mean agent Q:', torch.mean(agent_q))
        print('mean phys Q:', torch.mean(phys_q))
        record_phys_q.append(torch.mean(phys_q))
        record_agent_q.append(torch.mean(agent_q))

        print('agent_actions：', agent_actions)
        print('phys_actions：', phys_actions)

    # ===========画图=============================
    x_length_list = list(range(len(record_b)))
    plt.figure()
    plt.title('Training')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x_length_list, record_b)
    np.save('验证集/loss.npy', record_b)
    agent_length_list = list(range(len(record_agent_q)))
    plt.figure()
    plt.title('Training')
    plt.xlabel("epoch")
    plt.ylabel("mean Q value")
    plt.plot(agent_length_list, record_agent_q, label='record_agent_q ')
    np.save('验证集/mean_agent_q.npy', record_agent_q)
    phys_length_list = list(range(len(record_phys_q)))
    np.save('验证集/mean_phys_q.npy', record_phys_q)

    # =================测试集，评估test set================================================================
    Y90_test = reformat5[test, outcome]
    SOFA_test = reformat5[test, 57]
    do_test(model, Xtest, actionbloctest, bloctest, Y90_test, SOFA_test, reward_value, beta)

    elapsed = (time.perf_counter() - start)
    print("Time used:", elapsed)
    plt.show()
