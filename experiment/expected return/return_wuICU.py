import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    DQN_45 = np.load('ICU/DQN-45.npy', allow_pickle=True)
    DQN_37 = np.load('ICU/DQN-37.npy', allow_pickle=True)
    DDQN_45 = np.load('ICU/DDQN-45.npy', allow_pickle=True)
    DDQN_37 = np.load('ICU/DDQN-37.npy', allow_pickle=True)
    D3QN_45 = np.load('ICU/D3QN-45.npy', allow_pickle=True)
    D3QN_37 = np.load('ICU/D3QN-37.npy', allow_pickle=True)
    ID3QN = np.load('ICU/ID3QN.npy', allow_pickle=True)
    ID3QNE = np.load('ICU/ID3QNE.npy', allow_pickle=True)
    x_length_list = list(range(len(DQN_45)))
    ax = plt.figure()
    plt.plot(x_length_list, DQN_45, '-.', label='DQN_45', markersize=3)
    plt.plot(x_length_list, DQN_37, 'v', label='DQN_37', markersize=1)
    plt.plot(x_length_list, DDQN_45, '.', label='DDQN_45', markersize=3)
    plt.plot(x_length_list, DDQN_37, '*', label='DDQN_37', markersize=3)
    plt.plot(x_length_list, D3QN_45, '<', label='D3QN_45', markersize=3)
    plt.plot(x_length_list, D3QN_37, '+', label='D3QN_37', markersize=3)
    plt.plot(x_length_list, ID3QN, '--', color='y', label='ID3QN')
    plt.plot(x_length_list, ID3QNE, '-', color='r', label='ID3QNE')
    font1 = {'family': 'Arial', 'weight': 'normal'}
    plt.legend(prop=font1)
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontname="Arial")
    plt.yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24], fontname="Arial")
    plt.xlabel("Epochs", font1)
    plt.ylabel("Expected return", font1)
    plt.show()
