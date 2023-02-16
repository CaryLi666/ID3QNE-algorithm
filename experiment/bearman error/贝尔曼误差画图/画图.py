import numpy as np
import matplotlib.pyplot as plt

bellman_error_DQN=np.load("record_bellman_error_DQN.npy",allow_pickle=True)
bellman_error_D3QN=np.load("record_bellman_error_D3QN.npy",allow_pickle=True)
bellman_error_WD3QN=np.load("record_bellman_error_WD3QN.npy",allow_pickle=True)

x_length_list = list(range(len(bellman_error_DQN)))
plt.figure()

plt.plot(x_length_list, bellman_error_DQN,'--',label= 'Dueling DQN')
plt.plot(x_length_list, bellman_error_D3QN,'-.',label= 'D3QN')
plt.plot(x_length_list, bellman_error_WD3QN,label= 'WD3QN',color='red')

font1 = {'family': 'Arial', 'weight': 'normal','size':12}

plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontname="Arial", )

plt.legend(prop=font1)
plt.xlabel("Epochs", font1)
plt.ylabel("Bellman error", font1)
#
plt.show()