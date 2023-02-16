#灵敏度分析中的  期望回报图

import pickle   #打开pkl包
import numpy as np
import matplotlib.pyplot as plt


loss_8=np.load('第一次数据保存全/8小时验证集/mean_agent_q.npy',allow_pickle=True)
loss_6=np.load('验证集/mean_agent_q.npy',allow_pickle=True)
loss_4=np.load('4小时验证集/mean_agent_q.npy',allow_pickle=True)
loss_2=np.load('第一次数据保存全/2小时验证集/mean_agent_q.npy',allow_pickle=True)
loss_1=np.load('第一次数据保存全/1小时验证集/mean_agent_q.npy',allow_pickle=True)

loss_length = list(range(len(loss_8)))
plt.figure()


plt.plot(loss_length ,loss_8,'-o',label= '1 hour',markersize=3,color='burlywood')
plt.plot(loss_length ,loss_6,'-v',label= '2 hours',markersize=4,color='plum')
plt.plot(loss_length ,loss_4,'-',label= '4 hours',markersize=3,color='red')
plt.plot(loss_length ,loss_2,'--',label= '6 hours',markersize=3,color='cornflowerblue')
plt.plot(loss_length ,loss_1,'-.',label= '8 hours',markersize=3,color='lime')


font1 = {'family': 'Arial', 'weight': 'normal','size':12}

plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontname="Arial", )
plt.yticks([0, 5, 10, 15, 20, 25], fontname="Arial")

plt.legend(prop=font1)
plt.xlabel("Epochs", font1)
plt.ylabel("Expected return", font1)

plt.show()