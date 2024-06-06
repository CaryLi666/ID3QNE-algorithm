#灵敏度分析中的  loss 图
import pickle   #打开pkl包
import numpy as np
import matplotlib.pyplot as plt


loss_8=np.load('第一次数据保存全/8小时验证集/loss.npy')
loss_6=np.load('第一次数据保存全/6小时验证集/loss.npy')
loss_4=np.load('第一次数据保存全/4小时验证集/loss.npy')
loss_2=np.load('第一次数据保存全/2小时验证集/loss.npy')
loss_1=np.load('第一次数据保存全/1小时验证集/loss.npy')

plt.figure()
# plt.title('Training')
plt.plot(loss_length,loss_1,'-o',label= '1 hour',markersize=3,color='burlywood')
plt.plot(loss_length,loss_2,'-v',label= '2 hours',markersize=3,color='plum')
plt.plot(loss_length,loss_4,'-',label= '4 hours',markersize=3,color='red')
plt.plot(loss_length,loss_6,'--',label= '6 hours',markersize=3,color='cornflowerblue')
plt.plot(loss_length,loss_8,'-.',label= '8 hours',markersize=3,color='lime')


font1 = {'family': 'Arial', 'weight': 'normal','size':12}

plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontname="Arial", )
plt.yticks([6, 8, 10, 12, 14, 16, 18, 20, 22], fontname="Arial")

plt.legend(prop=font1)
plt.xlabel("Epochs", font1)
plt.ylabel("Loss value", font1)
plt.show()

print()
