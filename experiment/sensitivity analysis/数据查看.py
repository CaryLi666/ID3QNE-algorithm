#有用  做的一个数据集分割  timestep

import time
import pickle   #打开pkl包
import numpy as np

from scipy.stats import zscore, rankdata
def my_zscore(x):
    return zscore(x,ddof=1),np.mean(x,axis=0),np.std(x,axis=0,ddof=1)
with open('1小时.pkl', 'rb') as file:
    MIMICtable = pickle.load(file)  # [278751 rows x 59 columns]的数据
reformat5 = MIMICtable.values.copy()




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
MIMICtablecopy=MIMICtable
stayID=reformat5[:,1]
blocID=reformat5[:,1]
bloc=1
cout=0
timestep=8
for i in range(0,len(blocID)-timestep,timestep):
    MIMICtablecopy.iloc[cout,1:]=MIMICtable.iloc[i,1:]

    MIMICtablecopy.iloc[cout, 0] = bloc
    cout = cout + 1
    bloc=bloc+1
    if stayID[i]!=stayID[i+timestep]:
        bloc = 1
MIMICtablecopydown=MIMICtablecopy.iloc[0:cout,:]




