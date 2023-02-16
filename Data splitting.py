import time
import pickle
import numpy as np

if __name__ == '__main__':
    # 计算时间
    start = time.perf_counter()
    # 载入总数据
    with open('患者总数据.pkl', 'rb') as file:
        MIMICtable = pickle.load(file)

    #####################模型参数设置##############################
    ncv = 10  # nr of crossvalidation runs (each is 80% training / 20% test)交叉验证运行的Nr(每次为80%训练/ 20%测试)
    icustayidlist = MIMICtable['icustayid']
    icuuniqueids = np.unique(icustayidlist)
    N = icuuniqueids.size
    grp = np.floor(ncv * np.random.rand(N,
                                        1) + 1)
    crossval1 = 1  # 利用等于1和不等于1
    crossval2 = 2  # 利用等于1和不等于1
    trainidx = icuuniqueids[np.where(grp > crossval2)[0]]
    validationidx = icuuniqueids[np.where(grp == crossval1)[0]]
    testidx = icuuniqueids[np.where(grp == crossval2)[0]]
    train = np.isin(icustayidlist, trainidx)
    validation = np.isin(icustayidlist, validationidx)
    test = np.isin(icustayidlist, testidx)
    # 保存到文件
    np.save('数据集/train.npy', train)
    np.save('数据集/validation.npy', validation)
    np.save('数据集/test.npy', test)

