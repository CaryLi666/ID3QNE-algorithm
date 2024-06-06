import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

if __name__ == '__main__':

    fig = plt.figure(figsize=(20, 4), facecolor='w')

    ax = fig.add_subplot(131, projection='3d')

    D3Q_actions = np.load('D3Q-37/agent_actionsb.npy', allow_pickle=True)
    phys_actions = np.load('phys_actionsb.npy', allow_pickle=True)

    inv_action_map = {}
    count = 0
    for i in range(5):
        for j in range(5):
            inv_action_map[count] = [i, j]
            count += 1

    phys_actions_tuple = [None for i in range(len(phys_actions))]

    for i in range(len(phys_actions)):
        phys_actions_tuple[i] = inv_action_map[phys_actions[i]]

    phys_actions_tuple = np.array(phys_actions_tuple)

    phys_actions_iv = phys_actions_tuple[:, 0]
    phys_actions_vaso = phys_actions_tuple[:, 1]
    hist, x_edges, y_edges = np.histogram2d(phys_actions_iv, phys_actions_vaso, bins=5)

    x_edges = np.arange(-0.5, 5)
    y_edges = np.arange(-0.5, 5)

    Z = hist.reshape(25)
    z_sun = sum(Z)
    height = np.zeros_like(Z)
    xx, yy = np.meshgrid(x_edges, y_edges)  # 网格化坐标
    X, Y = xx.ravel(), yy.ravel()

    width = depth = 0.6  # 柱子的长和宽

    for i in range(0, 5):
        for j in range(0, 5):
            z = hist[i][j]  # 该柱的高
            norm = plt.Normalize(0, 0.25)
            norm_values = norm(z / z_sun)
            map_vir = cm.get_cmap(name='spring')
            colors = map_vir(norm_values)
            sc1 = ax.bar3d(j, i, height, width, depth, z, color=colors)


    font1 = {'family': 'Arial', 'weight': 'normal'}

    ax.set_xlabel('VP dose', fontsize=14, fontname="Arial")
    ax.set_ylabel('IV fluid dose', fontsize=14, fontname="Arial")
    ax.set_zlabel('Action counts', fontsize=14, fontname="Arial")
    ax.set_title("Physician policy", fontsize=15, fontname="Arial")

    sm = cm.ScalarMappable(cmap=map_vir, norm=norm)  # norm设置最大最小值
    sm.set_array([])
    cb = plt.colorbar(sm)
    cb.ax.tick_params(labelsize=10)  # 设置色标刻度字体大小。
    cb.ax.set_yticklabels([0, 1500, 3000, 4500, 6000, 7500], fontname="Arial")

    ax.set_ylim(5, -0.5)
    ax.view_init(32, 43)

    # =========================第二个图===================================================================
    ax2 = fig.add_subplot(133, projection='3d')

    IDDE_actions = np.load('IDDE/Q值/agent_actionsb.npy', allow_pickle=True)

    IDDE_actions_tuple = [None for i in range(len(IDDE_actions))]
    for i in range(len(IDDE_actions)):
        IDDE_actions_tuple[i] = inv_action_map[IDDE_actions[i]]

    IDDE_actions_tuple = np.array(IDDE_actions_tuple)
    IDDE_actions_iv = IDDE_actions_tuple[:, 0]
    IDDE_actions_vaso = IDDE_actions_tuple[:, 1]
    IDDE_hist, x_edges, y_edges = np.histogram2d(IDDE_actions_iv, IDDE_actions_vaso, bins=5)

    x_edges = np.arange(-0.5, 5)
    y_edges = np.arange(-0.5, 5)

    Z = hist.reshape(25)
    z_sun = sum(Z)
    height = np.zeros_like(Z)
    xx, yy = np.meshgrid(x_edges, y_edges)  # 网格化坐标
    X, Y = xx.ravel(), yy.ravel()

    for i in range(0, 5):
        for j in range(0, 5):
            z = IDDE_hist[i][j]  # 该柱的高
            norm = plt.Normalize(0, 0.25)
            norm_values = norm(z / z_sun)
            map_vir = cm.get_cmap(name='cool')
            colors = map_vir(norm_values)
            sc2 = ax2.bar3d(j, i, height, width, depth, z, color=colors)

    ax2.set_xlabel('VP dose', fontsize=14, fontname="Arial")
    ax2.set_ylabel('IV fluid dose', fontsize=14, fontname="Arial")
    ax2.set_zlabel('Action counts', fontsize=14, fontname="Arial")
    ax2.set_title("ID3QNE policy", fontsize=15, fontname="Arial")

    sm2 = cm.ScalarMappable(cmap=map_vir, norm=norm)  # norm设置最大最小值
    sm2.set_array([])
    cb2 = plt.colorbar(sm2)
    cb2.ax.tick_params(labelsize=10)  # 设置色标刻度字体大小。
    cb2.ax.set_yticklabels([0, 1500, 3000, 4500, 6000, 7500], fontname="Arial")

    ax2.set_ylim(5, -0.5)
    ax2.view_init(32, 43)

    # ================第三张图======================================================================

    ax3 = fig.add_subplot(132, projection='3d')

    D3Q_actions_tuple = [None for i in range(len(D3Q_actions))]
    for i in range(len(D3Q_actions)):
        D3Q_actions_tuple[i] = inv_action_map[D3Q_actions[i]]

    D3Q_actions_tuple = np.array(D3Q_actions_tuple)
    D3Q_actions_iv = D3Q_actions_tuple[:, 0]
    D3Q_actions_vaso = D3Q_actions_tuple[:, 1]
    D3Q_hist, x_edges, y_edges = np.histogram2d(D3Q_actions_iv, D3Q_actions_vaso, bins=5)

    x_edges = np.arange(-0.5, 5)
    y_edges = np.arange(-0.5, 5)

    Z = hist.reshape(25)
    z_sun = sum(Z)
    height = np.zeros_like(Z)
    xx, yy = np.meshgrid(x_edges, y_edges)  # 网格化坐标
    X, Y = xx.ravel(), yy.ravel()

    for i in range(0, 5):
        for j in range(0, 5):
            z = D3Q_hist[i][j]  # 该柱的高
            norm = plt.Normalize(0, 0.25)
            norm_values = norm(z / z_sun)
            map_vir = cm.get_cmap(name='winter')
            colors = map_vir(norm_values)
            sc2 = ax3.bar3d(j, i, height, width, depth, z, color=colors)

    ax3.set_xlabel('VP dose', fontsize=14, fontname="Arial")
    ax3.set_ylabel('IV fluid dose', fontsize=14, fontname="Arial")
    ax3.set_zlabel('Action counts', fontsize=14, fontname="Arial")
    ax3.set_title("D3QN-37 policy", fontsize=15, fontname="Arial")

    sm3 = cm.ScalarMappable(cmap=map_vir, norm=norm)  # norm设置最大最小值
    sm3.set_array([])
    cb3 = plt.colorbar(sm3)
    cb3.ax.tick_params(labelsize=10)  # 设置色标刻度字体大小。
    cb3.ax.set_yticklabels([0, 1500, 3000, 4500, 6000, 7500], fontname="Arial")

    ax3.set_ylim(5, -0.5)
    ax3.view_init(32, 43)

    plt.show()
