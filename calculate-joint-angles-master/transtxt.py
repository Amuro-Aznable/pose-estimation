import numpy as np

# 从txt文件中读取数据
txt_file = r'C:\Users\20630\Desktop\calculate-joint-angles-master\datasets\NTU\nturgb+d_npy\openpose3D_trans180.txt'
data = np.loadtxt(txt_file)  # 假设txt文件包含25行，每行包含3列

# 将数据保存为npy文件
npy_file = '180.npy'
np.save(npy_file, data)
