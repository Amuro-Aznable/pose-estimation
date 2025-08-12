import numpy as np
from math import degrees, acos

def calculate_angle_between_joints(joint1, joint2, joint3):
    vector1 = joint1 - joint2
    vector2 = joint3 - joint2
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    angle_rad = acos(dot_product / norm_product)
    angle_deg = degrees(angle_rad)
    return angle_deg

# 从txt文件中加载坐标数据
def load_coordinates_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    coordinates = []
    for line in lines:
        # 假设每行的格式为 x,y,z
        x, y, z = map(float, line.strip().split())
        coordinates.append([x, y, z])
    return coordinates

# 从txt文件中加载关节点的3D坐标数据
file_path = '/home/pbc/project/zxh/calculate-angle/openpose3D_trans28.txt'
coordinates = load_coordinates_from_txt(file_path)

# 选择要计算角度的关节点索引（从0开始）
joint_index1 = 5
joint_index2 = 17

# 检查关节点索引是否超出坐标数据的范围
if joint_index1 >= len(coordinates) or joint_index2 >= len(coordinates):
    print("Invalid joint index")
else:
    # 获取关节点的坐标
    joint1 = np.array(coordinates[joint_index1])
    joint2 = np.array(coordinates[joint_index2])

    # 计算角度
    angle = calculate_angle_between_joints(joint1, joint2, joint1)
    print(f"The angle between joint {joint_index1} and joint {joint_index2} is: {angle} degrees.")
