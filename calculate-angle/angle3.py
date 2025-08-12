import numpy as np
from math import degrees, acos

def calculate_angle_between_points(point1, point2, point3):
    vector1 = point1 - point2
    vector2 = point3 - point2
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    angle_rad = acos(dot_product / norm_product)
    angle_deg = degrees(angle_rad)
    return angle_deg

def calculate_distance_between_points(point1, point2):
    distance = np.linalg.norm(point2 - point1)
    return distance

def load_coordinates_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    coordinates = []
    for line in lines:
        # 假设每行的格式为 x y z，使用空格分隔
        x, y, z = map(float, line.strip().split())
        coordinates.append([x, y, z])
    return coordinates

# 从txt文件中加载坐标数据
file_path = '/home/pbc/project/zxh/calculate-angle/openpose3D_trans28.txt'
coordinates = load_coordinates_from_txt(file_path)

# 检查坐标点数量是否足够
if len(coordinates) < 3:
    print("Insufficient number of coordinates")
else:
    # 获取三个坐标点
    point1 = np.array(coordinates[2])
    point2 = np.array(coordinates[5])
    point3 = np.array(coordinates[8])

    # 计算角度
    angle = calculate_angle_between_points(point1, point2, point3)
    print(f"The angle between the points is: {angle:.2f}°")

    # 计算距离
    distance = calculate_distance_between_points(point1, point2)
    print(f"The distance between point1 and point2 is: {distance} units.")
