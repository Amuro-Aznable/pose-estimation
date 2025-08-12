import smplx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from vector import Vector
import trimesh
import math
import torch.nn.functional as F
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_humanmodel(joints_input, vertices_input):
    verts = vertices_input.detach().cpu().numpy()
    joint = joints_input.detach().cpu().numpy()
    verts = (verts + 1) / 2 * 255
    joint = (joint + 1) / 2 * 255

    verts_x = verts[:, :, 0]
    verts_y = verts[:, :, 2]
    verts_z = verts[:, :, 1]
    joints_x = joint[:, :24, 0]
    joints_y = joint[:, :24, 2]
    joints_z = joint[:, :24, 1]

    x_max = np.max([np.max(verts_x), np.max(joints_x)])
    x_min = np.min([np.min(verts_x), np.min(joints_x)])

    y_max = np.max([np.max(verts_y), np.max(joints_y)])
    y_min = np.min([np.min(verts_y), np.min(joints_y)])

    z_max = np.max([np.max(verts_z), np.max(joints_z)])
    z_min = np.min([np.min(verts_z), np.min(joints_z)])

    def calculate_Vector(coord1, coord2, coord3):
        """
        coord1,coord2,coord3:三个关节点


        """
        vec1_x = joints_x[0, coord1] - joints_x[0, coord2]
        vec1_y = joints_y[0, coord1] - joints_y[0, coord2]
        vec1_z = joints_z[0, coord1] - joints_z[0, coord2]

        vec2_x = joints_x[0, coord3] - joints_x[0, coord2]
        vec2_y = joints_y[0, coord3] - joints_y[0, coord2]
        vec2_z = joints_z[0, coord3] - joints_z[0, coord2]

        v = [str(vec1_x), str(vec1_y), str(vec1_z)]
        w = [str(vec2_x), str(vec2_y), str(vec2_z)]

        vec1 = Vector(v)
        vec2 = Vector(w)

        return vec1.angle_with(vec2)
    
    def calculate_Vector2(coord1, coord2, normal):
        """
        coord1:顶点
        normal：地平面法向量


        """
        # vec1_x = joints_x[0, coord1] - joints_x[0, coord2]
        # vec1_y = joints_y[0, coord1] - joints_y[0, coord2]
        # vec1_z = joints_z[0, coord1] - joints_z[0, coord2]

        vec1_x = verts_x[0, coord1] - verts_x[0, coord2]
        vec1_y = verts_y[0, coord1] - verts_y[0, coord2]
        vec1_z = verts_z[0, coord1] - verts_z[0, coord2]

        

        v = [str(vec1_x), str(vec1_y), str(vec1_z)]
        w = [str(normal[0]), str(normal[1]), str(normal[2])]

        vec1 = Vector(v)
        vec2 = Vector(w)

        return abs(90-vec1.angle_with(vec2))
    
    def calculate_Vector3(coord1, coord2, normal):
        """
        coord1:关节点
        
        
        """
        vec1_x = joints_x[0, coord1] - joints_x[0, coord2]
        vec1_y = joints_y[0, coord1] - joints_y[0, coord2]
        vec1_z = joints_z[0, coord1] - joints_z[0, coord2]

        

        

        v = [str(vec1_x), str(vec1_y), str(vec1_z)]
        w = [str(normal[0]), str(normal[1]), str(normal[2])]

        vec1 = Vector(v)
        vec2 = Vector(w)

        return vec1.angle_with(vec2)
    

    
    def plane(point):
        x1 = point[0][0]
        y1 = point[0][1]
        z1 = point[0][2]
        x2 = point[1][0]
        y2 = point[1][1]
        z2 = point[1][2]
        x3 = point[2][0]
        y3 = point[2][1]
        z3 = point[2][2]
        A = (y2-y1)*(z3-z1) - (z2-z1)*(y3-y1)
        B = (x3-x1)*(z2-z1) - (x2-x1)*(z3-z1)
        C = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
        D = -(A*x1 + B*y1 + C*z1)
        # print('平面合结果拟为：%.3f * x + %.3f * y + %.3f *z + %.3f = 0'%(A,B,C,D))
        # print(A,B,C,D)
        # ans = A*variable[0] + B*variable[1] + C*variable[2] + D
        return A,B,C,D
    
    def distance_point_to_line(point, line_point, line_direction):
        """
        计算点到直线的距离
        :param point: 要计算距离的点，格式为 (x, y, z)
        :param line_point: 直线上的一点，格式为 (x0, y0, z0)
        :param line_direction: 直线的方向向量，格式为 (a, b, c)
        :return: 点到直线的距离
        """
        point = np.array(point)
        line_point = np.array(line_point)
        line_direction = np.array(line_direction)
        distance = np.linalg.norm(np.cross(line_direction, point - line_point)) / np.linalg.norm(line_direction)
        return distance
    

    

    

    ground_points = np.loadtxt("/home/pbc/project/inference/pose_3d/ground_point_in_smpl.txt")
    delta= np.loadtxt("/home/pbc/project/inference/cores/OUTPUT_FOLDER/last_res/delta.txt")
    ground_points[:,0]-=delta[0]
    ground_points[:,1]-=delta[1]
    ground_points[:,2]-=delta[2]

    ground_points = ground_points[[0,5,9]]
    a,b,c,_ = plane(ground_points)
    normal_vector = np.array([a,b,c])
    print(normal_vector)
    level_vector = np.array([0,1,0])
    vertical_vector = np.array([0,0,1])
    # print(normal_vector)

    for i in range(verts.shape[0]):
        fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.view_init(elev=20, azim=-120)  # 调整观察角度
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=20, azim=120)  # 调整观察角度
        ax.set_box_aspect([1, 1, 1])  # 设置坐标轴比例为1:1:1
        ax.set_proj_type('ortho')
        ax.scatter(verts_x[i, :], verts_y[i, :], verts_z[i, :], s=0.05, c='y')  # 画顶点
        # ax.plot_trisurf(verts_x[i, :], verts_y[i, :], faces, verts_z[i, :], shade=True, color='y')
        

        ax.scatter(joints_x[i, :23], joints_y[i, :23], joints_z[i, :23], s=3, c='r')

  
        for n in range(24):
            ax.text(joints_x[i, n] + 2, joints_y[i, n] + 2, joints_z[i, n] + 2, '{}'.format(n), fontsize=7)

        # line_map = [[1, 4], [4, 7], [2, 5], [5, 8], [17, 19], [19, 21],
        #             [16,18], [18, 20], [12, 9], [9, 0]]
        line_map = [[16, 17],[15,12]]
        
        for i in range(len(line_map)):
            first_point = line_map[i][0]
            second_point = line_map[i][1]
            line_x = [joints_x[0, first_point], joints_x[0, second_point]]
            line_y = [joints_y[0, first_point], joints_y[0, second_point]]
            line_z = [joints_z[0, first_point], joints_z[0, second_point]]
            ax.plot(line_x, line_y, line_z, c='r')



        ax.set_xlim3d(x_min - 50, x_max + 50)
        ax.set_ylim3d(y_min - 50, y_max + 50)
        ax.set_zlim3d(z_max + 50, z_min - 50)
            

        print('右侧大腿与小腿间的角度：', calculate_Vector(2, 5, 8))
        print('左侧大腿与小腿间的角度：', calculate_Vector(1, 4, 7))
        print('右臂打开的角度：', calculate_Vector(17, 19, 21))
        # print("两个肩膀和水平面的角度：",calculate_Vector2(3011,6470,normal=level_vector))
        print("两个肩膀和水平面的角度：",90-calculate_Vector3(16,17,normal=level_vector))
        print("高低肩的两肩高度差：",verts_y [0,3011]-verts_y [0,6470])
        # print("头前伸的角度",calculate_Vector2())
        # 判断头部前移 （颈椎前曲）  头部关节点 
        # 首先判断头部偏离肩膀连线的水平距离，在判断头部与肩膀中线连接与水平面的夹角
        # 1 颈部关节点15，左肩膀 16 右肩膀 17
        distance = joints_y[0,15]-joints_y[0,12]
        print("头部偏离肩膀的距离：",distance)
        # print("头部偏离肩膀的角度：",calculate_Vector3(15,12,normal=shuiping_vector))
        # 头部倾斜(身体倾斜)：计算头部关节点偏离人体中轴的角度
        # 人体中轴的定义为：过根节点且垂直地平面的平面的方向向量
        
        print("头部侧倾斜的角度",calculate_Vector3(22,17,vertical_vector),)
        # 计算头部倾斜偏离中轴线的距离:15号点到人体中轴线的距离
        
        print(f"骨盆倾斜的角度：{calculate_Vector2(1,2,normal=level_vector)}")
        
        
    
        
        # 给出人体中轴线的直线方程
        COG = joint[0, 0]
        line = normal_vector
        point = joint[0, 15]

        distance = distance_point_to_line(point, COG, line)
        
        print("头部偏离中轴线的距离为:", distance, "cm")
        plt.show()
        
    

idx = 391
with open(f'/home/pbc/project/inference/cores/OUTPUT_FOLDER/last_res/shape/{idx}_betas.txt', 'r') as f:
    beta_values = np.loadtxt(f)
with open(f'/home/pbc/project/inference/cores/OUTPUT_FOLDER/last_res/pose/{idx}_pose.txt', 'r') as f:
    pose_values = np.loadtxt(f)


body_model = smplx.create(model_path="SMPLX_MALE.npz", model_type='smplx')

with torch.no_grad():
    body_model.betas[0] = torch.tensor(beta_values)
    body_model.body_pose[0] = torch.tensor(pose_values)

faces = body_model.faces
output = body_model(return_verts=True, return_full_pose=False)

joints = output.joints 
vertices = output.vertices

import numpy as np

def plane_from_points(A, B, C):
    # 将点 A, B, C 转换为 numpy 数组
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    # 计算向量 AB 和 AC
    AB = B - A
    AC = C - A

    # 计算法向量 N = AB x AC
    N = np.cross(AB, AC)

    # 从法向量和点 A 计算平面方程的常数 d
    d = -np.dot(N, A)

    # 平面方程是 ax + by + cz + d = 0
    return N[0], N[1], N[2], d

# 示例脚踝点和参考点
ankle1 = joints[0,8].detach().numpy()
ankle2 = joints[0,7].detach().numpy()
reference_point = joints[0,0].detach().numpy()
# 计算平面方程
a, b, c, d = plane_from_points(ankle1, ankle2, reference_point)
print(f"平面方程：{a}x + {b}y + {c}z + {d} = 0")
# 调用计算关节角度的函数
plot_humanmodel(joints_input=joints, vertices_input=vertices)

# 保存生成的三维模型
# vertices = vertices.detach().cpu().numpy().squeeze()
# # 创建三维模型
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

# # 保存为OBJ文件
# output_path = "/home/pbc/project/zxh/calculate-angle/stand.obj"  # 选择保存路径
# mesh.export(output_path)

# print(f"三维模型已保存为 {output_path}")