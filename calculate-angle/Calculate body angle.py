import smplx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from vector import Vector
import trimesh
import math
import torch.nn.functional as F
import json

def apply_symmetry(points):
    # Symmetry transformation: negate x-coordinate
    points[:, :, 0] = -points[:, :, 0]
    return points

def plot_humanmodel(joints_input, vertices_input, vertex_alpha=1.0,scale_factor=1.0):
    joints_input = apply_symmetry(joints_input)
    vertices_input = apply_symmetry(vertices_input)

    verts = vertices_input.detach().cpu().numpy()
    joint = joints_input.detach().cpu().numpy()
   
    verts = (verts + 1) / 2 * 255 
    joint = (joint + 1) / 2 * 255 
    

    verts_x = verts[:, :, 0]
    verts_y = verts[:, :, 2]
    verts_z = verts[:, :, 1]
    joints_x = joint[:, :, 0]
    joints_y = joint[:, :, 2]
    joints_z = joint[:, :, 1]


    x_max = np.max([np.max(verts_x), np.max(joints_x)])
    x_min = np.min([np.min(verts_x), np.min(joints_x)])

    y_max = np.max([np.max(verts_y), np.max(joints_y)])
    y_min = np.min([np.min(verts_y), np.min(joints_y)])

    z_max = np.max([np.max(verts_z), np.max(joints_z)])
    z_min = np.min([np.min(verts_z), np.min(joints_z)])

    def calculate_Vector(coord1, coord2, coord3):
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
    
    for i in range(verts.shape[0]):
        fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.view_init(elev=20, azim=-120)  # 调整观察角度
        
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=20, azim=120)  # 调整观察角度
        ax.set_box_aspect([1, 1, 1])  # 设置坐标轴比例为1:1:1
        ax.set_proj_type('ortho')
        ax.scatter(verts_x[i, :], verts_y[i, :], verts_z[i, :], s=0.05, c='b', alpha=vertex_alpha)  # 画顶点
        
        ax.scatter(joints_x[i, :23], joints_y[i, :23], joints_z[i, :23], s=3, c='r')
        
        for n in range(24):
            ax.text(joints_x[i, n] + 2, joints_y[i, n] + 2, joints_z[i, n] + 2, '{}'.format(n), fontsize=7)
            
        # line_map = [[4, 5]]
        # for i in range(len(line_map)):
        #     first_point = line_map[i][0]
        #     second_point = line_map[i][1]
        #     line_x = [joints_x[0, first_point], joints_x[0, second_point]]
        #     line_y = [joints_y[0, first_point], joints_y[0, second_point]]
        #     line_z = [joints_z[0, first_point], joints_z[0, second_point]]
        #     if line_map[i] == [2, 8] or line_map[i] == [1, 7]:
        #         ax.plot(line_x, line_y, line_z, c='r')  # Change the color to yellow
        #     else:
        #         ax.plot(line_x, line_y, line_z, c='g')
            # ax.plot(line_x, line_y, line_z, c='r')
        ax.set_xlim3d(x_min - 50, x_max + 50)
        ax.set_ylim3d(y_min - 50, y_max + 50)
        ax.set_zlim3d(z_max + 50, z_min - 50)
        # ax.set_xlim3d(x_min - 25, x_max + 25)
        # ax.set_ylim3d(y_min - 25, y_max + 25)
        # ax.set_zlim3d(z_max + 25, z_min - 25)
        
        
        print('Left knee joint angle is:', calculate_Vector(1, 4, 7))
        print('Right knee joint angle is:', calculate_Vector(2, 5, 8))
        # print('Right knee joint angle is:', calculate_Vector(0, 17, 21))
        # print('Left knee joint angle is:', calculate_Vector(9, 10, 11))
        # print('Light ankle joint angle is', calculate_Vector(10, 11, 22))
        # print('Left elbow joint angle is', calculate_Vector(2, 3, 4))
      
      
        plt.show()

# with open('/home/pbc/project/inference/cores/OUTPUT_FOLDER/last_res/shape/126_betas.txt', 'r') as f:
#     beta_values = np.loadtxt(f)
# with open('/home/pbc/project/inference/cores/OUTPUT_FOLDER/last_res/pose/126_pose.txt', 'r') as f:
#     pose_values = np.loadtxt(f)


body_model = smplx.create(model_path="SMPLX_MALE.npz", model_type='smplx')

# with torch.no_grad():
#     body_model.betas[0] = torch.tensor(beta_values)
#     body_model.body_pose[0] = torch.tensor(pose_values)

# with open ('/home/pbc/project/inference/pose_3d/cam_para.json') as f:
#     cam_para = json.load(f)
# torch_param = {}
# for key in cam_para.keys():
#     if key in ['front2', 'right1', 'left3','side4']:
#         continue
#     else:
#         torch_param[key] = torch.tensor(cam_para[key])
# output = body_model(return_verts=True, return_full_pose=False,**torch_param)
output = body_model(return_verts=True, return_full_pose=False)
joints = output.joints 
vertices = output.vertices


plot_humanmodel(joints_input=joints, vertices_input=vertices, vertex_alpha=0.4,scale_factor=1.0)