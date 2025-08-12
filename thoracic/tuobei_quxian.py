import pickle
import smplx
import torch
import numpy as np
import trimesh
from scipy.optimize import leastsq

body_model_pred = smplx.create(model_path=r"C:\Users\20630\Desktop\thoracic(1)\thoracic\SMPLX_MALE.npz", model_type='smplx')
with torch.no_grad():
    body_model_pred.betas[0] = torch.Tensor([
        -0.3804152 , -0.9013766 ,  0.52706015,  0.41512382,  0.47160444,
        -0.08584227,  0.28042233, -0.0588825 , -0.26926714,  0.773316 ]
    )
#     body_model_pred.body_pose[0] = torch.Tensor( [0.057333845645189285, -0.033230073750019073, 0.007310014218091965, 0.057333845645189285, 0.033230073750019073, -0.007310014218091965, 0.0003911537933163345, -0.02989751100540161, 0.016725363209843636, 0.07142435014247894, 0.0, 0.0, 0.07142435014247894, 0.0, 0.0, 0.019599782302975655, -0.006483905017375946, -0.02004614844918251, 0.0, 0.02901981770992279, 0.0, 0.0, -0.0855453610420227, 0.0, 0.03167254850268364, 0.023192578926682472, 0.017747951671481133, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.006921793334186077, 0.0, 0.0, 0.0, 0.0, 0.015285802073776722, 0.0, 0.0, 0.08174237608909607, 0.03824948891997337, 0.0, 0.0, -0.17000000178813934, 0.0, -0.8188637495040894, -0.17000000178813934, 0.0, 0.8184133768081665, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#  ]
#    )
    body_model_pred.body_pose[0,47] = -0.87
    body_model_pred.body_pose[0,50] = 0.87
    # body_model_pred.body_pose[0,45] = -0.17
    # body_model_pred.body_pose[0,48] = -0.17


body_model_output = body_model_pred(return_verts=True)
verts_pred = body_model_output.vertices

# out_mesh = trimesh.Trimesh(verts_pred.detach().cpu().numpy().squeeze(), body_model_pred.faces, process=False)
# out_mesh.export("last/shentao_std.obj")

with torch.no_grad():
    body_model_pred.betas[0] = torch.Tensor([
        -0.3804152 , -0.9013766 ,  0.52706015,  0.41512382,  0.47160444,
        -0.08584227,  0.28042233, -0.0588825 , -0.26926714,  0.773316 ]
    )
    body_model_pred.body_pose[0,6] =0.10
    body_model_pred.body_pose[0,24] =0.10
    body_model_pred.body_pose[0,47] = -0.87
    body_model_pred.body_pose[0,50] = 0.87

body_model_output = body_model_pred(return_verts=True)
verts_add = body_model_output.vertices


verts = verts_pred.detach().numpy().squeeze()
idx = np.loadtxt('last/idx.txt',dtype=int)
verts[:,[1,2]] = verts[:,[2,1]]
verts = verts[:,1:3]
line_std = verts[idx]
line_std = line_std[np.argsort(line_std[:,1])]
# line_std = line_std[18:42]

verts_tuobei = np.loadtxt('last/shentao_tuobei_adst.txt')
verts_tuobei = verts_tuobei[:,0:3]
verts_tuobei[:,[1,2]] = verts_tuobei[:,[2,1]]
verts_tuobei = verts_tuobei[:,1:3]
line_tuobei = verts_tuobei[idx]
line_tuobei = line_tuobei[np.argsort(line_tuobei[:,1])]



verts_add = verts_add.detach().numpy().squeeze()
verts_add[:,[1,2]] = verts_add[:,[2,1]]
verts_add = verts_add[:,1:3]
line_add = verts_add[idx]
line_add = line_add[np.argsort(line_add[:,1])]


# 两个向量
def angle(x,y):
    Lx=np.sqrt(x.dot(x))
    Ly=np.sqrt(y.dot(y))
    cos_angle=x.dot(y)/(Lx*Ly)
    angle=np.arccos(cos_angle)
    angle2=angle*360/2/np.pi
    # print(angle2)
    return angle2


x_std = line_std[15] - line_std[29]
y_std = line_std[40] - line_std[29]

x_tuobei = line_tuobei[15] - line_tuobei[25]
y_tuobei = line_tuobei[40] - line_tuobei[25]

x_add = line_add[15] - line_add[27]
y_add = line_add[40] - line_add[27]

ag_std = angle(x_std, y_std)
ag_tuobei = angle(x_tuobei, y_tuobei)
ag_add = angle(x_add, y_add)

print(ag_std, ag_add, ag_tuobei)


x_tt = line_tuobei[15] - line_tuobei[40]
y_tt = line_std[15] - line_std[40]

print(angle(x_tt,y_tt))











from matplotlib import pyplot as plt

plt.figure(1)

plt.scatter(line_std[:,0],line_std[:,1],c='red',marker='.')
plt.scatter(line_tuobei[:,0],line_tuobei[:,1],c='blue',marker='.')
plt.scatter(line_add[:,0],line_add[:,1],c='green',marker='.')

plt.scatter(line_std[18:43,0],line_std[18:43,1],c='red')
plt.scatter(line_tuobei[18:43,0],line_tuobei[18:43,1],c='blue')
plt.scatter(line_add[18:43,0],line_add[18:43,1],c='green')

# plt.scatter(line_std[29,0],line_std[29,1],c='black')
# plt.scatter(line_tuobei[25,0],line_tuobei[25,1],c='black')
# # plt.scatter(line_add[27,0],line_add[27,1],c='black')


# line_s = np.vstack((line_std[15],line_std[29],line_std[40]))
# line_a = np.vstack((line_add[15],line_add[27],line_add[40]))
# line_t = np.vstack((line_tuobei[15],line_tuobei[25],line_tuobei[40]))


line_s = np.vstack((line_std[15],line_std[40]))
# line_a = np.vstack((line_add[15],line_add[27],line_add[40]))
line_t = np.vstack((line_tuobei[15],line_tuobei[40]))

plt.plot(line_s[:,0],line_s[:,1],c='black',linewidth=2)
plt.plot(line_t[:,0],line_t[:,1],c='black',linewidth=2)
# plt.plot(line_a[:,0],line_a[:,1],c='black',linewidth=2)


plt.axis('off')
plt.axis('equal')
plt.show()














# with open('last/tuobei.pkl','rb') as f:
#     para = pickle.load(f)

# print(para.body_pose)





# # 最小二乘
# def fun(p,x):
#     a, b = p
#     return x*a+b

# def error(p,x,y):
#     return fun(p,x)-y

# p0 = [0.1,-1]
# para_std = leastsq(error,p0, args=(line_std[:,0],line_std[:,1]))
# tmp_std = np.linspace(start=-0.2,stop=0.02)
# y_fitted_std = fun(para_std[0],tmp_std)

# para_tuobei = leastsq(error,p0, args=(line_tuobei[:,0],line_tuobei[:,1]))
# tmp_tuobei = np.linspace(start=-0.2,stop=0.02)
# y_fitted_tuobei = fun(para_tuobei[0],tmp_tuobei)