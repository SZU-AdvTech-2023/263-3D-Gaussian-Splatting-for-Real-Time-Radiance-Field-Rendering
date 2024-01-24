import torch
import os
import open3d as o3d
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
import math


trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]],dtype=float)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]],dtype=float)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]],dtype=float)

#相机绕物体一圈位姿
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return  c2w 

def setup_camera(w, h, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()  # [4,4]
    cam_center = torch.inverse(w2c)[:3, 3]  # 
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0], 
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)   

    cam = {  "image_height":h,
        "image_width":w,
        "tanfovx":w / (2 * fx),
        "tanfovy":h / (2 * fy),
        "bg":torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        "scale_modifier":1.0,
        "viewmatrix":w2c,
        "projmatrix":full_proj,
        "sh_degree":0,
        "campos":cam_center,
        "prefiltered":False,
        "debug": False}
    return cam


def params2rendervar(params):

    rendervar = {
        'means3D': params['means3D'],
         # 'colors_precomp': params['rgb_colors'],
        'shs':   torch.cat((params['feature_dc'], params['feature_rest']), dim=1)  ,  
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(params['log_scales']),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar

def smooth_l1_loss(x, y, beta=0.1):
    """
    Smooth L1 Loss function.

    Parameters:
    - predictions: Tensor, predicted values.
    - targets: Tensor, target values.
    - beta: float, controls the point where the loss changes from L2 to L1.

    Returns:
    - loss: Tensor, smooth L1 loss.
    """
    diff = torch.abs(x - y)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    
    return loss.mean()

def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def o3d_knn(pts, num_knn): 
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()  
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd) 
    for p in pcd.points:                     
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)   
        indices.append(i[1:])
        sq_dists.append(d[1:])   
    return np.array(sq_dists), np.array(indices)


def params2cpu(params):
    res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items()}  
    return res


def save_params(output_params,exp,seq):
    os.makedirs(f"./output/{exp}/{seq}", exist_ok=True)
    np.savez(f"./output/{exp}/{seq}/params", **output_params)


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def inverse_sigmoid(x):
    return np.log(x / (1 - x))