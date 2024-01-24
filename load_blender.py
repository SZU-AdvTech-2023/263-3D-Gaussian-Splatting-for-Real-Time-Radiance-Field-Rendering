import numpy as np
from utils import setup_camera,o3d_knn,fov2focal,inverse_sigmoid
import torch
import os
import json
from PIL import Image

def read_json(path, file_name ):
    cam_centers = []
    dataset = []
    h,w = 800,800
    cx,cy = 400,400
    with open(os.path.join(path,file_name)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        frames = contents["frames"]
        for idx, frame in enumerate(frames):

            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            cam_centers.append(c2w[:3,3])  
            fx = fov2focal(fovx,w)
            k = np.zeros((3,3),dtype=float)
            k[0][0], k[1][1], k[0][2], k[1][2] = fx, fx, cx, cy
            cam = setup_camera(h, w, k, w2c, near=0.01, far=100)
            image_path = os.path.join(path, frame["file_path"] + ".png")
            bg = [0,0,0]
            norm_data = np.array(Image.open(image_path).convert("RGBA")) / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            im = torch.from_numpy(arr).float().cuda().permute(2, 0, 1).clamp(0.0,1.0) 

            dataset.append({'cam': cam, 'im': im, 'id': idx})

    return dataset,cam_centers

def load_train_blender(path):
    num_pts = 100_000
    xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3 
    feature_dc = np.random.random((num_pts,1, 3)) / 255.0
    feature_rest = np.zeros((num_pts,15,3))

    sq_dist, _ = o3d_knn(xyz,3) 
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)

    params = {
        'means3D': xyz,                                                 
        'feature_dc':  feature_dc,
        'feature_rest': feature_rest,
        'unnorm_rotations': np.tile([1, 0, 0, 0], (num_pts, 1)), 
        'logit_opacities': inverse_sigmoid(0.1 * np.ones((num_pts, 1))),
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
    }

    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in params.items()}
    dataset_train , cam_centers_train = read_json(path,"transforms_train.json") 

    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers_train - np.mean(cam_centers_train, 0)[None], axis=-1))
    variables = {'max_2D_radius': torch.zeros(num_pts).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(num_pts).cuda().float(),
                 'denom': torch.zeros(num_pts).cuda().float()}
    
    return params, variables,dataset_train

def load_test_blender(path):
    
    dataset_test, cam_centers_test = read_json(path,"transforms_test.json")

    for data in dataset_test:
        data['cam']['sh_degree']=3

    return dataset_test

def load_val_blender(path):
    
    dataset_valid, cam_centers_valid = read_json(path,"transforms_val.json")

    for data in dataset_valid:
        data['cam']['sh_degree']=3

    return dataset_valid
    
    


