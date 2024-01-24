import os
import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as raster_settings
from utils import fov2focal,pose_spherical,setup_camera
import imageio
from eval import load_scene_data

w, h = 800, 800
near, far = 0.01, 100.0
view_scale = 3.9

def render(w2c,k,rendervar):
    cam = setup_camera(w, h, k, w2c, near, far)
    settings  = raster_settings(
            image_height=cam['image_height'],
            image_width=cam['image_width'],
            tanfovx=cam['tanfovx'],
            tanfovy=cam['tanfovy'],
            bg=cam['bg'],
            scale_modifier=cam['scale_modifier'],
            viewmatrix=cam['viewmatrix'],
            projmatrix=cam['projmatrix'],
            sh_degree=cam['sh_degree'],
            campos=cam['campos'],
            prefiltered=cam['prefiltered'],
            debug= cam['debug']
    )
    image,depth = Renderer(raster_settings=settings)(**rendervar)
    return image, depth



def init_cameras():
    c2w_matrixs = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    fx = fov2focal(0.6911112070083618,800.0)
    k = np.array([[fx,0,400.0],[0,fx,400.0],[0,0,1]],dtype=float)
    return c2w_matrixs, k



def visualize(exp_name):
    rendervar = load_scene_data(exp_name)
    c2w_matrixs,k = init_cameras()
    images =[]
    to8b = lambda x : (255*np.clip(x.detach().cpu().numpy(),0,1)).astype(np.uint8)
    for c2w in c2w_matrixs:
        c2w[:3, 1:3] *= -1  # 改为colmap
        w2c = np.linalg.inv(c2w)
        image, depth = render(w2c, k, rendervar)
        images.append(to8b(image).transpose(1,2,0))
    
    imageio.mimwrite(os.path.join("./output/{}/".format(exp_name),'video_rgb.mp4'), images, fps=30)


if __name__=="__main__":
    exp_name= 'lego'
    visualize(exp_name)


