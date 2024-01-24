import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as raster_settings
from load_blender import load_test_blender
from utils import fov2focal,params2rendervar
from external import calc_ssim,calc_psnr
from tqdm import tqdm

def load_scene_data(exp):
    params = dict(np.load(f"./output/{exp}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    rendervar = params2rendervar(params)
    return rendervar


def get_settings(curr_data):
    cam = raster_settings( 
            image_height=curr_data['cam']['image_height'],
            image_width=curr_data['cam']['image_width'],
            tanfovx=curr_data['cam']['tanfovx'],
            tanfovy=curr_data['cam']['tanfovy'],
            bg=curr_data['cam']['bg'],
            scale_modifier=curr_data['cam']['scale_modifier'],
            viewmatrix=curr_data['cam']['viewmatrix'],
            projmatrix=curr_data['cam']['projmatrix'],
            sh_degree=curr_data['cam']['sh_degree'],
            campos=curr_data['cam']['campos'],
            prefiltered=curr_data['cam']['prefiltered'],
            debug= curr_data['cam']['debug']
    )
    return cam

def get_matric(data, rendervar):
    with torch.no_grad():
        rendervar['means2D'].retain_grad()
        raster_settings = get_settings(data)
        render_image, radius= Renderer(raster_settings=raster_settings)(**rendervar)
        gt_image = data["im"] 
        psnr = calc_psnr(gt_image,render_image).mean()
        ssim = calc_ssim(gt_image,render_image)
    return ssim,psnr



def test(exp_name):

    datapath = "./dataset/nerf_synthetic/{}/".format(exp_name)
    dataset = load_test_blender(datapath)
    l = len(dataset)
    print(l)
    test_psnr = 0.000
    test_ssim = 0.000
    progress_bar = tqdm(l, desc="Testing progress")

    rendervar = load_scene_data(exp_name)

    for data in dataset: 
        data['cam']['sh_degree']=3
        ssim,psnr = get_matric(data, rendervar)
        test_psnr += psnr
        test_ssim += ssim
        progress_bar.set_postfix({"PSNR": f"{psnr:.{7}f}","SIMM":f"{ssim:.{7}f}"})
        progress_bar.update(1)
    progress_bar.close()
    avg_test_psnr = test_psnr / len(dataset)
    avg_test_ssim = test_ssim / len(dataset)
    print("avg_test_psnr: {} avg_test_ssim: {}".format(avg_test_psnr,avg_test_ssim))


if __name__=="__main__":
    exp_name= 'lego'
    test(exp_name)