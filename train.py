import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
from load_blender import load_train_blender,load_test_blender
import torch
from tqdm import tqdm
from random import randint
from utils import l1_loss_v1, params2rendervar, params2cpu, save_params,l1_loss_v2,smooth_l1_loss,weighted_l2_loss_v1,weighted_l2_loss_v2
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from external import calc_ssim,calc_psnr,densify
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings as raster_settings
import copy
from PIL import Image
from external import get_expon_lr_func
from torch.utils.tensorboard import SummaryWriter

def initialize_optimizer(params, variables):
    lrs = {
        'means3D': 0.00016 * variables['scene_radius'], 
        'feature_dc': 0.0025,
        'feature_rest':0.000125,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,

    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


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

def test(params,test_data):
    
    rendervar = params2rendervar(params)  
    raster_settings = get_settings(test_data)
    render_image, radius= Renderer(raster_settings=raster_settings)(**rendervar)
    gt_image = test_data["im"]
    psnr = calc_psnr(gt_image,render_image).mean()

    return psnr


def get_loss(params, curr_data,variables):

    rendervar = params2rendervar(params)  
    rendervar['means2D'].retain_grad()
    raster_settings = get_settings(curr_data)
    render_image, radius= Renderer(raster_settings=raster_settings)(**rendervar)

    gt_image = curr_data["im"] 
    loss_l2 = l1_loss_v1(render_image,gt_image)
    Loss = 0.8 * loss_l2 + 0.2 * (1.0-calc_ssim(render_image,gt_image))

    psnr = calc_psnr(gt_image,render_image).mean()

    variables['means2D'] = rendervar['means2D'] 
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return Loss,variables,psnr

def report_progress(psnr, i , progress_bar, loss, points, every_i=10):
    if i % every_i==0:
        progress_bar.set_postfix({"PSNR": f"{psnr:.{7}f}","Loss":f"{loss:.{7}f}", "points":f"{points}"})
        progress_bar.update(every_i)


def level_up_sh(dataset,todo_dataset,test_dataset):
    for data in dataset:
        data['cam']['sh_degree']+=1
    if todo_dataset is not None:
        for data in todo_dataset:
            data['cam']['sh_degree']+=1
    for data in test_dataset:
        data['cam']['sh_degree']+=1


def train(exp_name):
    datapath = "./dataset/nerf_synthetic/{}/".format(exp_name)
    params, variables, dataset = load_train_blender(datapath)
    test_dataset = load_test_blender(datapath)
    optimizer = initialize_optimizer(params, variables)
    writer = SummaryWriter()

    progress_bar = tqdm(range(30000), desc="Training progress")
    todo_dataset = []
    xyz_lr_scheduler = get_expon_lr_func(lr_init= 0.00016 * variables['scene_radius'],lr_final= 0.0000016 * variables['scene_radius'],
                                                    lr_delay_mult=0.01,max_steps=30000)
    
    for i in range(1,30001): 
        
        for param_group in optimizer.param_groups:
            if param_group["name"] == "means3D":
                lr = xyz_lr_scheduler(i)
                param_group['lr'] = lr
                break
        # 每隔1k次增加一次sh
        if i % 1000 == 0 and dataset[0]["cam"]['sh_degree'] < 3 :
            level_up_sh(dataset,todo_dataset,test_dataset)
            
            
        if not todo_dataset:
            todo_dataset = copy.deepcopy(dataset)
        curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1)) 

        loss,variables,psnr = get_loss(params, curr_data, variables)

        writer.add_scalar('Train/Loss', loss.item(), i)
        writer.add_scalar('Train/PSNR', psnr, i)
        writer.add_scalar('Gaussions count', params['means3D'].shape[0], i)

        loss.backward()
        with torch.no_grad():
            report_progress(psnr, i, progress_bar,loss, params['means3D'].shape[0])
            params, variables = densify(params, variables, optimizer, i)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if  i == 7000 or i == 30000:
            print("save model in {} iteration".format(i))
            output_params = params2cpu(params)
            seq = os.path.basename(datapath)
            save_params(output_params,exp_name,seq)
        
        if i % 10 == 0:
            test_data = test_dataset[randint(0, len(test_dataset) - 1)]
            test_psnr = test(params,test_data)
            writer.add_scalar('Test/PSNR', test_psnr, i)                 
    writer.close()
    progress_bar.close()


if __name__=='__main__':
    exp_name = "lego" 
    train(exp_name)







