import sys
import os
project_directory = '..'
sys.path.append(os.path.abspath(project_directory))

import torch
import numpy as np
from plyfile import PlyData, PlyElement
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from mem import Motion_Estimation_Module
from scene.Motion_Grid import Motion_Grid
from argparse import ArgumentParser, Namespace

def fetchXYZ(path):
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    return torch.tensor(xyz, dtype=torch.float, device="cuda")

def get_xyz_bound(xyz, percentile=90):
    half_percentile = (100 - percentile) / 200
    xyz_bound_min = torch.quantile(xyz,half_percentile,dim=0)
    xyz_bound_max = torch.quantile(xyz,1 - half_percentile,dim=0)
    return xyz_bound_min, xyz_bound_max   

def get_contracted_xyz(xyz):
    xyz_bound_min, xyz_bound_max = get_xyz_bound(xyz, 90)
    normalzied_xyz=(xyz-xyz_bound_min)/(xyz_bound_max-xyz_bound_min)
    return normalzied_xyz

@torch.compile
def quaternion_multiply(a, b):
    a_norm=nn.functional.normalize(a)
    b_norm=nn.functional.normalize(b)
    w1, x1, y1, z1 = a_norm[:, 0], a_norm[:, 1], a_norm[:, 2], a_norm[:, 3]
    w2, x2, y2, z2 = b_norm[:, 0], b_norm[:, 1], b_norm[:, 2], b_norm[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack([w, x, y, z], dim=1)

def quaternion_loss(q1, q2):
    cos_theta = F.cosine_similarity(q1, q2, dim=1)
    cos_theta = torch.clamp(cos_theta, -1+1e-7, 1-1e-7)
    return 1-torch.pow(cos_theta, 2).mean()

def l1loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()
    
def adjust_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--pcd_path', type=str, default='/mnt/Projects/4DGC/test/coffee_martini/point_cloud/iteration_10000/point_cloud.ply')
    parser.add_argument('--q', type=int, default=1)
    parser.add_argument('--output_path', type=str, default='/mnt/Projects/4DGC/mem/coffee_martini.pth')
    args = parser.parse_args(sys.argv[1:])

    pcd_path=args.pcd_path
    model = Motion_Grid(q = args.q)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.get_optparam_groups())
    xyz=fetchXYZ(pcd_path)
    normalzied_xyz=get_contracted_xyz(xyz)
    mask = (normalzied_xyz >= 0) & (normalzied_xyz <= 1)
    mask = mask.all(dim=1)
    mem_inputs=torch.cat([normalzied_xyz[mask]],dim=-1)
    noisy_inputs = mem_inputs + 0.01 * torch.rand_like(mem_inputs)
    d_xyz_gt=torch.tensor([0.,0.,0.]).cuda()
    d_rot_gt=torch.tensor([1.,0.,0.,0.]).cuda()

    def cacheloss(resi):
        masked_d_xyz=resi[:,:3]
        masked_d_rot=resi[:,3:7]
        loss_xyz=l1loss(masked_d_xyz,d_xyz_gt)
        loss_rot=quaternion_loss(masked_d_rot,d_rot_gt)
        loss=loss_xyz+loss_rot
        return loss , loss_xyz.item(), loss_rot.item()
    for iteration in tqdm(range(0,3000),leave=False):      
        mem_inputs_w_noisy = torch.cat([noisy_inputs, mem_inputs, torch.rand_like(mem_inputs)],dim=0) 
        mem_output=model(mem_inputs_w_noisy).to(torch.float64)

        loss, loss_xyz, loss_rot=cacheloss(mem_output)
        if iteration % 100 ==0:
            print(loss.item(),loss_xyz,loss_rot)
        if iteration == 1500:
            adjust_learning_rate(optimizer, 1e-5)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none = True)

    mem=Motion_Estimation_Module(model,get_xyz_bound(xyz)[0],get_xyz_bound(xyz)[1])
    torch.save(mem.state_dict(),args.output_path)
    print('Done')

