import torch

class Motion_Estimation_Module(torch.nn.Module):
    def __init__(self, model, xyz_bound_min, xyz_bound_max):
        super(Motion_Estimation_Module, self).__init__()
        self.model = model
        self.register_buffer('xyz_bound_min',xyz_bound_min)
        self.register_buffer('xyz_bound_max',xyz_bound_max)
        
    def dump(self, path):
        ckpt = self.state_dict()
        for key in list(self.model.state_dict().keys()):
            ckpt.pop(f'model.{key}')

        torch.save(ckpt,path)
        self.model.save(path.replace('.pth','_model.pth'))

    def load(self, path):
        ckpt = torch.load(path)
        self.load_state_dict(ckpt,strict=False)
        self.model.load(path.replace('.pth','_model.pth'))
        
    def get_contracted_xyz(self, xyz):
        contracted_xyz = (xyz - self.xyz_bound_min) / (self.xyz_bound_max - self.xyz_bound_min)
        return contracted_xyz
        
    def forward(self, xyz:torch.Tensor):
        contracted_xyz=self.get_contracted_xyz(xyz)                       
        
        mask = (contracted_xyz >= 0) & (contracted_xyz <= 1)
        mask = mask.all(dim=1)
        
        mem_inputs=torch.cat([contracted_xyz[mask]],dim=-1)
        
        resi=self.model(mem_inputs)
        masked_d_xyz=resi[:,:3]
        masked_d_rot=resi[:,3:7]
        
        d_xyz = torch.full((xyz.shape[0], 3), 0.0, dtype=torch.float, device="cuda")
        d_rot = torch.full((xyz.shape[0], 4), 0.0, dtype=torch.float, device="cuda")
        d_rot[:, 0] = 1.0

        d_xyz[mask] = masked_d_xyz
        d_rot[mask] = masked_d_rot
        
        return mask, d_xyz, d_rot   
        
