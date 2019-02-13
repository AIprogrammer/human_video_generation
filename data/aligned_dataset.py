### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import torch
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.input_paths = []

        ### input A (label maps)
        if opt.multinput:
            for folder in opt.multinput:
                self.input_paths.append(sorted(make_dataset(os.path.join(opt.dataroot, folder))))
            self.dataset_size = len(self.input_paths[0])
        else:
            dir_A = '_A' if self.opt.label_nc == 0 else '_label'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A))
            self.dataset_size = len(self.A_paths)

            ### input B (real images)
        if opt.isTrain:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))


      
    def __getitem__(self, index):        
        ### input A (label maps)
        tensors = []
        i_paths = []
        if self.input_paths:
            for paths in self.input_paths:
                path = paths[index]
                i_paths.append(path)
                img = Image.open(path)
                params = get_params(self.opt, img.size)
                transform_img = get_transform(self.opt, params)
                img_tensor = transform_img(img.convert('RGB'))
                tensors.append(img_tensor)
        else:
            A_path = self.A_paths[index]
            A = Image.open(A_path)
            params = get_params(self.opt, A.size)
            if self.opt.label_nc == 0:
                transform_A = get_transform(self.opt, params)
                A_tensor = transform_A(A.convert('RGB'))
            else:
                transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
                A_tensor = transform_A(A) * 255.0

        #A_tensor = torch.stack(tensors, 0)
        A_tensor = torch.cat(tensors, dim=0)
        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))
        if self.opt.multinput:
            input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                          'feat': feat_tensor, 'path': i_paths}
        else:
            input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                          'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'