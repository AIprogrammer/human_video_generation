import os.path
import torch
from data.base_dataset import BaseDataset, get_params, get_transform, normalize, get_optical_flow_transform
from data.image_folder import make_dataset, atoi, natural_keys, DensePose
from PIL import Image
import numpy as np


class Video_Test_Dataset_Runtime(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.input_paths = []
        self.dp = DensePose((opt.loadSize, opt.loadSize), oob_ocluded=True, naive_warp = opt.naive_warp)

        sample_folders = os.listdir(opt.dataroot)
        sample_folders.sort(key=natural_keys)
        for folder in sample_folders:
            current_path = os.path.join(opt.dataroot, folder)
            dp_target_folders = (make_dataset(os.path.join(current_path, "dp_target")))
            target_folders = (make_dataset(os.path.join(current_path, "target")))
            dp_target_folders.sort(key=natural_keys)
            target_folders.sort(key=natural_keys)
            self.input_paths.append({'dp_target': dp_target_folders,
                                     'dp_source': dp_target_folders[0],
                                     'source': target_folders[0],
                                     'path': folder})
        self.dataset_size = len(sample_folders)

    def __getitem__(self, index):
        dp_target_video = []
        texture_video = []
        current_paths = self.input_paths[index]

        for dp_target_path in current_paths['dp_target']:
            img = Image.open(dp_target_path)
            params = get_params(self.opt, img.size)
            transform_img = get_transform(self.opt, params)
            img_tensor = transform_img(img.convert('RGB'))
            dp_target_video.append(img_tensor)


        dp_target = torch.stack(dp_target_video, 0)


        dp_source = Image.open(current_paths['dp_source'])
        params = get_params(self.opt, dp_source.size)
        transform_img = get_transform(self.opt, params)
        dp_source_tensor = transform_img(dp_source.convert('RGB'))

        source = Image.open(current_paths['source'])
        params = get_params(self.opt, source.size)
        transform_img = get_transform(self.opt, params)
        source_tensor = transform_img(source.convert('RGB'))

        grid_source_tensors = []
        img_source_dp = Image.open(current_paths['dp_source'])
        img_source_dp = img_source_dp.convert('RGB')
        np_source_dp = np.array(img_source_dp )
        for dp_target_path in current_paths['dp_target']:
            img = Image.open(dp_target_path)
            img_target_dp = img.convert('RGB')
            img_target_dp = np.array(img_target_dp)
            grid_source= self.dp.get_grid_warp(np_source_dp, img_target_dp)
            grid_source = torch.from_numpy(grid_source).float()
            grid_source = grid_source.permute(2,0,1)
            grid_source_tensors.append(grid_source)

        grid_source_tensor = torch.stack(grid_source_tensors, 0)


        grid_tensors = []
        for i in xrange(len(current_paths['dp_target']) - 1):
            previous_frame, current_frame =  current_paths['dp_target'][i], current_paths['dp_target'][i + 1]
            previous_frame = Image.open(previous_frame)
            previous_frame = previous_frame.convert('RGB')
            np_previous_frame = np.array(previous_frame)

            current_frame = Image.open(current_frame)
            current_frame = current_frame.convert('RGB')
            np_current_frame = np.array(current_frame)

            grid= self.dp.get_grid_warp(np_previous_frame, np_current_frame)
            grid = torch.from_numpy(grid).float()
            grid = grid.permute(2,0,1)
            grid_tensors.append(grid)

        grid_tensor = torch.stack(grid_tensors, 0)


        input_dict = {'dp_target': dp_target, 'dp_source': dp_source_tensor, 'source_frame': source_tensor,
                      'grid': grid_tensor, 'grid_source': grid_source_tensor, 'path':  current_paths['path']}

        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'Video_Test_Dataset_Runtime'
