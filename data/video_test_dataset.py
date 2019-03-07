import os.path
import torch
from data.base_dataset import BaseDataset, get_params, get_transform, normalize, get_optical_flow_transform
from data.image_folder import make_dataset, atoi, natural_keys
from PIL import Image
import numpy as np


class Videos_Test_Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.input_paths = []

        sample_folders = os.listdir(opt.dataroot)
        for folder in sample_folders:
            current_path = os.path.join(opt.dataroot, folder)
            dp_target_folders = (make_dataset(os.path.join(current_path, "dp_target")))
            target_folders = (make_dataset(os.path.join(current_path, "target")))
            dp_source_folders = (make_dataset(os.path.join(current_path, "dp_source")))
            source_folders = (make_dataset(os.path.join(current_path, "source")))
            of_x = (make_dataset(os.path.join(current_path, "of_x")))
            of_y = (make_dataset(os.path.join(current_path, "of_y")))
            of_x_source = (make_dataset(os.path.join(current_path, "of_x_source")))
            of_y_source = (make_dataset(os.path.join(current_path, "of_y_source")))
            dp_target_folders.sort(key=natural_keys)
            target_folders.sort(key=natural_keys)
            dp_source_folders.sort(key=natural_keys)
            source_folders.sort(key=natural_keys)
            of_x.sort(key=natural_keys)
            of_y.sort(key=natural_keys)
            of_x_source.sort(key=natural_keys)
            of_y_source.sort(key=natural_keys)

            for i in range(0, opt.source_num):
                texture = (make_dataset(os.path.join(current_path, "texture%d"%i)))
                texture.sort(key=natural_keys)
                self.input_paths.append({'dp_target': dp_target_folders, 'target':  target_folders,
                                         'dp_source': dp_source_folders[i],
                                         'source': source_folders[i],
                                         'of_x':  of_x,
                                         'of_y': of_y,
                                         'of_x_source': of_x_source[i],
                                         'of_y_source': of_y_source[i],
                                         'texture': texture,
                                         'path': folder+"_" + str(i)})
        self.dataset_size = len(sample_folders*opt.source_num)

    def __getitem__(self, index):
        dp_target_video = []
        target_video = []
        texture_video = []
        current_paths = self.input_paths[index]

        for dp_target_path in current_paths['dp_target']:
            img = Image.open(dp_target_path)
            params = get_params(self.opt, img.size)
            transform_img = get_transform(self.opt, params)
            img_tensor = transform_img(img.convert('RGB'))
            dp_target_video.append(img_tensor)

        for target_path in current_paths['target']:
            img = Image.open(target_path)
            params = get_params(self.opt, img.size)
            transform_img = get_transform(self.opt, params)
            img_tensor = transform_img(img.convert('RGB'))
            target_video.append(img_tensor)

        for texture_path in current_paths['texture']:
            img = Image.open(texture_path)
            params = get_params(self.opt, img.size)
            transform_img = get_transform(self.opt, params)
            img_tensor = transform_img(img.convert('RGB'))
            texture_video.append(img_tensor)


        dp_target = torch.stack(dp_target_video, 0)
        target = torch.stack(target_video, 0)
        texture = torch.stack(texture_video, 0)

        dp_source = Image.open(current_paths['dp_source'])
        params = get_params(self.opt, dp_source.size)
        transform_img = get_transform(self.opt, params)
        dp_source_tensor = transform_img(dp_source.convert('RGB'))

        source = Image.open(current_paths['source'])
        params = get_params(self.opt, source.size)
        transform_img = get_transform(self.opt, params)
        source_tensor = transform_img(source.convert('RGB'))

        img_Y = Image.open(current_paths['of_x_source'])
        img_X = Image.open(current_paths['of_y_source'])

        img_X = np.array(img_X)
        img_Y = np.array(img_Y)

        t_X, t_Y = get_optical_flow_transform(np.shape(img_Y))

        images = [t_X(img_X[..., np.newaxis].astype('float32')), t_Y(img_Y[..., np.newaxis].astype('float32'))]
        grid = torch.stack(images, dim=1)
        grid_source = grid.squeeze(dim=0)


        grid_tensors = []
        for of_x_path, of_y_path in zip(current_paths['of_x'], current_paths['of_y']):
            img_Y = Image.open(of_x_path)
            img_X = Image.open(of_y_path)

            img_X = np.array(img_X)
            img_Y = np.array(img_Y)

            t_X, t_Y = get_optical_flow_transform(np.shape(img_Y))

            images = [t_X(img_X[..., np.newaxis].astype('float32')), t_Y(img_Y[..., np.newaxis].astype('float32'))]
            grid = torch.stack(images, dim=1)
            grid = grid.squeeze(dim=0)
            grid_tensors.append(grid)

        grid_tensor = torch.stack(grid_tensors, 0)


        input_dict = {'dp_target': dp_target, 'target': target, 'texture': texture,
                      'dp_source': dp_source_tensor, 'source': source_tensor,
                      'grid': grid_tensor, 'grid_source': grid_source, 'path':  current_paths['path']}

        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'Videos_Test_Dataset'