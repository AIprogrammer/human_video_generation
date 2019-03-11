import os.path
import torch
from data.base_dataset import BaseDataset, get_params, get_transform, normalize, get_optical_flow_transform
from data.image_folder import make_dataset, atoi, natural_keys
from PIL import Image
import numpy as np


class Videos_Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.input_paths = []
        self.dataset_size = 0

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
                self.input_paths.append({'dp_target': dp_target_folders[0], 'target':  target_folders[0],
                                         'texture': texture[0],
                                         'previous_frame': source_folders[i],
                                         'of_x': of_x_source[i], 'of_y': of_y_source[i]})
                self.dataset_size += 1
            for j in range(1, len(target_folders)):
                self.input_paths.append({'dp_target': dp_target_folders[j], 'target': target_folders[j],
                                         'texture': texture[j],
                                         'previous_frame': target_folders[j-1],
                                         'of_x': of_x[j-1], 'of_y': of_y[j-1]})
                self.dataset_size += 1

        self.dataset_size -=2

    def __getitem__(self, index):


        transform_img = get_transform(self.opt, {})
        result_dict = {'input': [], 'target': [], 'previous_frame': [], 'grid': []}
        output = {}
        output["paths"] = self.input_paths[index]
        for i in range(3):
            current_paths = self.input_paths[index+i]
            input_tensors = []
            img = Image.open(current_paths['texture'])
            img_tensor = transform_img(img.convert('RGB'))
            input_tensors.append(img_tensor)


            img = Image.open(current_paths['dp_target'])
            img_tensor = transform_img(img.convert('RGB'))
            input_tensors.append(img_tensor)

            result_dict['input'].append(torch.cat(input_tensors, dim=0))

            img = Image.open(current_paths['target'])
            img_tensor = transform_img(img.convert('RGB'))
            result_dict['target'].append(img_tensor)

            img = Image.open(current_paths['previous_frame'])
            img_tensor = transform_img(img.convert('RGB'))
            result_dict['previous_frame'].append(img_tensor)

            img_Y = Image.open(current_paths['of_x'])
            img_X = Image.open(current_paths['of_y'])

            img_X = np.array(img_X)
            img_Y = np.array(img_Y)

            t_X, t_Y = get_optical_flow_transform(np.shape(img_Y))

            images = [t_X(img_X[..., np.newaxis].astype('float32')), t_Y(img_Y[..., np.newaxis].astype('float32'))]
            grid = torch.stack(images, dim=1)
            grid = grid.squeeze(dim=0)

            result_dict['grid'].append(grid)


        for key, value in result_dict.iteritems():
            output[key] = torch.stack(value, dim = 0)

        return output

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'Videos_Dataset'