import os.path
import torch
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset, atoi, natural_keys
from PIL import Image


class Videos_Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.input_paths = []

        sample_folders = os.listdir(opt.dataroot)
        for folder in sample_folders:
            current_path = os.path.join(opt.dataroot, folder)
            dp_target_folders = (make_dataset(os.path.join(current_path, "dp_target")))#.sort(key=natural_keys)
            target_folders = (make_dataset(os.path.join(current_path, "target")))#.sort(key=natural_keys)
            dp_source_folders = (make_dataset(os.path.join(current_path, "dp_source")))#.sort(key=natural_keys)
            source_folders = (make_dataset(os.path.join(current_path, "source")))#.sort(key=natural_keys)
            dp_target_folders.sort(key=natural_keys)
            target_folders.sort(key=natural_keys)
            dp_source_folders.sort(key=natural_keys)
            source_folders.sort(key=natural_keys)
            for i in range(0, opt.source_num):
                texture = (make_dataset(os.path.join(current_path, "texture%d"%i)))
                texture.sort(key=natural_keys)
                self.input_paths.append({'dp_target': dp_target_folders, 'target':  target_folders,
                                         'dp_source': dp_source_folders[i],
                                         'source': source_folders[i],
                                         'texture': texture})

        self.dataset_size = len(sample_folders*opt.source_num)

    def __getitem__(self, index):

        dp_target_video = []
        target_video = []
        texture_video = []
        current_paths = self.input_paths[index]

        transform_img = get_transform(self.opt, {})
        for dp_target_path in current_paths['dp_target']:
            img = Image.open(dp_target_path)
            img_tensor = transform_img(img.convert('RGB'))
            dp_target_video.append(img_tensor)


        for target_path in current_paths['target']:
            img = Image.open(target_path)
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
        dp_source_tensor = transform_img(dp_source.convert('RGB'))

        source = Image.open(current_paths['source'])
        source_tensor = transform_img(source.convert('RGB'))

        input_dict = {'dp_target': dp_target, 'target': target, 'texture': texture,
                      'dp_source': dp_source_tensor, 'source': source_tensor}
        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'Videos_Dataset'