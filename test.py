### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import numpy as np
import imageio

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)

    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx

video_group = 0

for i, data in enumerate(dataset):
    with torch.no_grad():
        if i >= opt.how_many:
            break

        data["dp_target"] = data["dp_target"].permute(1, 0, 2, 3, 4)
        data["grid"] = data["grid"].permute(1, 0, 2, 3, 4)
        data["grid_source"] = data["grid_source"].permute(1, 0, 2, 3, 4)

        generated_video = []
        real_video = []

        generated = model.inference(data['dp_target'][0],
                                        data['source_frame'], data['source_frame'],
                                        data['grid_source'][0], data['grid_source'][0])

        stacked_images = np.hstack(util.tensor2im(generated.data[i]) for i in range(0, 5))
        stacked_images_source = np.hstack(util.tensor2im(data['source_frame'][i]) for i in range(0, 5))
        generated_video.append(stacked_images)
        visuals = OrderedDict([('source_images', stacked_images_source),
                                ('synthesized_image', stacked_images)])
        img_path = str(0)
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)
        visualizer.display_current_results(visuals, 100, 12345)


        for i in range(1, data["dp_target"].shape[0]):


            generated = model.inference(data['dp_target'][i],
                                            data['source_frame'], generated,
                                            data['grid_source'][i], data['grid'][i-1])


            stacked_images = np.hstack(util.tensor2im(generated.data[i]) for i in range(0, 5))
            visuals = OrderedDict([('synthesized_image', stacked_images)])
            img_path = str(i)
            print('process image... %s' % img_path)
            visualizer.save_images(webpage, visuals, img_path)
            visualizer.display_current_results(visuals, 100, 12345)
            generated_video.append(stacked_images)

        imageio.mimsave(os.path.join(web_dir, 'movie%d.gif' % video_group), generated_video)
        imageio.imsave(os.path.join(web_dir, 'source%d.jpg' % video_group), stacked_images_source)
        video_group += 1

webpage.save()
