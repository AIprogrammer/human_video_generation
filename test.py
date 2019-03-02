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

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
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
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx
    
for i, data in enumerate(dataset):
    with torch.no_grad():
        if i >= opt.how_many:
            break
        data["dp_target"] = data["dp_target"].permute(1, 0, 2, 3, 4)
        data["target"] = data["target"].permute(1, 0, 2, 3, 4)
        data["texture"] = data["texture"].permute(1, 0, 2, 3, 4)

        generated_video = []
        real_video = []

        for i in range(0, data["dp_target"].shape[0]):

            label_tensors = []
            for folder in opt.multinput:
                if data[folder].dim() == 5:
                    label_tensors.append(data[folder][i])
                else:
                    label_tensors.append(data[folder])

            for lt in label_tensors:
                print lt.size()

            label = torch.cat(label_tensors, dim=1)

            generated = model.inference(label, None)
            print generated.size()

            visuals = OrderedDict([('synthesized_image', util.tensor2im(generated.data[0]))])
            img_path = data['path'][i]
            print('process image... %s' % img_path)
            visualizer.save_images(webpage, visuals, img_path[0].rsplit("/",1)[0] + str(i))
            visualizer.display_current_results(visuals, 100, 12345)


webpage.save()
