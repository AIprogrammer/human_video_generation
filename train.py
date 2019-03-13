### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from collections import OrderedDict
import torch.nn.functional as functional
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
from torch.autograd import Variable

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    for i, data in enumerate(dataset, start=epoch_iter):
        data["input"] = data["input"].permute(1, 0, 2, 3, 4)
        data["target"] = data["target"].permute(1, 0, 2, 3, 4)
        data["previous_frame"] = data["previous_frame"].permute(1, 0, 2, 3, 4)
        data["grid"] = data["grid"].permute(1, 0, 2, 3, 4)

        iter_start_time = time.time()
        epoch_iter += opt.batchSize
        total_steps += opt.batchSize
        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        grid_set = torch.cat([data['input'][0], data['grid'][0]], dim = 1)
        ############## Forward Pass ######################
        losses, generated, grid, grid_output, grid_normal = model(data['input'][0], data['target'][0], data['previous_frame'][0],
                                  data['grid'][0], grid_set, infer=save_fake)

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)


        ### display output images
        # if save_fake:
        #     img = (data['previous_frame'][0]).cuda()
        #     grid = grid.permute(0, 3, 1, 2).detach()
        #     grid = functional.interpolate(grid, (256,256), mode='bilinear')
        #     grid_output = functional.interpolate(grid_output.detach(), (256,256), mode='bilinear')
        #     warp = functional.grid_sample(img, grid.permute(0, 2, 3, 1), padding_mode='reflection')
        #     warp_2 = functional.grid_sample(img, grid_output.permute(0, 2, 3, 1), padding_mode='reflection')
        #     warp_3 = functional.grid_sample(img, grid_normal.permute(0, 2, 3, 1), padding_mode='reflection')
        #     visuals = OrderedDict([('synthesized_video', util.tensor2im(generated.data[0])),
        #                            ('real_video', util.tensor2im(data['target'][0][0])),
        #                            ('warped_previous_image', util.tensor2im(warp[0])),
        #                            ('warped_previous_image_2', util.tensor2im(warp_2[0])),
        #                            ('warped_previous_image_3', util.tensor2im(warp_3[0]))])
        #     visualizer.display_current_results(visuals, epoch, total_steps)
        for f in range(1, 3):
            grid_set = torch.cat([data['input'][f], data['grid'][f]], dim=1)
            ############## Forward Pass ######################
            losses, generated, grid, grid_output, grid_normal = model(data['input'][f], data['target'][f],
                                                                      generated,
                                                                      data['grid'][f],
                                                                      grid_set, infer=save_fake)

            # sum per device losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_D += (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G += loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)

            ### display output images
            if save_fake:
                visuals = OrderedDict([('synthesized_video_%d'%f, util.tensor2im(generated.data[0])),
                                       ('real_video_%d'%f, util.tensor2im(data['target'][f][0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)
                ############### Backward Pass ####################

        # update generator weights
        model.module.optimizer_G.zero_grad()
        loss_G.backward()
        model.module.optimizer_G.step()

        # update discriminator weights
        model.module.optimizer_D.zero_grad()
        loss_D.backward()
        model.module.optimizer_D.step()
        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            for key, value in (data['paths']).iteritems():
                print key + "      " + value[0]

        if epoch_iter >= dataset_size:
            break
       
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()


    #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

        ############## Display results and errors ##########
        ### print out errors
        #if total_steps % opt.print_freq == print_delta:
         #   errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
          #  t = (time.time() - iter_start_time) / opt.batchSize
           # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            #visualizer.plot_current_errors(errors, total_steps)