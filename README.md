<br><br><br><br>

# Generating human motion videos

## Prerequisites
- Linux or macOS
- Python 2 or 3
- NVIDIA GPU (12G or 24G memory) + CUDA cuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install python libraries [dominate](https://github.com/Knio/dominate).
```bash
pip install dominate
```

### Testing

- Test the model (`bash ./scripts/test_1024p.sh`):
```bash
python test.py --name posedata --dataroot ./datasets/MVC_pix2pix_validation  --label_nc 0 --no_instance --nThreads 1 --data_type 32 --loadSize 256 --multinput source dp_target dp_source texture --input_nc 12 --resize_or_crop resize_and_crop --phase val --how_many 1000 --which_epoch 60
```
The test results will be saved to a html file here: `./results/posedata/test_latest/index.html`.

### Dataset
For an example of the dataset of a shart video clip please refer to this link: 


### Training
- Train a model at 1024 x 512 resolution (`bash ./scripts/train_512p.sh`):
```bash
#!./scripts/train_512p.sh
python train.py --name label2city_512p
```
- To view training results, please checkout intermediate results in `./checkpoints/posedata/web/index.html`.
If you have tensorflow installed, you can see tensorboard logs in `./checkpoints/posedata/logs` by adding `--tf_log` to the training scripts.

### Multi-GPU training
- Train a model using multiple GPUs (`bash ./scripts/train_512p_multigpu.sh`):
```bash
#!./scripts/train_512p_multigpu.sh
python train.py --name label2city_512p --batchSize 8 --gpu_ids 0,1,2,3,4,5,6,7
```
Note: this is not tested and we trained our model using single GPU only. Please use at your own discretion.

### Training at full resolution
- To train the images at full resolution (2048 x 1024) requires a GPU with 24G memory (`bash ./scripts/train_1024p_24G.sh`).
If only GPUs with 12G memory are available, please use the 12G script (`bash ./scripts/train_1024p_12G.sh`), which will crop the images during training. Performance is not guaranteed using this script.

### Training with your own dataset
- The default setting for preprocessing is `scale_width`, which will scale the width of all training images to `opt.loadSize` (1024) while keeping the aspect ratio. If you want a different setting, please change it by using the `--resize_or_crop` option. For example, `scale_width_and_crop` first resizes the image to have width `opt.loadSize` and then does random cropping of size `(opt.fineSize, opt.fineSize)`. `crop` skips the resizing step and only performs random cropping. If you don't want any preprocessing, please specify `none`, which will do nothing other than making sure the image is divisible by 32.


## Acknowledgments
This code is based on Pix2PixHD model (https://github.com/NVIDIA/pix2pixHD).
