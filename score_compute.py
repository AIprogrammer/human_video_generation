import os
from argparse import ArgumentParser

from inception_score import get_inception_score
from skimage.io import imread, imsave
from skimage.measure import compare_ssim
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from data.image_folder import natural_keys, make_dataset


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from frefrechet_video_distance.frechet_video_distance import calculate_fvd, create_id3_embedding, preprocess


def l1_score(generated_images, reference_images):
    score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        score = np.abs(2 * (reference_image/255.0 - 0.5) - 2 * (generated_image/255.0 - 0.5)).mean()
        score_list.append(score)
    return np.mean(score_list)


def ssim_score(generated_images, reference_images):
    ssim_score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        ssim = compare_ssim(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, multichannel=True,
                            data_range=generated_image.max() - generated_image.min())
        ssim_score_list.append(ssim)
    return np.mean(ssim_score_list)


def fvd(generated_images, reference_images):
  with tf.Graph().as_default():

    result = calculate_fvd(
        create_id3_embedding(preprocess(first_set_of_videos,
                                                (256, 256))),
        create_id3_embedding(preprocess(second_set_of_videos,
                                                (256, 256))))

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("FVD is: %.2f." % sess.run(result))


def load_generated_images_fvd(images_folder_fake, images_folder_real, length):
    target_images = []
    generated_images = []

    sample_folders = os.listdir(images_folder_real)
    sample_folders.sort(key=natural_keys)

    for folder in sample_folders:
        current_path_real = os.path.join(opt.dataroot, folder)
        current_path_fake = os.path.join(opt.dataroot, folder)
        real_images = (make_dataset(current_path_real))
        fake_images = (make_dataset(current_path_fake))
        real_images.sort(key=natural_keys)
        fake_images.sort(key=natural_keys)
        real_video = []
        fake_video = []
        i = 0
        for real_img_name, fake_img_name in zip(real_images, fake_images):
            if i >= length:
                break
            print real_img_name
            print fake_img_name
            img_fake = Image.open(fake_img_name)
            img_tensor_fake = img_fake.convert('RGB')
            fake_video.append(np.array(img_tensor_fake))

            img = Image.open(real_img_name)
            img_tensor = img.convert('RGB')
            real_video.append(np.array(img_tensor))
            i += 1
        generated_images.append(np.stack(fake_video, axis=0))
        target_images.append(np.stack(real_video, axis=0))
    target_images = np.stack(target_images, axis = 0)
    generated_image = np.stack(genertaed_images, axis = 0)
    return  target_images, generated_images

def load_generated_images(images_folder_fake, images_folder_real):
    target_images = []
    generated_images = []

    sample_folders = os.listdir(images_folder_real)
    sample_folders.sort(key=natural_keys)

    for folder in sample_folders:
        current_path_real = os.path.join(opt.dataroot, folder)
        current_path_fake = os.path.join(opt.dataroot, folder)
        real_images = (make_dataset(current_path_real))
        fake_images = (make_dataset(current_path_fake))
        real_images.sort(key=natural_keys)
        fake_images.sort(key=natural_keys)
        for real_img_name, fake_img_name in zip(real_images, fake_images):
            print real_img_name
            print fake_img_name
            img_fake = Image.open(fake_img_name)
            img_tensor_fake = img_fake.convert('RGB')
            generated_images.append(np.array(img_tensor_fake))

            img = Image.open(real_img_name)
            img_tensor = img.convert('RGB')
            target_images.append(np.array(img_tensor))

    return  target_images, generated_images

def test():
    parser = ArgumentParser()
    parser.add_argument("--load_generated_images",  action='store_true', default=True,
                    help="loading generated images")

    parser.add_argument( "--images_folder_fake",  type=str, default= " ")
    parser.add_argument( "--images_folder_real",  type=str, default= " ")
    parser.add_argument( "--length_of_videos_fvd",  type=int, default= 128)
    args = parser.parse_args()

    if args.load_generated_images:
        print ("Loading images...")
        target_images, generated_images = load_generated_images(args.images_folder_fake, args.images_folder_real)

    print ("Compute structured similarity score (SSIM)...")
    structured_score = ssim_score(generated_images, target_images)
    print ("SSIM score %s" % structured_score)

    print ("Compute l1 score...")
    norm_score = l1_score(generated_images, target_images)
    print ("L1 score %s" % norm_score)

    if args.load_generated_images:
        print ("Loading images to compute fvd score...")
        target_images, generated_images = load_generated_images(args.images_folder_fake, args.images_folder_real, args.length_of_videos_fvd)

    print ("Compute FVD score...")
    norm_score = fvd(generated_images, target_images)

    print (SSIM score = %s, l1 score = %s" %
           (structured_score, norm_score))



if __name__ == "__main__":
    test()
