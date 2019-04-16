
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


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
from imageio import mimread

from skimage import io, img_as_float32
from skimage.color import gray2rgb

import tensorflow as tf
from frechet_video_distance import calculate_fvd, create_id3_embedding, preprocess


def l1_score(generated_images, reference_images):
    score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        score = np.abs(2 * (reference_image/255.0 - 0.5) - 2 * (generated_image/255.0 - 0.5)).mean()
        score_list.append(score)
    return np.mean(score_list)


def ssim_score(generated_images, reference_images):
    ssim_score_list = []
    len_data = len(generated_images)
    counter = 1
    for reference_image, generated_image in zip(reference_images, generated_images):
        if counter % 1000 == 0:
		 print ("Computing SSIM for the  video %d/%d" %(counter, len_data))
        ssim = compare_ssim(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, multichannel=True,
                            data_range=generated_image.max() - generated_image.min())
        ssim_score_list.append(ssim)
        counter += 1
    return np.mean(ssim_score_list)


def fvd(generated_images, reference_images):
  with tf.Graph().as_default():
    print (generated_images.shape)
    print (reference_images.shape)
    print (type(generated_images))
    print (type(reference_images))
    generated_images = tf.convert_to_tensor(generated_images, dtype=np.uint8)
    print("Converted to tensor generated_images ")
    reference_images = tf.convert_to_tensor(reference_images, dtype=np.uint8)

    print("Converted to tensor reference images")
    print (tf.shape(generated_images))
    print (tf.shape(reference_images))
    result = calculate_fvd(
        create_id3_embedding(preprocess(generated_images,
                                                (256, 256))),
        create_id3_embedding(preprocess(reference_images,
                                                (256, 256))))

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("FVD is: %.2f." % sess.run(result))


def read_video(name, image_shape, length = 0, fvd = False):
    image_shape = tuple(image_shape)
    if name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + image_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    if not length == 0:
        video_array = video_array[:length]
    if not fvd:
        video_array = list(video_array)

    return video_array


def load_generated_images_from_one_img_fvd(images_folder_fake, images_folder_real, length):
    target_images = []
    generated_images = []

    sample_folders = os.listdir(images_folder_real)
    sample_folders.sort(key=natural_keys)
    new_len  = (len(sample_folders) // 16) * 16
    sample_folders = sample_folders[:new_len]

    for folder in sample_folders:
        current_path_real = os.path.join(images_folder_real, folder + "/target")
        real_images = (make_dataset(current_path_real))
        real_images.sort(key=natural_keys)

        fake_video = read_video(name, (256,256,3), length)
        for real_img_name, fake_frame in zip(real_images, fake_video):
            #print (real_img_name)
            generated_images.append(fake_frame)

            img = Image.open(real_img_name)
            img_tensor = img.convert('RGB')
            target_images.append(np.array(img_tensor))

    return  target_images, generated_images



def load_generated_images_fvd(images_folder_fake, images_folder_real, length):
    target_images = []
    generated_images = []

    sample_folders = os.listdir(images_folder_real)
    sample_folders.sort(key=natural_keys)
    new_len  = 16 #(len(sample_folders) // 16) * 16
    print (new_len)
    sample_folders = sample_folders[:new_len]
    for folder in sample_folders:
        current_path_real = os.path.join(images_folder_real, folder + "/target")
        current_path_fake = os.path.join(images_folder_fake,  folder)
        real_images = (make_dataset(current_path_real))
        fake_images = (make_dataset(current_path_fake))
        real_images.sort(key=natural_keys)
        fake_images.sort(key=natural_keys)
        real_video = []
        fake_video = []

        if not length == 0:
            real_images = real_images[:length]
            fake_images  = fake_images[:length]

        for real_img_name, fake_img_name in zip(real_images, fake_images):

            img_fake = Image.open(fake_img_name)
            img_tensor_fake = img_fake.convert('RGB')
            fake_video.append(np.array(img_tensor_fake))

            img = Image.open(real_img_name)
            img_tensor = img.convert('RGB')
            real_video.append(np.array(img_tensor))

        generated_images.append(np.stack(fake_video, axis=0))
        target_images.append(np.stack(real_video, axis=0))
    target_images = np.stack(target_images, axis = 0)
    generated_images = np.stack(generated_images, axis = 0)
    return  target_images, generated_images




def load_generated_images(images_folder_fake, images_folder_real, length = 0):
    target_images = []
    generated_images = []


    sample_folders = os.listdir(images_folder_real)
    sample_folders.sort(key=natural_keys)

    for folder in sample_folders:
        current_path_real = os.path.join(images_folder_real,  folder + "/target")
        current_path_fake = os.path.join(images_folder_fake,  folder)
        real_images = (make_dataset(current_path_real))
        fake_images = (make_dataset(current_path_fake))
        real_images.sort(key=natural_keys)
        fake_images.sort(key=natural_keys)
        if not length == 0:
            real_images = real_images[:length]
            fake_images  = fake_images[:length]
        for real_img_name, fake_img_name in zip(real_images, fake_images):

            img_fake = Image.open(fake_img_name)
            img_tensor_fake = img_fake.convert('RGB')
            generated_images.append(np.array(img_tensor_fake))

            img = Image.open(real_img_name)
            img_tensor = img.convert('RGB')
            target_images.append(np.array(img_tensor))

    return  target_images, generated_images


def load_generated_images_from_one_img(images_folder_fake, images_folder_real, length = 0):
    target_images = []
    generated_images = []

    sample_folders = os.listdir(images_folder_real)
    sample_folders.sort(key=natural_keys)

   
    fake_image_path = (make_dataset(images_folder_fake))

    for folder, fake_path in zip(sample_folders, fake_image_path):
        current_path_real = os.path.join(images_folder_real, folder + "/target")
        real_images = (make_dataset(current_path_real))
        real_images.sort(key=natural_keys)
        fake_video = read_video(fake_path, (256, 256, 3), length)
        if not length == 0:
            real_images = real_images[:length]
	for real_img_name, fake_frame in zip(real_images, fake_video):
            generated_images.append(fake_frame)

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
    parser.add_argument( "--load_from_one_image",  action='store_true', default=False,
                    help="loading video frames from one single image")
    parser.add_argument("--load_short_for_ssim",  action='store_true', default=False,
                    help="loading less videos for ssim")
    args = parser.parse_args()

    if args.load_generated_images:
        print ("Loading images...")
        if args.load_from_one_image:
            target_images, generated_images = load_generated_images_from_one_img(args.images_folder_fake, args.images_folder_real, args.length_of_videos_fvd if args.load_short_for_ssim else 0)
        else:
            target_images, generated_images = load_generated_images(args.images_folder_fake, args.images_folder_real, args.length_of_videos_fvd if args.load_short_for_ssim else 0)

    print ("Length of the datasets: %d %d" %(len(target_images), len(generated_images)))

    print ("Compute structured similarity score (SSIM)...")
    structured_score = ssim_score(generated_images, target_images)
    print ("SSIM score %s" % structured_score)

    print ("Compute l1 score...")
    norm_score = l1_score(generated_images, target_images)
    print ("L1 score %s" % norm_score)

   # if args.load_generated_images:
   #     print ("Loading images to compute fvd score...")
   #     target_images, generated_images = load_generated_images_fvd(args.images_folder_fake, args.images_folder_real, args.length_of_videos_fvd)

   # print ("Compute FVD score...")
  #  norm_score = fvd(generated_images, target_images)

   # print ("SSIM score = %s, l1 score = %s" %
   #        (structured_score, norm_score))



if __name__ == "__main__":
    test()
