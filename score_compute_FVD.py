
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


def fvd(generated_images, reference_images):
  fvd_list = []
  with tf.Graph().as_default():
    chunk_size= 8
    for i in range(0, generated_images.shape[0], chunk_size):
        generated_images_chunk = generated_images[i:i+chunk_size]
        reference_images_chunk = reference_images[i:i+chunk_size]

        generated_images_chunk = tf.convert_to_tensor(generated_images_chunk, dtype=np.uint8)
        print("Converted to tensor generated_images ")
        reference_images_chunk = tf.convert_to_tensor(reference_images_chunk, dtype=np.uint8)
        print("Converted to tensor reference images")


        result = calculate_fvd(
            create_id3_embedding(preprocess(generated_images_chunk,
                                                    (224, 224))),
            create_id3_embedding(preprocess(reference_images_chunk,
                                                    (224, 224))))

        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          sess.run(tf.tables_initializer())
          fvd_list.append(sess.run(result))
  return np.mean(fvd_list)


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
    new_len  = (len(sample_folders) // 8) * 8

    fake_image_path = (make_dataset(images_folder_fake))

    for folder, fake_path in zip(sample_folders, fake_image_path):
        current_path_real = os.path.join(images_folder_real, folder + "/target")
        real_images = (make_dataset(current_path_real))
        real_images.sort(key=natural_keys)

        if not length == 0:
            real_images = real_images[:new_len]

        fake_images = read_video(fake_path, (256,256,3), new_len)

        real_video = []
        fake_video = []
        for real_img_name, fake_frame in zip(real_images, fake_images):

            fake_fideo.append(fake_frame)

            img = Image.open(real_img_name)
            img_tensor = img.convert('RGB')
            target_images.append(np.array(img_tensor))

    return  target_images, generated_images



def load_generated_images_fvd(images_folder_fake, images_folder_real, length):
    target_images = []
    generated_images = []

    sample_folders = os.listdir(images_folder_real)
    sample_folders.sort(key=natural_keys)
    new_len  = (len(sample_folders) // 8) * 8
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


def test():
    parser = ArgumentParser()
    parser.add_argument("--load_generated_images",  action='store_true', default=True,
                    help="loading generated images")

    parser.add_argument( "--images_folder_fake",  type=str, default= " ")
    parser.add_argument( "--images_folder_real",  type=str, default= " ")
    parser.add_argument( "--length_of_videos_fvd",  type=int, default= 128)
    parser.add_argument( "--load_from_one_image",  action='store_true', default=False,
                    help="loading video frames from one single image")

    args = parser.parse_args()

    if args.load_generated_images:
        print ("Loading images to compute fvd score...")
        target_images, generated_images = load_generated_images_fvd(args.images_folder_fake, args.images_folder_real, args.length_of_videos_fvd)

    print ("Compute FVD score...")
    print("FVD is: %.2f." %  fvd(generated_images, target_images))




if __name__ == "__main__":
    test()
