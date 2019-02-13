import os
from argparse import ArgumentParser

from inception_score import get_inception_score
from skimage.io import imread, imsave
from skimage.measure import compare_ssim
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms


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


def load_generated_images(images_folder_fake, images_folder_real):
    target_images = []
    generated_images = []

    for img_name in os.listdir(images_folder_real):
        img_fake = Image.open(os.path.join(images_folder_fake, img_name.split(".")[0] + "_synthesized_image.jpg"))
        img_tensor_fake = img_fake.convert('RGB')
        generated_images.append(np.array(img_tensor_fake))

        img = Image.open(os.path.join(images_folder_real, img_name))
        transform_list = []
        osize = [128, 128]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transformer = transforms.Compose(transform_list)
        img_tensor = transformer(img.convert('RGB'))
        target_images.append(np.array(img_tensor))

    return  target_images, generated_images



def test():
    parser = ArgumentParser()
    parser.add_argument("--load_generated_images",  action='store_true', default=True,
                    help="loading generated images")

    parser.add_argument( "--images_folder_fake",  type=str, default= " ")
    parser.add_argument( "--images_folder_real",  type=str, default= " ")
    args = parser.parse_args()

    if args.load_generated_images:
        print ("Loading images...")
        target_images, generated_images = load_generated_images(args.images_folder_fake, args.images_folder_real)

    print ("Compute inception score...")
    inception_score = get_inception_score(generated_images)
    print ("Inception score %s" % inception_score[0])

    print ("Compute structured similarity score (SSIM)...")
    structured_score = ssim_score(generated_images, target_images)
    print ("SSIM score %s" % structured_score)

    print ("Compute l1 score...")
    norm_score = l1_score(generated_images, target_images)
    print ("L1 score %s" % norm_score)


    print ("Inception score = %s, SSIM score = %s, l1 score = %s" %
           (inception_score[0], structured_score, norm_score))



if __name__ == "__main__":
    test()
