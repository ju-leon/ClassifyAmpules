from PIL import Image, ImageFilter
import PIL
import time
import os
import tqdm
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Preprocess raw images by intelligent downscaling')

    parser.add_argument('--in_dir', required=True)
    parser.add_argument('--out_dir', required=True)

    args = parser.parse_args()

    path = args.in_dir


    valid_images = ".bmp"
    for f in tqdm.tqdm(os.listdir(path)):
        if f.endswith(valid_images):
            img = PIL.Image.open(path + f).filter(ImageFilter.MinFilter(size=9))
            # resized = img.resize((int(img.size[0]/2), int(img.size[1]/2)), PIL.Image.LANCZOS)
            img.crop((50, 100, img.size[0] - 50, img.size[1] - 100)).resize(
                (int(img.size[0] / 10), int(img.size[1] / 10)), Image.LANCZOS).save(
                args.out_dir + f.split('.')[0] + ".png", "PNG")



if __name__ == "__main__":
    main()
