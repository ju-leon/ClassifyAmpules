from PIL import Image, ImageFilter
import PIL
import time
import os
import tqdm
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Combine up to 3 images into a single image to reduce data size')

    parser.add_argument('in_dir')
    parser.add_argument('out_dir')

    args = parser.parse_args()
    path = args.in_dir


    files = set()
    valid_images = ".png"

    for f in tqdm.tqdm(os.listdir(path)):
        if f.endswith(valid_images):
            files.add(f.split("_")[0])

    for file in tqdm.tqdm(files):
        try:
            image1 = PIL.Image.open(path + file + "_1.png").convert('L')
        except:
            continue

        try:
            image2 = PIL.Image.open(path + file + "_2.png").convert('L')
        except:
            image2 = image1

        try:
            image3 = PIL.Image.open(path + file + "_2.png").convert('L')
        except:
            image3 = image1

        for x in range(image1.size[0]):
            for y in range(image1.size[1]):
                image1.putpixel((x, y), min(image1.getpixel((x, y)), image2.getpixel((x, y)), image3.getpixel((x, y))))

        image1.save(args.out_dir + file + ".png", "PNG")


if __name__ == "__main__":
    main()
