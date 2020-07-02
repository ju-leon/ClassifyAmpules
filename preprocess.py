from PIL import Image, ImageFilter
import PIL
import time
import os
import tqdm
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Preprocess raw images to be classifiable')

    parser.add_argument('--in_dir', required=True)
    parser.add_argument('--cache_dir', required=True)
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
                args.cache_dir + f.split('.')[0] + ".png", "PNG")


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
