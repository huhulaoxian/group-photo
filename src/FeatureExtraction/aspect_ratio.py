"""
This script goes through the image folder (for image indices in good_indices)
and saves the aspect ratio information in ../../Data/images_size.npy
"""

from PIL import Image
import numpy as np


def main():
    path = "../../data/GPD/"
    good_indices = list(np.load('../../data/good_indices.npy'))
    image_sizes = []

    for image_name in good_indices:
        current_image = path + image_name
        img = Image.open(current_image)
        img = np.array(img)
        print(img.shape)
        image_sizes.append([img.shape[0], img.shape[1]])
    print("共处理", len(image_sizes),"张图片尺寸")
    # np.save('../../data/image_sizes.npy', image_sizes)


if __name__ == "__main__":
    main()