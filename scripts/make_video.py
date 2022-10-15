import urllib
from overfit.utils.img2vid import display_video
import urllib.request

import torchvision.transforms.functional as FT
from overfit.utils.img2vid import zigzag
import matplotlib.pyplot as plt
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("crop_fraction", type=int, help="Crop fraction")
parser.add_argument("max_length", type=int, help="Max length")

args = parser.parse_args()

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try:
    urllib.request.URLopener().retrieve(url, filename)
except:  # noqa
    urllib.request.urlretrieve(url, filename)

input_image = Image.open("dog.jpg")
input_tensor = FT.to_tensor(input_image)
_, h, w = input_tensor.size()

crop_fraction = args.crop_fraction
hcrop = h // crop_fraction
wcrop = w // crop_fraction
input_video = zigzag(input_tensor, hcrop, wcrop, args.max_length)

# print(len(input_video))
vid = display_video(input_video)
# plt.show()
# x = input("Do you wish to save this video?")
# if x == "yes":
vid.save(f"data/samoyed_{crop_fraction}_{len(input_video)}.mp4")