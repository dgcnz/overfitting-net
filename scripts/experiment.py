import urllib

import urllib.request

import mlflow
import torchvision.transforms.functional as FT
import argparse
from torchvision.io import read_video
from PIL import Image
from overfit.utils.img2vid import zigzag
from overfit.trainers.overfit import OverfitTrainer
from torchvision.models import ResNet34_Weights, resnet34

parser = argparse.ArgumentParser(description="")
# parser.add_argument("video_path", type=str, help="Video path.")
parser.add_argument("crop_fraction", type=int, help="Crop fraction")
parser.add_argument("max_length", type=int, help="Max length")

args = parser.parse_args()


def normalize_rgb(img):
    return FT.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


input_image = Image.open("dog.jpg")
input_tensor = FT.to_tensor(input_image)
_, h, w = input_tensor.size()

crop_fraction = args.crop_fraction
hcrop = h // crop_fraction
wcrop = w // crop_fraction
input_video = zigzag(input_tensor, hcrop, wcrop, args.max_length)

# print(input_video.unsqueeze())

input_video = [normalize_rgb(frame).unsqueeze(0) for frame in input_video]
srcnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).eval()
tgtnet_trainer = OverfitTrainer()
tgtnet_trainer.set(
    pretrained_classifier=srcnet,
    num_classes=1000,
    confidence=0.5,
    weight_decay=0.9,
    max_lr=0.1,
    momentum=0.9,
)
mlflow.set_tracking_uri("http://localhost:5050")
with mlflow.start_run(experiment_id="0") as run:
    # mlflow.pytorch.log_model(
    #     tgtnet_trainer.model,
    #     artifact_path="overfitmodel",
    #     registered_model_name="overfit",
    # )
    mlflow.log_param("Crop fraction", crop_fraction)
    mlflow.log_param("Frames", len(input_video))
    tgtnet_trainer.test(input_video, [258] * len(input_video))
