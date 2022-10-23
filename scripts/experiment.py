import argparse
import logging
import os

import mlflow
from overfit.trainers.overfit import OverfitTrainer
from overfit.utils.io import normalize_rgb, uint8_to_float32
from overfit.utils.misc import parse_video_filename_params
from torchvision.io import read_video
from torchvision.models import ResNet34_Weights, resnet34

MLFLOW_SERVER = os.environ["MLFLOW_SERVER"]

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description="")
parser.add_argument("video_path", type=str, help="Video path.")
parser.add_argument("--confidence", type=float, help="Confidence", default=0.5)
parser.add_argument("--weight_decay", type=float, help="Weight Decay", default=0.9)
parser.add_argument("--max_lr", type=float, help="Max Learning rate", default=0.1)
parser.add_argument("--momentum", type=float, help="Momentum", default=0.9)

args = parser.parse_args()

vid = read_video(args.video_path, output_format="TCHW")[0]
vid = uint8_to_float32(vid)
vid = normalize_rgb(vid)
crop_fraction, n_frames = parse_video_filename_params(args.video_path)
logging.info(crop_fraction)
logging.info(n_frames)
assert len(vid) == int(n_frames)

logging.info("Creating trainer")
srcnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).eval()
tgtnet_trainer = OverfitTrainer()
tgtnet_trainer.set(
    pretrained_classifier=srcnet,
    num_classes=1000,
    confidence=args.confidence,
    weight_decay=args.weight_decay,
    max_lr=args.max_lr,
    momentum=args.momentum,
)


logging.info("Starting experiment")
mlflow.set_tracking_uri(f"http://{MLFLOW_SERVER}:5050")
with mlflow.start_run(experiment_id="0") as run:
    mlflow.log_param("Crop fraction", crop_fraction)
    mlflow.log_param("Frames", n_frames)
    with open("imagenet_classes.txt", "r") as f:
        categories = f.readlines()
        categories = [cat.rstrip("\n") for cat in categories]
        tgtnet_trainer.test(vid, [258] * n_frames, categories)
