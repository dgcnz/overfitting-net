import urllib
import urllib.request

import mlflow
import torchvision.transforms.functional as FT
from overfit.trainers.overfit import OverfitTrainer
from overfit.utils.img2vid import zigzag
from PIL import Image
from torchvision.models import ResNet34_Weights, resnet34

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try:
    urllib.request.URLopener().retrieve(url, filename)
except:  # noqa
    urllib.request.urlretrieve(url, filename)

input_image = Image.open("dog.jpg")
input_tensor = FT.to_tensor(input_image)
_, h, w = input_tensor.size()
input_video = zigzag(input_tensor, h // 3, w // 3)
# vid = display_video(input_video)
# vid.save("dog.mp4")
print(len(input_video))


def normalize_rgb(img):
    return FT.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


input_video = [normalize_rgb(frame).unsqueeze(0) for frame in input_video]
srcnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).eval()
tgtnet_trainer = OverfitTrainer()
tgtnet_trainer.set(
    pretrained_classifier=srcnet,
    num_classes=1000,
    confidence=0.1,
    weight_decay=10,
    max_lr=0.1,
    momentum=0.9,
)
mlflow.set_tracking_uri("http://localhost:5050")
with mlflow.start_run(experiment_id="0") as run:
    mlflow.pytorch.log_model(
        tgtnet_trainer.model,
        artifact_path="overfitmodel",
        registered_model_name="overfit",
    )
    tgtnet_trainer.test(input_video, [258] * len(input_video))
