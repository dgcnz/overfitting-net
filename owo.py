import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as FT
from PIL import Image
from torchvision import transforms

from utils.img2vid import display_video, zigzag

input_image = Image.open("dog.jpg")
input_tensor = FT.to_tensor(input_image)
_, h, w = input_tensor.size()
input_video = zigzag(input_tensor, h // 4, w // 4)
display_video(input_video)


def normalize_rgb(img):
    return FT.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


input_tensor = normalize_rgb(input_tensor)
net = torchvision.models.resnet50(pretrained=True)
input_batch = input_tensor.unsqueeze(0)
out = net(input_batch)

print(F.softmax(out, dim=1))
