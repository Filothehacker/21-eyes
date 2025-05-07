import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import json
import os
from torchsummary import summary

cwd = os.getcwd()
model_config_path = os.path.join(cwd, "configurations", "yolo_v1.json")

with open(model_config_path, "r") as f:
    model_config = json.load(f)
    # Retrieve the parameters
MODEL_PARAMS = model_config["MODEL_PARAMS"]


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = tuple(args)
    def forward(self, x):
        return torch.reshape(x, (-1, *self.shape))


    

class DetectionNet(nn.Module):
    """The layers added on for detection as described in the paper."""

    def __init__(self, in_channels):
        super().__init__()

        inner_channels = 1024
        self.depth = 5 * MODEL_PARAMS["B"] + MODEL_PARAMS["C"]
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=2, padding=1),   # (Ch, 14, 14) -> (Ch, 7, 7)
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Flatten(),

            nn.Linear(7 * 7 * inner_channels, 4096),
            # nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Linear(4096, MODEL_PARAMS["S"] * MODEL_PARAMS["S"] * self.depth)
        )

    def forward(self, x):
        return torch.reshape(
            self.model.forward(x),
            (-1, MODEL_PARAMS["S"],MODEL_PARAMS["S"], self.depth)
        )


class YOLOv1ResNet(nn.Module):
    def __init__(self,model_params):
        super().__init__()
        #the configurations are in the json file in the configurations folder
        # Load configuration
        self.params = model_params
        self.depth = MODEL_PARAMS["B"]* 5 +  MODEL_PARAMS["C"]

        # Load backbone ResNet
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.requires_grad_(False)            # Freeze backbone weights

        # Delete last two layers and attach detection layers
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()

        self.model = nn.Sequential(
            backbone,
            Reshape(2048, 14, 14),
            DetectionNet(2048)              # 4 conv, 2 linear
        )

    def forward(self, x):
        return self.model.forward(x)
    

if __name__ == "__main__":
    # Test the model
    model = YOLOv1ResNet(MODEL_PARAMS)
    x = torch.randn(1, 3, 448, 448)
    y = model(x)
    print(y.shape)  # Should be (1, S, S, B*5 + C)
    summary(model, (3, 448, 448), device="cpu")

    str_arch_path = os.path.join(cwd, "configurations", "yolo_v1_architecture.txt")
    with open(str_arch_path, "w") as f:
        f.write(str(model))