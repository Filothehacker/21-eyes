import os
import torch
import torch.nn as nn
from torchsummary import summary
import json
import yaml


class ConvolutionBlock(nn.Module):

    def __init__(self, in_c, channels, kernels, strides, pool):
        super(ConvolutionBlock, self).__init__()

        convolutions = [nn.Conv2d(
            in_channels=in_c,
            out_channels=channels[0],
            kernel_size=kernels[0],
            stride=strides[0],
            padding=kernels[0]//2
            ),
            nn.LeakyReLU(negative_slope=0.1)
        ]
        if len(channels) > 1:
            for i in range(len(channels)-1):
                convolutions.append(nn.Conv2d(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=kernels[i+1],
                stride=strides[i+1],
                padding=kernels[i+1]//2
                ))
                convolutions.append(nn.LeakyReLU(negative_slope=0.1))
        
        if pool:
            convolutions.append(nn.MaxPool2d(
            kernel_size=pool[0],
            stride=pool[1]
            ))

        self.convolutions = nn.Sequential(*convolutions)


    def forward(self, x):
        return self.convolutions(x)


class Mlp(nn.Module):

    def __init__(self, in_size, hidden_sizes, out_size):
        super(Mlp, self).__init__()

        fully_connected = [
            nn.Linear(
                in_features=in_size,
                out_features=hidden_sizes[0]
            ),
            nn.LeakyReLU(negative_slope=0.1)
        ]
        
        if len(hidden_sizes) > 1:
            for i in range(len(hidden_sizes)-1):
                fully_connected.append(nn.Linear(
                    in_features=hidden_sizes[i],
                    out_features=hidden_sizes[i+1]
                ))
                fully_connected.append(nn.LeakyReLU(negative_slope=0.1))

        fully_connected.append(nn.Linear(
            in_features=hidden_sizes[-1],
            out_features=out_size
        ))

        self.fully_connected = nn.Sequential(*fully_connected)
    
    
    def forward(self, x):
        return self.fully_connected(x)
    

class Darknet(nn.Module):
    def __init__(self, model_params, cnn_blocks, n_classes):
        super(Darknet, self).__init__()

        self.params = model_params
        self.cnn = nn.ModuleList(
            [
                ConvolutionBlock(
                    in_c=block["in_c"],
                    channels=block["channels"],
                    kernels=block["kernels"],
                    strides=block["strides"],
                    pool=block["pool"]
                ) for block in cnn_blocks
            ]
        )

        self.pool_flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.mlp = torch.nn.Linear(
            in_features=cnn_blocks[-1]["channels"][-1],
            out_features=n_classes
        )

        self.softmax = nn.Softmax(dim=1)
    

    def init_weights(self, *args):

        for params in self.cnn:
            if isinstance(params, nn.Conv2d) or isinstance(params, nn.Linear):
                torch.nn.init.kaiming_normal_(params.weight, mode='fan_out', nonlinearity='leaky_relu')
                if params.bias is not None:
                    torch.nn.init.zeros_(params.bias)
            elif isinstance(params, nn.BatchNorm2d):
                torch.nn.init.ones_(params.weight)
                torch.nn.init.zeros_(params.bias)


    def forward(self, x):
        for conv in self.cnn:
            x = conv(x)
        x = self.pool_flatten(x)
        x = self.mlp(x)
        
        return x
    

    def predict(self, x):
        for conv in self.cnn:
            x = conv(x)
        x = self.pool_flatten(x)
        x = self.mlp(x)
        probs = self.softmax(x)
        
        return probs


class YoloV1(nn.Module):

    def __init__(self, model_params, cnn_blocks, mlp_dict):
        super(YoloV1, self).__init__()

        self.params = model_params
        self.cnn = nn.ModuleList(
            [
                ConvolutionBlock(
                    in_c=block["in_c"],
                    channels=block["channels"],
                    kernels=block["kernels"],
                    strides=block["strides"],
                    pool=block["pool"]
                ) for block in cnn_blocks
            ]
        )

        self.pool_flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.mlp = Mlp(
            in_size=mlp_dict["in_size"],
            hidden_sizes=mlp_dict["hidden_sizes"],
            out_size=mlp_dict["out_size"]
        )
    
    
    def forward(self, x):
        for conv in self.cnn:
            x = conv(x)
        x = self.pool_flatten(x)
        x = self.mlp(x)
        x = x.view(-1, self.params["S"], self.params["S"], self.params["B"]*5 + self.params["C"])

        return x
    

if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", DEVICE)
    cwd = os.getcwd()

    # Load the model configuration file
    config_path = os.path.join(cwd, "configurations", "yolo_v1.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Retrieve the parameters
    MODEL_PARAMS = config["MODEL_PARAMS"]
    CNN_DICT = config["CNN"]
    MLP_DICT = config["MLP"]
    OUTPUT_SIZE = MODEL_PARAMS["S"]*MODEL_PARAMS["S"] * (MODEL_PARAMS["B"]*5+MODEL_PARAMS["C"])
    MLP_DICT["out_size"] = OUTPUT_SIZE

    # Load the classes
    classes_path = os.path.join(cwd, "configurations", "classes.yaml")
    with open(classes_path, "r") as f:
        classes = yaml.safe_load(f)
    CLASSES = classes["classes"]

    # Create the darknet model
    darknet = Darknet(
        model_params=MODEL_PARAMS,
        cnn_blocks=CNN_DICT,
        n_classes=len(CLASSES)
    ).to(DEVICE)
    print("Darknet")
    summary(darknet, (3, 416, 416), device=DEVICE)

    # Create the full model
    print("YoloV1")
    yolo_v1 = YoloV1(
        model_params=MODEL_PARAMS,
        cnn_blocks=CNN_DICT,
        mlp_dict=MLP_DICT
    ).to(DEVICE)

    summary(yolo_v1, (3, 416, 416), device=DEVICE)