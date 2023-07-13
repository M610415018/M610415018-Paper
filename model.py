import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(2048, 512)
        
        # 添加ResNet殘差塊
        self.residual_block = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=1),
            nn.BatchNorm2d(2048),
        )

    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        # 使用殘差塊
        residual = x
        x = self.residual_block(x)
        x += residual

        x = self.resnet50.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.resnet50.fc(x)

        return x
    
class FC_Model(nn.Module):
    def __init__(self) -> None:
        super(FC_Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.resnet50 = ResNet50()
        # self.fc_layer = nn.Linear(2048, 512)
        self.fc_layer = nn.Linear(in_features = 1*1026, out_features = 1*2)
        self.new_output = torch.zeros((1, 2)).to(self.device)

    # def forward(self, images_A, images_B, images_C):
    def forward(self, images_front, images_seg, images_satellite, image_line):

        images_front = images_front.to(self.device)
        images_seg = images_seg.to(self.device)
        images_satellite = images_satellite.to(self.device)
        images_line = image_line.to(self.device)
        
        features_front = self.resnet50(images_front)
        features_seg = self.resnet50(images_seg)
        features_line = self.resnet50(images_line)

        features_fs = torch.add(features_front, features_seg)
        # features_D = torch.add(features_A, features_B)
        features_fsl = torch.add(features_fs, features_line)

        features_satellite = self.resnet50(images_satellite)
        # features_DC = torch.cat((features_C, features_D), dim=1)
        features_fsls = torch.cat((features_fsl, features_satellite), dim=1)
        input_data = torch.cat((self.new_output, features_fsls), dim=1)
        # input_data = torch.cat((self.new_output, features_DC), dim=1)

    # Move the fully connected layer to the device
        output = self.fc_layer(input_data)    # shape of output = (B, 2); (delta_x, delta_y)
        

        return output

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(512, 512)

        # add residual blocks
        self.residual_blocks = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # adaptive avgpool to make output shape of residual_blocks to be 512x1x1
        )

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        # apply residual blocks
        residual = self.residual_blocks(x)
        x = x + residual

        x = x.view(x.size(0), -1)
        x = self.resnet18.fc(x)

        return x