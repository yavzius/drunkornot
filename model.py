# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class IRClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=False)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.fc = nn.Identity()
        self.cnn = base

        self.angle_embed = nn.Embedding(4, 16)
        self.time_embed = nn.Embedding(4, 16)

        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, angle, time):
        x_feat = self.cnn(x)
        angle_feat = self.angle_embed(angle)
        time_feat = self.time_embed(time)
        x = torch.cat([x_feat, angle_feat, time_feat], dim=1)
        return self.classifier(x).squeeze(1)
