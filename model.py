import torch
import torch.nn as nn


class YoloV1(nn.Module):
    def __init__(self, s=7, b=2, c=20):
        super(YoloV1, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(192, 128, kernel_size=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),

            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),

            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),

            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),

            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),

            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),

            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),

            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        )

        self.neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(s * s * 1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, s * s * (b * 5 + c))
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.neck(out)

        return out


if __name__ == "__main__":
    model = YoloV1()

    x = torch.randn((1, 3, 448, 448))

    output = model(x)

    print(output.size())
