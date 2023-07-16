import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from custom.nn.modules import (Conv, Conv2, Bottleneck, C2f, SPPF, Concat, DFL)
from .commons import check_version, autopad, dist2bbox, bbox2dist, make_anchors

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.i = 1
        self.f = [15, 18, 21]
        self.t = 'custom.nn.yolov8s.Detect'
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

     
        
class Yolov8s(nn.Module):
    def __init__(self, ch=3):  # model, input channels, number of classes
        super(Yolov8s, self).__init__()
        self.i = 0
        self.f = -1
        self.t = 'custom.nn.yolov8s.Yolov8s'
        self.conv1 = Conv(ch, 32, k=3, s=2)
        self.conv2 = Conv(32, 64, k=3, s=2)
        self.c2f_1 = C2f(64, 64, n=1, shortcut=True)
        self.conv3 = Conv(64, 128, k=3, s=2)
        self.c2f_2 = C2f(128, 128, n=2, shortcut=True)
        self.conv4 = Conv(128, 256, k=3, s=2)
        self.c2f_3 = C2f(256, 256, n=2, shortcut=True)
        self.conv5 = Conv(256, 512, k=3, s=2)
        self.c2f_4 = C2f(512, 512, n=1, shortcut=True)
        self.sppf = SPPF(512, 512, k=5)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.cat1 = Concat(dimension=1)
        self.c2f_5 = C2f(768, 256, n=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.cat2 = Concat(dimension=1)
        self.c2f_6 = C2f(384, 128, n=1)
        self.conv6 = Conv(128, 128, k=3, s=2)
        self.cat3 = Concat(dimension=1)
        self.c2f_7 = C2f(384, 256, n=1)
        self.conv7 = Conv(256, 256, k=3, s=2)
        self.cat4 = Concat(dimension=1)
        self.c2f_8 = C2f(768, 512, n=1)
        #self.detect = Detect(nc=4, ch=[128, 256, 512])
        
    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv2(x0)
        x2 = self.c2f_1(x1)
        x3 = self.conv3(x2)
        x4 = self.c2f_2(x3)
        x5 = self.conv4(x4)
        x6 = self.c2f_3(x5)
        x7 = self.conv5(x6)
        x8 = self.c2f_4(x7)
        x9 = self.sppf(x8)
        x10 = self.up1(x9)
        x11 = self.cat1([x10, x6])
        x12 = self.c2f_5(x11)
        x13 = self.up2(x12)
        x14 = self.cat2([x13, x4])
        x15 = self.c2f_6(x14)
        x16 = self.conv6(x15)
        x17 = self.cat3([x16, x12])
        x18 = self.c2f_7(x17)
        x19 = self.conv7(x18)
        x20 = self.cat4([x19, x9])
        x21 = self.c2f_8(x20)
        #x22 = self.detect([x15, x18, x21])
        return [x15, x18, x21]
    
  
if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Yolov8s(ch=3).to(device)
    #summary(model, (3, 640, 640))
    img = torch.rand(1, 3, 640, 640).to(device)
    y = model(img)
    print(len(y), y[0].shape, y[1].shape, y[2].shape)
    # 3 torch.Size([1, 68, 80, 80]) torch.Size([1, 68, 40, 40]) torch.Size([1, 68, 20, 20])
    #print(model)

