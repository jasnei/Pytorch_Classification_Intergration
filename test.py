# !/usr/bin/python
# -*- coding: utf-8 -*-
# jasnei@163.com
import os
from torchsummary import summary
from models.resnext import resNeXt18_32x4d_SE

# model = resNeXt50_32x4d()
model = resNeXt18_32x4d_SE(num_classes=4)
summary(model, (3, 224, 224), device='cpu')
