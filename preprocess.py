# coding: utf-8
import torch
import numpy as np
import pandas as pd  # 用于更容易地进行csv解析
from skimage import io, transform  # 用于图像的IO和变换
import cv2

class Rescale(object):
    """将样本中的图像重新缩放到给定大小。.

    Args:
         output_size（tuple或int）：所需的输出大小。 如果是元组，则输出为
         与output_size匹配。 如果是int，则匹配较小的图像边缘到output_size保持纵横比相同。
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return image


class CenterCrop(object):
    """裁剪样本中的图像.

    Args:
       output_size（tuple或int）：所需的输出大小。 如果是int，方形裁剪是。
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        if h > new_h:
            top =int((h - new_h)/2)
        else:
            top = 0
        if w > new_w:
            left = int((w - new_w)/2)
        else:
            left = 0

        image = image[top: top + new_h,
                      left: left + new_w]

        return image


class HorizontalFlip(object):
    """将样本中的图像进行水平翻转"""

    def __init__(self, flip_flag=False):
        self.flip_flag = flip_flag

    def __call__(self, img):
        # 根据flip_flag决定是否执行水平翻转
        if self.flip_flag:
            return cv2.flip(img,1,dst=None)
        return img


class NoiseGauss(object):
    """对样本中的图像添加高斯噪声"""

    def __init__(self, sigma=0):
        self.sigma = sigma

    def __call__(self, image):
        temp_img = np.float64(np.copy(image))
        h, w = image.shape[:2]
        noise = np.random.randn(h, w) * self.sigma
        noisy_img = np.zeros(temp_img.shape, np.float64)
        if len(temp_img.shape) == 2:
            noisy_img = temp_img + noise
        else:
            noisy_img[:, :, 0] = temp_img[:, :, 0] + noise
            noisy_img[:, :, 1] = temp_img[:, :, 1] + noise
            noisy_img[:, :, 2] = temp_img[:, :, 2] + noise
        return noisy_img


class ToTensor(object):
    """将样本中的ndarrays转换为Tensors."""

    def __call__(self, image):
        # 交换颜色轴因为
        # numpy包的图片是: H * W * C
        # torch包的图片是: C * H * W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        return image


class ColorNormalize(object):
    """将样本中的图像进行归一化"""

    def __call__(self, image):
        image = image / 255.0
        return image
