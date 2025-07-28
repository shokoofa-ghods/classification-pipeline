"""
Transforms module
"""

import random
from typing import Any, Callable

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms
from PIL import Image
from torch import nn
import kornia.augmentation as K
import imgaug.augmenters as iaa


class VideoAugmentation(nn.Module):
    """
    Class to apply transforms to all frames of a video
    """
    def __init__(self, augmentations, num_select=2, seed=None):
        super().__init__()
        self.augmentations = augmentations
        self.num_select = num_select
        self.seed = seed

    def forward(self, video_tensor: torch.Tensor):

        first_frame_shape = video_tensor[0].shape

        for aug in self.augmentations:
            if isinstance(aug, ImgAugWrapper):
                aug.set_deterministic()
            elif hasattr(aug, "set_deterministic"):
                if "img_shape" in aug.set_deterministic.__code__.co_varnames:
                    aug.set_deterministic(img_shape=first_frame_shape) #pass image shape as argument
                else:
                    aug.set_deterministic()

        selected = random.sample(self.augmentations, k=self.num_select)

        for aug in selected:
            video_tensor = torch.stack([aug(frame) for frame in video_tensor])

        return video_tensor
    

class AddGaussianNoise(nn.Module):
    """Add Gaussian noise to a video."""
    def __init__(self, mean=0., std=0.1):
        super().__init__()
        self.mean = mean
        self.std = std
    def forward(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class MotionBlurWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.blur = K.RandomMotionBlur(kernel_size=5, angle=[-10., 10.], direction=[-0.5, 0.5], p=1.0)

    def forward(self, x):
        return self.blur(x.unsqueeze(0)).squeeze(0)
    

class RandomErasingWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse_dropout = K.RandomErasing(scale=(0.015, 0.04), ratio=(0.2, 3.3), p=0.2)

    def forward(self, x):
        return self.coarse_dropout(x.unsqueeze(0)).squeeze(0)


class QualityDecay(nn.Module):
    def __init__(self, strength=0.5, mode="noise_blur"):
        super().__init__()
        self.strength = strength
        self._deterministic = False
        self.mode = mode

    def set_deterministic(self):
        self._deterministic = True

    def forward(self, x):
        C, H, W = x.shape
        device = x.device

        # Create vertical + radial mask (0 at top/edge, 1 at bottom/center)
        y = torch.linspace(0, 1, H, device=device).unsqueeze(1).repeat(1, W)  # vertical gradient
        x_pos = torch.linspace(-1, 1, W, device=device).unsqueeze(0).repeat(H, 1)  # horizontal wedge
        r = torch.sqrt(x_pos**2 + (y - 0.5)**2)
        decay_mask = ((0.3 * y + 0.7 * r) ).clamp(0, 1)

        decay_mask = decay_mask.unsqueeze(0).repeat(C, 1, 1)  # [C, H, W]

        if self.mode == "noise_blur":
            if self._deterministic:
                self._noise = torch.randn_like(x) * self.strength
                noise = self._noise * decay_mask
            else:
                noise = torch.randn_like(x) * decay_mask * self.strength
            x = x + noise
            # Apply slight blur by average pooling, weighted by mask
            blurred = F.avg_pool2d(x.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
            x = (1 - decay_mask) * x + decay_mask * blurred

        elif self.mode == "contrast_drop":
            mean = x.mean(dim=(1, 2), keepdim=True)
            x = (1 - decay_mask) * x + decay_mask * mean  # pull toward mean (less contrast)

        return x


class RandomSectorCrop(nn.Module):
    def __init__(self, width_range=(0.5, 1.0), fill=0):
        super().__init__()
        self.width_range = width_range
        self.fill = fill
        self._diterministic = None
    
    def set_deterministic(self, img_shape=None):
        if img_shape is not None:
            _, W, _ = img_shape
        else:
            W = 224  # fallback, needs a proper size
        scale_w = torch.empty(1).uniform_(*self.width_range).item()
        new_W = int(W * scale_w)
        self._diterministic = new_W
    
    def forward(self, img):
        _, W, H = img.shape
        # scale_w = torch.empty(1).uniform_(*self.width_range).item()
        # new_W = int(W * scale_w)

        new_W = self._diterministic

        center = W//2
        left = max(center - new_W//2, 0)
        right = left + new_W
 
        cropped = img[:, :, left:right]

        pad_left = (W - new_W) // 2
        pad_right = W - new_W - pad_left
        
        padded = TF.pad(cropped, [pad_left, 0, pad_right, 0], fill=self.fill)

        return padded


class ImgAugWrapper:
    def __init__(self, augmenter):
        self.augmenter = augmenter
        self.det_augmenter = None

    def set_deterministic(self):
        self.det_augmenter = self.augmenter.to_deterministic()

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        is_single = frames.dim() == 3
        if is_single:
            frames = frames.unsqueeze(0)  # [1, C, H, W]

        frames_np = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        
        augmented_np = self.det_augmenter(images=frames_np)
        augmented = torch.from_numpy(augmented_np).float() / 255.0
        augmented = augmented.permute(0, 3, 1, 2)

        if is_single:
            return augmented[0]
        return augmented

    @property
    def name(self):
        return self.augmenter.name if hasattr(self.augmenter, 'name') else str(self.augmenter)

    def __repr__(self):
        return f"ImgAugWrapper({self.name})"


video_augs = [
    QualityDecay(strength=0.3, mode='contrast_drop'),
    RandomSectorCrop(width_range=(0.35, 0.8)),
    ImgAugWrapper(iaa.KeepSizeByResize(iaa.Crop(percent=(0.2, 0.2, 0.2, 0.2)))),
    ImgAugWrapper(iaa.Fliplr(0.5)), #horizontally
    ImgAugWrapper(iaa.Flipud(0.5)), #vetrically
    ImgAugWrapper(iaa.GammaContrast((0.8, 1.2))),
    ImgAugWrapper(iaa.SigmoidContrast(gain=(5, 10))),
    ImgAugWrapper(iaa.LogContrast(gain=(0.8, 1.2))),
    ImgAugWrapper(iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))),
    ImgAugWrapper(iaa.GaussianBlur(sigma=(0, 1.5))),
    ImgAugWrapper(iaa.CoarseDropout((0.02, 0.1), size_percent=(0.05, 0.05), per_channel=False)),
    ImgAugWrapper(iaa.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), shear=(-8, 8), mode='constant', cval=0)),
    ImgAugWrapper(iaa.PerspectiveTransform(scale=(0.01, 0.05), mode='constant', cval=0)),
]

video_aug_pipeline = VideoAugmentation(video_augs, num_select=5)
