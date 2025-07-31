"""
Custom Dataset module
"""

import os
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import ast
from utils import log
import cv2
from common.constant import INITIAL_LABELS


REQUIRES_AUG = [3, 4, 10, 6, 7] # PSAX-apical, PSAX-mid, suprasternal, apical-3ch, apical 5-ch

class CustomDatasetVideo(Dataset):
    """
    Custom dataset implementation
    """
    def __init__(self,
                csv_info,
                root_dir,
                VIEW_MAP,
                data_mean = 0.5,
                data_std = 0.5,
                use_npy = False,
                remove_ecg=True, 
                remove_static=True,
                #  :Callable[[Image.Image | torch.Tensor], Image.Image | torch.Tensor],
                #  frames:int,
                    # frame_select:Callable[[torch.Tensor, int], torch.Tensor]=
                    #     lambda t, m: torch.stack([t[len(t) * i // m] for i in range(m)]),
                transform = None,
                ):
        """
        frames must be set to the number of frames to use per sample
        frame_select(frames, output_size) MUST return a tensor with 'outputsize' frames
        """
        self.meta = csv_info
        self.root_dir = root_dir
        self.VIEW_MAP = VIEW_MAP
        self.data_mean = data_mean
        self.data_std = data_std
        self.use_npy = use_npy
        self.transform = transform
        self.remove_ecg = remove_ecg
        self.remove_static = remove_static
        self.labels_require_aug = [3, 4, 10, 6, 7]
  

    def __len__(self):
        return len(self.meta)

    def remove_static_background(self, images, k:int=5)-> torch.Tensor:
        """
        Preprocess a video (images)
        
        Args:
            images (Tensor shape:(slices, channels, height, width)):
                The video to process
            k (int optional default=5):
                The max number of steps
        Returns:
            The video with unchanged voxels set to 0
        """
        image_array = torch.stack([TF.to_tensor(img) for img in images])
        if image_array.max() > 1.0:
            image_array /= 255.0

        slices, ch, height, width = image_array.shape
        mask = torch.zeros((ch, height, width), dtype=torch.uint8)
        steps = min(5, slices)
        for i in range(steps - 1):
            mask |= (image_array[i] != image_array[i + 1])  # element-wise comparison
        output = image_array * mask  # static pixels become zero
        return output
    
    def remove_ecg_line(self, images):
        """
        Preprocess a video (images)
        
        Args:
            images (Tensor shape:(slices, channels, height, width)):
                The video to process

        Returns:
            The video with removed green ecg pixels
        """
        output = []
        for img in images:
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, (40, 40, 40), (80, 255, 255))
            img[mask > 0] = 0
            output.append(img)
        return output
    

    def _index(self, name:str) -> int:
        """
        Get the index from a path string
        """
        return int(os.path.basename(name).split('_')[-1].split('.')[0])
    
    def ensure_tensor_rgb(self, image):
        """
        Ensure image type to be tensor
        Args:
            image (Numpy or Tensor or Image type)
        Returns:
            image (Tensor type)
        """
        if isinstance(image, np.ndarray):
            image = torch.tensor(image).float() / 255.0
            image = image.permute(2, 0, 1)  # [H, W, C] → [C, H, W]
        elif isinstance(image, Image.Image):
            image = TF.to_tensor(image)
        elif isinstance(image, torch.Tensor):
            if image.max() > 1.0:
                image = image / 255.0
        return image
    
    def __getitem__(self, index):
        row = self.meta.iloc[index]
        path = row['path']
        case_path = os.path.basename(path)
        end = row['end']
        frame_ids = np.linspace(0, end, 10, dtype=int)
        label = torch.tensor(self.VIEW_MAP.index(row['label'].lower()), dtype=torch.long)


        # Load images
        images = []
        for frame_id in frame_ids:
            img_path = os.path.join(self.root_dir, path, f"{case_path}_{frame_id}.png")
            image = Image.open(img_path).convert("RGB")
            images.append(np.array(image))  # Save as numpy arrays for preprocessing

        # ECG Removal
        if self.remove_ecg:
            images = self.remove_ecg_line(images)

        # Background Removal
        if self.remove_static:
            images = self.remove_static_background(images)

        # Convert to tensor: [T, C, H, W]
        frames_tensor = torch.stack([self.ensure_tensor_rgb(img) for img in images])

        # Apply optional transforms (assumed batch-safe)
        if self.transform is not None:
            frames_tensor = self.transform(frames_tensor)

        # Grayscale, Resize and Normalize
        processed_frames = []
        for frame in frames_tensor:
            frame = transforms.Grayscale(num_output_channels=3)(frame)
            frame = TF.resize(frame, size=[224, 224])
            frame = TF.normalize(frame, mean=self.data_mean, std=self.data_std)
            processed_frames.append(frame)
            
        frames_tensor = torch.stack(processed_frames)


        return frames_tensor, label


class CustomDatasetImg(Dataset):
    """
    Custom dataset implementation
    """
    def __init__(self,
                csv_info,
                root_dir,
                VIEW_MAP,
                data_mean = 0.5,
                data_std = 0.5,
                use_npy = False,
                remove_ecg=True, 
                remove_static=True,
                #  :Callable[[Image.Image | torch.Tensor], Image.Image | torch.Tensor],
                #  frames:int,
                    # frame_select:Callable[[torch.Tensor, int], torch.Tensor]=
                    #     lambda t, m: torch.stack([t[len(t) * i // m] for i in range(m)]),
                transform = None,
                ):
        """
        frames must be set to the number of frames to use per sample
        frame_select(frames, output_size) MUST return a tensor with 'outputsize' frames
        """
        self.meta = csv_info
        self.VIEW_MAP = VIEW_MAP
        self.root_dir = root_dir
        self.data_mean = data_mean
        self.data_std = data_std
        self.use_npy = use_npy
        self.transform = transform
        self.remove_ecg = remove_ecg
        self.remove_static = remove_static
        self.labels_require_aug = [3, 4, 10, 6, 7]
  

    def __len__(self):
        return len(self.meta)

    def remove_static_background(self, img, path, case_path, k:int=5)-> torch.Tensor:
        """
        Preprocess a video (images)
        
        Args:
            images (Tensor shape:(slices, channels, height, width)):
                The video to process
            k (int optional default=5):
                The max number of steps
        Returns:
            The video with unchanged voxels set to 0
        """
        images = []
        frame_ids = 5
        for frame_id in range(frame_ids):
            img_path = os.path.join(self.root_dir, path, f"{case_path}_{frame_id}.png")
            image = TF.to_tensor(Image.open(img_path).convert("RGB"))
            images.append(image)  # Save as numpy arrays for preprocessing

        frames = torch.stack(images).float() / 255.0

        # image_array = TF.to_tensor(img)
        # if image_array.max() > 1.0:
        #     image_array /= 255.0

        slices, ch, height, width = frames.shape
        mask = torch.zeros((ch, height, width), dtype=torch.uint8)
        steps = min(5, slices)
        for i in range(steps - 1):
            mask |= (frames[i] != frames[i + 1])  # element-wise comparison
        img = self.ensure_tensor_rgb(img)
        output = img * mask  # static pixels become zero
        return output
    
    def remove_ecg_line(self, img):
        """
        Preprocess a video (images)
        
        Args:
            images (Tensor shape:(slices, channels, height, width)):
                The video to process

        Returns:
            The video with removed green ecg pixels
        """
        if isinstance(img, Image.Image):
            img = np.array(img)
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (40, 40, 40), (80, 255, 255))
        img[mask > 0] = 0
        return img
    

    def _index(self, name:str) -> int:
        """
        Get the index from a path string
        """
        return int(os.path.basename(name).split('_')[-1].split('.')[0])
    
    def ensure_tensor_rgb(self, image):
        """
        Ensure image type to be tensor
        Args:
            image (Numpy or Tensor or Image type)
        Returns:
            image (Tensor type)
        """
        if isinstance(image, np.ndarray):
            image = torch.tensor(image).float() / 255.0
            image = image.permute(2, 0, 1)  # [H, W, C] → [C, H, W]
        elif isinstance(image, Image.Image):
            image = TF.to_tensor(image)
        elif isinstance(image, torch.Tensor):
            if image.max() > 1.0:
                image = image / 255.0
        return image
    
    def __getitem__(self, index):
        row = self.meta.iloc[index]
        frame = row['frame']
        path = row['path']
        case_path = os.path.basename(path)
        end = row['end']
        label = torch.tensor(self.VIEW_MAP.index(row['label'].lower()), dtype=torch.long)

        # Load images
        
        img_path = os.path.join(self.root_dir, path, f"{case_path}_{frame}.png")
        image = Image.open(img_path).convert("RGB")

        # ECG Removal
        if self.remove_ecg:
            image = self.remove_ecg_line(image)

        # Background Removal
        if self.remove_static:
            image = self.remove_static_background(image, path, case_path)

        # Apply optional transforms (assumed batch-safe)
        if self.transform is not None:
            image = self.transform(image)

        # Grayscale, Resize and Normalize

        image = transforms.Grayscale(num_output_channels=3)(image)
        image = TF.resize(image, size=[224, 224])
        image = TF.normalize(image, mean=self.data_mean, std=self.data_std)


        return image, label

