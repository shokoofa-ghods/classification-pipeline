import pydicom
import os
import numpy as np
import cv2
import pydicom
from glob import glob
from tqdm import tqdm
from PIL import Image

def extract_frames_from_dicom(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    frames = ds.pixel_array  # shape: (num_frames, H, W, 3) or (num_frames, H, W)
    if frames.ndim == 3:  # If grayscale, convert to RGB
        frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) for frame in frames]
    return frames.astype(np.uint8) 

def resize_frame(frame, target_size=(256, 256)):
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

def save_frames_as_images_and_npy(frames, output_dir, base_name="frame"):
    os.makedirs(output_dir, exist_ok=True)
    saved_frames = []
    for i, frame in enumerate(frames):
        frame_path = os.path.join(output_dir, f"{base_name}_{i}.png")
        img = Image.fromarray(frame, mode="RGB")
        img.save(frame_path, format='PNG')
        # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(frame_path, frame)
        saved_frames.append(frame)
    np.save(os.path.join(output_dir, "frames.npy"), np.stack(saved_frames))

def get_all_dicom_paths(input_root):
    return glob(os.path.join(input_root, "**/*"))


def get_patient_and_case_id(dcm_path):    
    patient_id, case_id = dcm_path.split('/')[-2:]
    return patient_id, case_id

def preprocess_all_videos(input_root, output_root, target_size=(256, 256)):
    dcm_paths = get_all_dicom_paths(input_root)
    total_sum = 0.0
    total_squared = 0.0
    total_pixels = 0

    for dcm_path in tqdm(dcm_paths, desc="Processing videos"):
        patient_id, case_id = get_patient_and_case_id(dcm_path)
        output_dir = os.path.join(output_root, f"{patient_id}", f"{case_id}")
        os.makedirs(output_dir, exist_ok=True)

        frames = extract_frames_from_dicom(dcm_path)

        resized_frames = []
        for frame in frames:
            resized = resize_frame(frame, target_size)
            norm_frame = resized.astype(np.float32) / 255.0
            total_sum += norm_frame.sum(axis=(0, 1))
            total_squared += (norm_frame ** 2).sum(axis=(0, 1))
            total_pixels += norm_frame.shape[0] * norm_frame.shape[1]
            resized_frames.append(resized)

        save_frames_as_images_and_npy(resized_frames, output_dir, case_id)


    mean = total_sum / total_pixels
    std = np.sqrt((total_squared / total_pixels) - mean ** 2)

    print("\nGlobal Dataset Mean (per channel):", mean)
    print("Global Dataset Std (per channel):", std)
    return mean, std

input_path = '/home/shokoo/EchoView/video_class/Tahseen'
output_path = '/home/shokoo/EchoView/video_class/TTE_processed'

final_mean, final_std = preprocess_all_videos(input_path, output_path)