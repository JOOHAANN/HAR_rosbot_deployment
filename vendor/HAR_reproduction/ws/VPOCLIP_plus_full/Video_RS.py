import numpy as np
import torch

def Load_video_feature():
    # Assume the video features have already been saved to a NumPy file
    video_feature_np = np.load('video_feature_raw.npy')  # replace with your file path
    video_feature_torch = torch.from_numpy(video_feature_np).float()  # convert to a PyTorch tensor with float dtype
    return video_feature_torch

