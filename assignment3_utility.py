"""
assignment3_utility.py

This file contains helper functions for preprocessing Atari Pong frames
and handling reward transformations for the DQN agent.

Sections:
    1. Image preprocessing functions
       - Crop, downsample, grayscale, normalize.
    2. Reward transform helper
       - Convert raw reward signal into a simpler form.
    3. Frame processing wrapper
       - Combine all steps into a single call used by the agent.

All functions here are general-purpose and can be reused in both
training and evaluation.
"""

import numpy as np


# ==============================
# 1. IMAGE PREPROCESSING HELPERS
# ==============================

def img_crop(img):
    """
    Crop the raw Pong image.

    - The top and bottom areas of the Atari frame often contain
      scores, borders, and irrelevant information.
    - Cropping focuses only on the playing area (the middle).

    Parameters
    ----------
    img : np.ndarray
        The original RGB frame from the environment, shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Cropped RGB frame.
    """
    # Typical Pong frames are 210x160x3.
    # We cut off 30 pixels from the top and 12 from the bottom.
    return img[30:-12, :, :]


def downsample(img):
    """
    Downsample the image resolution by a factor of 2.

    - Reduces the size from (H, W) to roughly (H/2, W/2).
    - This makes the neural network smaller and faster,
      while preserving the game structure.

    Parameters
    ----------
    img : np.ndarray
        Cropped RGB frame.

    Returns
    -------
    np.ndarray
        Downsampled image.
    """
    # Take every second pixel in both height and width.
    return img[::2, ::2]


def to_grayscale(img):
    """
    Convert an RGB image to grayscale by averaging channels.

    Parameters
    ----------
    img : np.ndarray
        RGB frame, shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Grayscale image, shape (H, W), dtype uint8.
    """
    # Average along the color channel axis (axis=2).
    return np.mean(img, axis=2).astype(np.uint8)


def normalize_grayscale(img):
    """
    Normalize grayscale values to range [-1, 1].

    - Helps neural networks train more stably.
    - Original Pong pixel values are in [0, 255].

    Parameters
    ----------
    img : np.ndarray
        Grayscale image, shape (H, W), dtype uint8.

    Returns
    -------
    np.ndarray
        Normalized grayscale image, dtype float32, range [-1, 1].
    """
    # Convert to float32 and normalize
    # Step 1: shift by 128
    # Step 2: divide by 128 → roughly in [-1, 1]
    return ((img.astype(np.float32) - 128.0) / 128.0).astype(np.float32)


# =======================
# 2. REWARD TRANSFORMATION
# =======================

def transform_reward(reward):
    """
    Transform the raw environment reward to a simplified form.

    For Pong:
        - The game gives +1 or -1 when a point is scored, and 0 otherwise.
        - Using np.sign keeps the signal but removes magnitude (if any).

    Parameters
    ----------
    reward : float
        Raw reward from the environment.

    Returns
    -------
    float
        Transformed reward (e.g., -1.0, 0.0, or 1.0).
    """
    return float(np.sign(reward))


# ==========================
# 3. MAIN FRAME PREPROCESSOR
# ==========================

def process_frame(img, image_shape):
    """
    Full frame preprocessing pipeline for Pong frames.

    Steps:
        1. Crop the top and bottom borders (score/lives).
        2. Downsample by factor of 2 to reduce resolution.
        3. Convert from RGB to grayscale.
        4. Normalize grayscale to [-1, 1].
        5. Reshape into (H, W, 1) and add a batch dimension → (1, H, W, 1)

    Parameters
    ----------
    img : np.ndarray
        Raw RGB frame from the environment, shape (H, W, 3).
    image_shape : tuple[int, int]
        Desired (height, width) of the processed frame, e.g. (80, 80).

    Returns
    -------
    np.ndarray
        Processed frame, shape (1, H, W, 1), dtype float32.
        - The first dimension is batch size (1 frame).
        - The last dimension is channel (1 for grayscale).
    """
    # 1. Crop non-useful areas
    img = img_crop(img)

    # 2. Downsample by factor 2
    img = downsample(img)

    # 3. Convert to grayscale
    img = to_grayscale(img)

    # 4. Normalize to [-1, 1]
    img = normalize_grayscale(img)

    # 5. Resize to target shape if needed (simple center-crop or pad)
    target_h, target_w = image_shape
    h, w = img.shape

    # If the size doesn't match exactly, we can do a simple center-crop
    # (for Pong, often the shape already matches reasonably after crop+downsample).
    start_h = max((h - target_h) // 2, 0)
    start_w = max((w - target_w) // 2, 0)
    img = img[start_h:start_h + target_h, start_w:start_w + target_w]

    # Ensure the final size matches exactly (in case of rounding)
    img = img[:target_h, :target_w]

    # Add channel dimension (H, W) -> (H, W, 1)
    img = np.expand_dims(img, axis=-1)

    # Add batch dimension (H, W, 1) -> (1, H, W, 1)
    img = np.expand_dims(img, axis=0)

    return img
