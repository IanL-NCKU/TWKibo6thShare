
import random
from PIL import Image
import numpy as np
import copy
import os
import cv2
import math
from typing import Tuple, Union
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F


"""=========================== Generate Multitple Class image ============================"""
def get_foreground_bbox(image_array):
    """
    Get the bounding box of non-transparent pixels in an RGBA image.
    
    Args:
        image_array: RGBA image array
        
    Returns:
        tuple: (min_y, max_y, min_x, max_x) or None if no foreground found
    """
    if len(image_array.shape) != 3 or image_array.shape[2] != 4:
        raise ValueError("Image must be RGBA format")
    
    # Find non-transparent pixels (alpha > 0)
    alpha_channel = image_array[:, :, 3]
    non_transparent_pixels = np.where(alpha_channel > 0)
    
    if len(non_transparent_pixels[0]) == 0:
        return None  # No foreground found
    
    min_y, max_y = np.min(non_transparent_pixels[0]), np.max(non_transparent_pixels[0])
    min_x, max_x = np.min(non_transparent_pixels[1]), np.max(non_transparent_pixels[1])
    
    return (min_y, max_y, min_x, max_x)

def extract_foreground(image_array, bbox):
    """
    Extract the foreground region from an image using bounding box.
    
    Args:
        image_array: RGBA image array
        bbox: (min_y, max_y, min_x, max_x)
        
    Returns:
        numpy array: Cropped foreground region
    """
    min_y, max_y, min_x, max_x = bbox
    return image_array[min_y:max_y+1, min_x:max_x+1]

def find_valid_positions(blank_shape, foreground_shape, occupied_areas=None):
    """
    Find all valid positions where foreground can be placed.
    
    Args:
        blank_shape: (height, width) of blank image
        foreground_shape: (height, width) of foreground
        occupied_areas: List of occupied rectangles [(y1, x1, y2, x2), ...]
        
    Returns:
        list: Valid (y, x) positions
    """
    blank_h, blank_w = blank_shape[:2]
    fg_h, fg_w = foreground_shape[:2]
    
    valid_positions = []
    
    # Check all possible positions
    for y in range(blank_h - fg_h + 1):
        for x in range(blank_w - fg_w + 1):
            # Check if this position overlaps with any occupied area
            if occupied_areas:
                overlaps = False
                new_rect = (y, x, y + fg_h - 1, x + fg_w - 1)
                
                for occupied_rect in occupied_areas:
                    if rectangles_overlap(new_rect, occupied_rect):
                        overlaps = True
                        break
                
                if not overlaps:
                    valid_positions.append((y, x))
            else:
                valid_positions.append((y, x))
    
    return valid_positions

def rectangles_overlap(rect1, rect2):
    """
    Check if two rectangles overlap.
    
    Args:
        rect1, rect2: (y1, x1, y2, x2) format
        
    Returns:
        bool: True if rectangles overlap
    """
    y1_1, x1_1, y2_1, x2_1 = rect1
    y1_2, x1_2, y2_2, x2_2 = rect2
    
    return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)

def blend_foreground(blank_image, foreground, position):
    """
    Blend foreground onto blank image at specified position using alpha blending.
    
    Args:
        blank_image: Target image array (RGBA)
        foreground: Foreground image array (RGBA)
        position: (y, x) position to place foreground
    """
    y, x = position
    fg_h, fg_w = foreground.shape[:2]
    
    # Extract alpha channel
    alpha = foreground[:, :, 3:4] / 255.0
    
    # Blend RGB channels
    for c in range(3):
        blank_image[y:y+fg_h, x:x+fg_w, c] = (
            alpha[:, :, 0] * foreground[:, :, c] + 
            (1 - alpha[:, :, 0]) * blank_image[y:y+fg_h, x:x+fg_w, c]
        )
    
    # Update alpha channel (take maximum)
    blank_image[y:y+fg_h, x:x+fg_w, 3] = np.maximum(
        blank_image[y:y+fg_h, x:x+fg_w, 3],
        foreground[:, :, 3]
    )


def update_mask_channel(result_mask, mask_pil, position, channel_idx):
    """
    Update a specific channel of the result mask with the placed image mask.
    
    Args:
        result_mask: Multi-channel mask array (H, W, N)
        mask_pil: PIL Image mask (grayscale)
        position: (y, x) position where image was placed
        channel_idx: Which channel to update
    """
    y, x = position
    mask_array = np.array(mask_pil)
    mask_h, mask_w = mask_array.shape
    
    # Update the specific channel
    result_mask[y:y+mask_h, x:x+mask_w, channel_idx] = mask_array

def fitintoblank(blank_image_array, png_list, overlap=False, existing_mask=None, existing_yolo_list=None, 
                 max_attempts=5, down_rate=0.8, min_scale=0.4):
    """
    Fit PNG foregrounds into a blank image.
    
    Args:
        blank_image_array: Blank image array (RGB or RGBA)
        png_list: List of tuples (cls_id, x_cen, y_cen, w, h, classname, img_rgba, msk)
        overlap: If False, prevent foregrounds from overlapping
        existing_mask: Optional existing multi-channel mask to extend
        existing_yolo_list: Optional existing YOLO annotations to extend
        max_attempts: Maximum number of attempts to place image before resizing down
        down_rate: Rate to resize down the image (e.g., 0.9 = 90% of original size)
        min_scale: Minimum scale threshold (stop trying if image becomes too small)
        
    Returns:
        tuple: (new_yolo_list, result_msk, result_img, failed_placements)
            - new_yolo_list: List of YOLO format annotations
            - result_msk: Multi-channel mask array (H, W, N) 
            - result_img: Final composed image (RGBA format)
            - failed_placements: List of indices that couldn't be placed
    """
    # [Previous initialization code remains the same...]
    # Convert blank image to RGBA if needed
    if len(blank_image_array.shape) == 3 and blank_image_array.shape[2] == 3:
        alpha_channel = np.ones((blank_image_array.shape[0], blank_image_array.shape[1], 1), 
                               dtype=blank_image_array.dtype) * 255
        result_image = np.concatenate([blank_image_array, alpha_channel], axis=2)
    elif len(blank_image_array.shape) == 3 and blank_image_array.shape[2] == 4:
        result_image = blank_image_array.copy()
    else:
        raise ValueError("Blank image must be RGB or RGBA format")
    
    # Initialize multi-channel mask
    img_height, img_width = result_image.shape[:2]
    num_new_images = len(png_list)
    
    # Handle existing mask and YOLO list
    if existing_mask is not None:
        existing_channels = existing_mask.shape[2]
        total_channels = existing_channels + num_new_images
        result_mask = np.zeros((img_height, img_width, total_channels), dtype=np.uint8)
        result_mask[:, :, :existing_channels] = existing_mask
        channel_offset = existing_channels
    else:
        total_channels = num_new_images
        result_mask = np.zeros((img_height, img_width, total_channels), dtype=np.uint8)
        channel_offset = 0
    
    # Handle existing YOLO list
    if existing_yolo_list is not None:
        new_yolo_list = existing_yolo_list.copy()
    else:
        new_yolo_list = []
    
    # Initialize occupied areas for overlap prevention
    if not overlap:
        if existing_mask is not None:
            occupied_areas = []
            for ch in range(existing_mask.shape[2]):
                mask_channel = existing_mask[:, :, ch]
                if np.any(mask_channel > 0):
                    non_zero_pixels = np.where(mask_channel > 0)
                    if len(non_zero_pixels[0]) > 0:
                        min_y, max_y = np.min(non_zero_pixels[0]), np.max(non_zero_pixels[0])
                        min_x, max_x = np.min(non_zero_pixels[1]), np.max(non_zero_pixels[1])
                        occupied_areas.append((min_y, min_x, max_y, max_x))
        else:
            occupied_areas = []
    else:
        occupied_areas = None
    
    placed_count = 0
    failed_placements = []  # Track failed placements
    
    # Shuffle the list to add randomness to placement
    png_list_shuffled = png_list.copy()
    random.shuffle(png_list_shuffled)
    
    for list_idx, png_data in enumerate(png_list_shuffled):
        try:
            # Unpack png_data
            cls_id, x_cen, y_cen, w, h, classname, img_rgba, msk = png_data
            
            # Ensure the image is RGBA format
            if len(img_rgba.shape) != 3 or img_rgba.shape[2] != 4:
                print(f"Warning: Image {list_idx} is not RGBA format, skipping")
                continue
            
            # Get foreground bounding box
            bbox = get_foreground_bbox(img_rgba)
            if bbox is None:
                print(f"Warning: No foreground found in image {list_idx}")
                continue
            
            # Extract foreground
            foreground = extract_foreground(img_rgba, bbox)
            original_foreground = foreground.copy()  # Keep original for resizing
            current_scale = 1.0  # Track current scale
            
            # Try to place image with multiple attempts and resizing if needed
            placement_successful = False
            
            while current_scale >= min_scale and not placement_successful:
                # Try max_attempts times at current scale
                for attempt in range(max_attempts):
                    # Find valid positions
                    valid_positions = find_valid_positions(
                        result_image.shape, 
                        foreground.shape, 
                        occupied_areas
                    )
                    
                    if valid_positions:
                        # Successfully found a valid position
                        placement_successful = True
                        break
                    
                    # If overlap is allowed, we don't need to keep trying
                    if overlap:
                        # Find all possible positions (ignoring occupied areas)
                        valid_positions = find_valid_positions(
                            result_image.shape, 
                            foreground.shape, 
                            occupied_areas=None
                        )
                        if valid_positions:
                            placement_successful = True
                            break
                
                # If we still haven't found a position, resize down
                if not placement_successful and current_scale >= min_scale:
                    current_scale *= down_rate
                    
                    if current_scale >= min_scale:
                        # Resize foreground
                        new_h = max(1, int(original_foreground.shape[0] * current_scale))
                        new_w = max(1, int(original_foreground.shape[1] * current_scale))
                        foreground = cv2.resize(original_foreground, (new_w, new_h), 
                                              interpolation=cv2.INTER_AREA)
                        print(f"Resizing image {list_idx} to scale {current_scale:.2f} (attempt after {max_attempts} tries)")
                    else:
                        print(f"Image {list_idx} became too small (scale {current_scale:.2f} < {min_scale})")
                        break
            
            # Check if placement was successful
            if not placement_successful:
                print(f"Failed to place image {list_idx} after all attempts and resizing")
                failed_placements.append(list_idx)
                continue
            
            # Choose a random valid position
            position = random.choice(valid_positions)
            
            # Blend foreground onto result image
            blend_foreground(result_image, foreground, position)
            
            # Update the mask channel for this image
            min_y, max_y, min_x, max_x = bbox
            
            # If we resized, we need to resize the mask too
            if current_scale != 1.0:
                # Resize the mask to match the resized foreground
                original_mask_array = np.array(msk)[min_y:max_y+1, min_x:max_x+1]
                resized_mask = cv2.resize(original_mask_array, 
                                        (foreground.shape[1], foreground.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                cropped_mask_pil = Image.fromarray(resized_mask, mode='L')
            else:
                cropped_mask_array = np.array(msk)[min_y:max_y+1, min_x:max_x+1]
                cropped_mask_pil = Image.fromarray(cropped_mask_array, mode='L')
            
            # Use channel_offset to place in correct channel
            update_mask_channel(result_mask, cropped_mask_pil, position, channel_offset + list_idx)
            
            # Calculate YOLO format annotation
            y_pos, x_pos = position
            fg_h, fg_w = foreground.shape[:2]
            
            # Convert to normalized coordinates (YOLO format)
            x_center_norm = (x_pos + fg_w / 2) / img_width
            y_center_norm = (y_pos + fg_h / 2) / img_height
            width_norm = fg_w / img_width
            height_norm = fg_h / img_height
            
            # Create YOLO annotation
            yolo_annotation = [cls_id, x_center_norm, y_center_norm, width_norm, height_norm]
            new_yolo_list.append(yolo_annotation)
            
            # Update occupied areas if overlap is disabled
            if not overlap:
                occupied_areas.append((y_pos, x_pos, y_pos + fg_h - 1, x_pos + fg_w - 1))
            
            placed_count += 1
            # print(f"Placed image {list_idx} (cls_id: {cls_id}, class: {classname}) at position {position} with scale {current_scale:.2f}")
            
        except Exception as e:
            print(f"Error processing image {list_idx}: {str(e)}")
            failed_placements.append(list_idx)
            continue
    
    print(f"Successfully placed {placed_count}/{len(png_list)} new images")
    # print(f"Failed to place {len(failed_placements)} images: {failed_placements}")
    # print(f"Total images in result: {len(new_yolo_list)}")
    
    return new_yolo_list, result_mask, result_image



"""=========================== Data augmentation classes images ============================"""
class RandomGradientBrightness:
    """Apply a linear brightness gradient to a tensor image at any angle."""
    
    def __init__(self, min_factor=0.5, max_factor=1.5, angle=0.):
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.angle = angle

    def __call__(self, tensor_img):
        """
        Apply gradient brightness to tensor.
        
        Args:
            tensor_img: torch.Tensor of shape (C, H, W) in [0,1] range
            
        Returns:
            torch.Tensor: brightness-adjusted image with same shape, rescaled to [0,1]
        """
        C, H, W = tensor_img.shape
        
        # Random endpoints
        start = random.uniform(self.min_factor, self.max_factor)
        end = random.uniform(self.min_factor, self.max_factor)
        
        # Build coordinate grid
        ys = torch.linspace(0, H-1, H, device=tensor_img.device)
        xs = torch.linspace(0, W-1, W, device=tensor_img.device)
        Y, X = torch.meshgrid(ys, xs, indexing='ij')
        
        # Unit vector for direction angle
        theta = math.radians(self.angle % 360)
        ux, uy = math.cos(theta), math.sin(theta)
        
        # Projection of each (x,y) onto (ux,uy)
        P = X * ux + Y * uy
        
        # Normalize P to [0,1]
        Pmin, Pmax = P.min(), P.max()
        if Pmax > Pmin:
            P_norm = (P - Pmin) / (Pmax - Pmin)
        else:
            P_norm = torch.zeros_like(P)
        
        # Linear ramp: start + (end-start) * P_norm
        ramp = start + (end - start) * P_norm   # shape [H, W]
        ramp = ramp.unsqueeze(0)                # shape [1, H, W]
        
        # Apply same ramp to all channels
        result = tensor_img * ramp
        
        # Rescale to [0,1] range instead of clipping
        result_min = result.min()
        result_max = result.max()
        
        if result_max > result_min:
            # Rescale: (value - min) / (max - min)
            result = (result - result_min) / (result_max - result_min)
        else:
            # If all values are the same, set to zeros
            result = torch.zeros_like(result)
        
        return result

    def __repr__(self):
        return (f"{self.__class__.__name__}(min={self.min_factor}, max={self.max_factor}, "
                f"angle={self.angle}°, rescale=True)")


class RandomGaussianNoise:
    """Add Gaussian noise to a tensor image."""
    
    def __init__(self, mean=0.0, std_min=0.0, std_max=0.1, p=0.5):
        assert 0 <= p <= 1, "`p` must be in [0,1]"
        self.mean = mean
        self.std_min = std_min
        self.std_max = std_max
        self.p = p

    def __call__(self, tensor_img):
        """
        Apply Gaussian noise to tensor.
        
        Args:
            tensor_img: torch.Tensor of shape (C, H, W) in [0,1] range
            
        Returns:
            torch.Tensor: noisy image with same shape
        """
        # Check probability - return original if not applying noise
        if random.random() > self.p:
            return tensor_img
        
        # Choose a standard deviation
        std = random.uniform(self.std_min, self.std_max)
        
        # Sample noise with same shape as tensor
        noise = torch.randn_like(tensor_img) * std + self.mean
        
        # Add noise and clamp to valid range
        t_noisy = (tensor_img + noise).clamp(0, 1)
        
        return t_noisy

    def __repr__(self):
        return (f"{self.__class__.__name__}(mean={self.mean}, "
                f"std_range=[{self.std_min},{self.std_max}], p={self.p})")


class Augmentationtransform2D:
    """2D Augmentation for image-mask pairs using tensor operations."""

    def __init__(self, aug_dict, aug_type='inner'):
        if aug_type.lower() not in ['inner', 'outer']:
            raise ValueError("aug_type must be either 'inner' or 'outer'")

        self.aug_type = aug_type.lower()

        if self.aug_type == 'inner':
            inner_aug_dict = aug_dict['inner']
            self.do_flip = inner_aug_dict['flip']
            self.do_rotate = inner_aug_dict['rotate']
            self.do_scale = inner_aug_dict['scale']
            self.flip_rate = inner_aug_dict['flip_rate'] if self.do_flip else 0.0
            self.rotate_range = inner_aug_dict['rotate_range'] if self.do_rotate else (0, 0)
            self.scale_range = inner_aug_dict['scale_range'] if self.do_scale else (1, 1)

        elif self.aug_type == 'outer':
            outer_aug_dict = aug_dict['outer']
            self.rate_brightness = outer_aug_dict['brightness']
            self.rate_contrast = outer_aug_dict['contrast']
            self.rate_sharpness = outer_aug_dict['sharpness']
            self.rate_gaussian_noise = outer_aug_dict['gaussian_noise']
            self.rate_gaussian_blur = outer_aug_dict['gaussian_blur']
            self.rate_motion_blur = outer_aug_dict['motion_blur']
            self.rate_gradient = outer_aug_dict['gradient_brightness']
            self.rate_perspective = outer_aug_dict['perspective']

            brightness_range = outer_aug_dict['brightness_range']
            contrast_range = outer_aug_dict['contrast_range']
            sharpness_range = outer_aug_dict['sharpness_range']
            blur_kernel_size = outer_aug_dict['blur_kernel_size']
            blur_sigma = outer_aug_dict['blur_sigma'] if self.rate_gaussian_blur > 0 else (0.1, 2.0)
            motion_blur_kernel_size = outer_aug_dict['motion_blur_kernel_size']
            motion_blur_angle_range = outer_aug_dict['motion_blur_angle_range']
            motion_blur_distance_range = outer_aug_dict['motion_blur_distance_range']
            gradient_brightness_range = outer_aug_dict['gradient_brightness_range'] 
            perspective_distortion = outer_aug_dict['perspective_distortion']
            gaussian_noise_list = outer_aug_dict['gaussian_noise_list']

            # Store parameters (keep +1 adjustment for torchvision compatibility)
            self.brightness_range = tuple([1 + x for x in brightness_range])
            self.contrast_range = tuple([1 + x for x in contrast_range])
            self.sharpness_range = tuple([1 + x for x in sharpness_range])
            self.gaussian_blur_kernel = tuple([x for x in blur_kernel_size])
            self.gaussian_blur_sigma = tuple([x for x in blur_sigma])
            self.motion_blur_kernel_size = tuple([x for x in motion_blur_kernel_size])
            self.motion_blur_angle_range = tuple([x for x in motion_blur_angle_range])
            self.motion_blur_distance_range = tuple([x for x in motion_blur_distance_range])
            self.gradient_brightness_range = tuple([x for x in gradient_brightness_range])
            self.perspective_distortion = perspective_distortion
            self.gaussian_noise_list = gaussian_noise_list

    @staticmethod
    def numpy_to_tensor(img_arr, msk_arr):
        """Convert numpy arrays to stacked tensor (C, H, W) in [0,1]."""
        # Stack arrays: img + mask
        stacked = np.concatenate([img_arr, msk_arr], axis=2)  # (H, W, C)
        
        # Convert to tensor and change to (C, H, W)
        tensor = torch.from_numpy(stacked).permute(2, 0, 1).float()
        
        # Normalize to [0, 1] if needed
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
            
        return tensor

    @staticmethod
    def tensor_to_numpy(stacked_tensor, img_channels, msk_channels):
        """Convert stacked tensor back to separate numpy arrays."""
        # Convert from (C, H, W) to (H, W, C)
        stacked_np = stacked_tensor.permute(1, 2, 0).detach().numpy()
        
        # Scale back to [0, 255] and convert to uint8
        stacked_np = (stacked_np * 255.0).clip(0, 255).astype(np.uint8)
        
        # Split back to image and mask
        img_arr = stacked_np[:, :, :img_channels]
        msk_arr = stacked_np[:, :, img_channels:]
        
        # Binarize masks (threshold at 0.5 in [0,1] space, which is 127 in [0,255])
        msk_arr = (msk_arr > 127).astype(np.uint8) * 255
        
        return img_arr, msk_arr

    @staticmethod
    def create_motion_blur_kernel(size, angle, distance):
        """Create motion blur kernel."""
        angle_rad = math.radians(angle)
        center = size // 2
        kernel = np.zeros((size, size), dtype=np.float32)
        
        dx = math.cos(angle_rad) * distance
        dy = math.sin(angle_rad) * distance
        steps = max(abs(dx), abs(dy), 1)
        x_step = dx / steps
        y_step = dy / steps
        
        for i in range(int(steps) + 1):
            x = int(center + i * x_step + 0.5)
            y = int(center + i * y_step + 0.5)
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0
        
        if kernel.sum() > 0:
            kernel = kernel / kernel.sum()
        else:
            kernel[center, center] = 1.0
        
        return kernel

    @staticmethod
    def motion_blur_multichannel(tensor_img, angle=None, distance=None, kernel_size=15):
        """
        Apply motion blur to a tensor image.
        
        Args:
            tensor_img: torch.Tensor of shape (C, H, W) in [0,1] range
            angle: Motion angle in degrees (0-360). If None, defaults to 0
            distance: Motion distance (blur strength). If None, defaults to 0
            kernel_size: Size of motion blur kernel (should be odd)
        
        Returns:
            torch.Tensor: Motion blurred tensor with same shape as input
        """
        # Handle default values
        if angle is None:
            angle = 0
        if distance is None:
            distance = 0
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create motion blur kernel
        kernel = Augmentationtransform2D.create_motion_blur_kernel(kernel_size, angle, distance)
        
        # Convert tensor to numpy for OpenCV processing
        img_np = tensor_img.permute(1, 2, 0).detach().numpy()  # (H, W, C)
        img_np = (img_np * 255.0).astype(np.uint8)
        
        # Apply blur to each channel
        if len(img_np.shape) == 3:
            # Multi-channel tensor (C, H, W) -> numpy (H, W, C)
            blurred_channels = []
            
            for channel in range(img_np.shape[2]):
                # Apply convolution to each channel
                channel_data = img_np[:, :, channel].astype(np.float32)
                blurred_channel = cv2.filter2D(channel_data, -1, kernel)
                blurred_channels.append(blurred_channel)
            
            # Stack channels back
            blurred_np = np.stack(blurred_channels, axis=2)
        
        elif len(img_np.shape) == 2:
            # Single channel (should not happen with stacked tensors, but handle anyway)
            channel_data = img_np.astype(np.float32)
            blurred_np = cv2.filter2D(channel_data, -1, kernel)
            blurred_np = np.expand_dims(blurred_np, axis=2)
        
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor_img.shape}")
        
        # Convert back to tensor format
        blurred_np = np.clip(blurred_np, 0, 255).astype(np.uint8)
        blurred_tensor = torch.from_numpy(blurred_np).permute(2, 0, 1).float() / 255.0
        
        return blurred_tensor.clamp(0, 1)

    def __call__(self, img_arr, msk_arr):
        """Apply augmentation using tensor operations with original torchvision functions."""
        assert img_arr.ndim == 3, "Image array must be 3D"
        assert msk_arr.ndim == 3, "Mask array must be 3D"
        assert img_arr.shape[:2] == msk_arr.shape[:2], "Image and mask must have same height and width"

        img_channels = img_arr.shape[2]
        msk_channels = msk_arr.shape[2]

        # Convert to tensor format (C, H, W) in [0,1]
        stack_image_tensor = self.numpy_to_tensor(img_arr, msk_arr)

        if self.aug_type == 'inner':
            # Flip augmentation (applied to both image and mask)
            if self.do_flip:
                if random.random() < self.flip_rate:
                    stack_image_tensor = F.hflip(stack_image_tensor)
                if random.random() < self.flip_rate:
                    stack_image_tensor = F.vflip(stack_image_tensor)

            # Affine transformations (applied to both image and mask)
            if any([self.do_rotate, self.do_scale]):
                angle = round(random.uniform(*self.rotate_range), 2)
                scale = round(random.uniform(*self.scale_range), 2)
                
                # Apply affine transformation
                stack_image_tensor = F.affine(
                    stack_image_tensor, 
                    angle=angle, 
                    translate=(0, 0), 
                    scale=scale, 
                    shear=0
                )

        elif self.aug_type == 'outer':
            # Split tensor for selective processing
            img_tensor = stack_image_tensor[:img_channels]  # Image channels
            msk_tensor = stack_image_tensor[img_channels:]  # Mask channels
            
            # Apply photometric transforms only to image channels
            if any([random.random() < self.rate_brightness, random.random() < self.rate_contrast]):
                # from torchvision import transforms
                color_jitter = transforms.ColorJitter(
                    brightness=self.brightness_range, 
                    contrast=self.contrast_range
                )
                img_tensor = color_jitter(img_tensor)
            
            if random.random() < self.rate_sharpness:
                sharpness_factor = random.uniform(*self.sharpness_range)
                # from torchvision import transforms
                sharpness_transform = transforms.RandomAdjustSharpness(
                    sharpness_factor=sharpness_factor, p=1.0
                )
                img_tensor = sharpness_transform(img_tensor)
            
            # Recombine for geometric transforms that affect all channels
            stack_image_tensor = torch.cat([img_tensor, msk_tensor], dim=0)
            
            # Blur operations
            if random.random() < self.rate_gaussian_blur:
                gaussian_blur = transforms.GaussianBlur(
                    kernel_size=self.gaussian_blur_kernel, 
                    sigma=self.gaussian_blur_sigma
                )
                stack_image_tensor = gaussian_blur(stack_image_tensor)

            elif random.random() < self.rate_motion_blur:
                #kernel size select odd int from tuple range (5, 21)
                motion_blur_kernel_size = random.choice(
                    [x for x in range(*self.motion_blur_kernel_size) if x % 2 == 1]
                )
                motion_blur_angle = random.uniform(*self.motion_blur_angle_range)
                motion_blur_distance = random.uniform(*self.motion_blur_distance_range)
                # print(f"Applying motion blur with angle {motion_blur_angle}° and distance {motion_blur_distance}")
                # print(f"Kernel size: {motion_blur_kernel_size}")
                stack_image_tensor = self.motion_blur_multichannel(
                    stack_image_tensor,
                    angle=motion_blur_angle, 
                    distance=motion_blur_distance, 
                    kernel_size=motion_blur_kernel_size
                )

            # Split again for image-only transforms
            img_tensor = stack_image_tensor[:img_channels]
            msk_tensor = stack_image_tensor[img_channels:]
            msk_tensor = binarize_tensor(msk_tensor)  # Ensure mask is binary
            # print max and min msk tensor values
            # print(f"Mask tensor min: {msk_tensor.min()}, max: {msk_tensor.max()}")
            # thresholding mask tensor  in each channel to [0,1]
            # original Mask tensor min: 0.0, max: 1.0000001192092896


            # Apply gradient brightness only to image channels
            if random.random() < self.rate_gradient:
                grad_bright1 = random.uniform(*self.gradient_brightness_range)
                grad_bright2 = random.uniform(*self.gradient_brightness_range)
                grad_bright_max = max(grad_bright1, grad_bright2)
                grad_bright_min = min(grad_bright1, grad_bright2)
                gradient_brightness = RandomGradientBrightness(
                    min_factor= grad_bright_min,
                    max_factor= grad_bright_max, 
                    angle=random.uniform(0, 360)
                )
                img_tensor = gradient_brightness(img_tensor)
            
            # Recombine for perspective transform
            stack_image_tensor = torch.cat([img_tensor, msk_tensor], dim=0)

            # Apply perspective transformation
            if random.random() < self.rate_perspective:
                # from torchvision import transforms
                perspective_transform = transforms.RandomPerspective(
                    distortion_scale=self.perspective_distortion, 
                    p=1.0,
                    interpolation=transforms.InterpolationMode.NEAREST
                )
                stack_image_tensor = perspective_transform(stack_image_tensor)
            
            # Split again for noise (image only)
            img_tensor = stack_image_tensor[:img_channels]
            msk_tensor = stack_image_tensor[img_channels:]
            msk_tensor = binarize_tensor(msk_tensor)  # Ensure mask is binary

            # Apply random noise only to image channels
            if random.random() < self.rate_gaussian_noise:
                noise_mean_range, noise_std_min_range, noise_std_max_range = self.gaussian_noise_list
                noise_mean = random.uniform(*noise_mean_range)
                noise_std_min = random.uniform(*noise_std_min_range)
                noise_std_max = random.uniform(*noise_std_max_range)
                noise_transform = RandomGaussianNoise(
                    mean=noise_mean,
                    std_min=noise_std_min,
                    std_max=noise_std_max,
                    p=1.0  # Force application since we already checked probability
                )
                img_tensor = noise_transform(img_tensor)

            # Final recombine
            stack_image_tensor = torch.cat([img_tensor, msk_tensor], dim=0)

        # Convert back to numpy
        new_img_arr, new_msk_arr = self.tensor_to_numpy(stack_image_tensor, img_channels, msk_channels)

        return new_img_arr, new_msk_arr






def getbboxfrommsk(msk: np.ndarray, margin: Union[int, float] = 0) -> Tuple[float, float, float, float]:
    """
    Convert a mask to YOLO bounding box format.
    
    Args:
        msk (np.ndarray): Input mask - 2D or 3D array
                         Values can be [0,1], [0,255], or [0,n]
        margin (int/float): Margin to add around bounding box
                           If int: pixel margin
                           If float: percentage margin (0.1 = 10%)
    
    Returns:
        Tuple[float, float, float, float]: (x_center, y_center, width, height)
                                         All values normalized to [0, 1]
    
    Raises:
        ValueError: If mask is empty or has invalid dimensions
    """
    
    # Handle input validation
    if not isinstance(msk, np.ndarray):
        raise ValueError("Mask must be a numpy array")
    
    # Handle 3D masks (take any channel with non-zero values)
    if msk.ndim == 3:
        # Combine all channels - if any channel has non-zero, consider it foreground
        mask_2d = np.any(msk > 0, axis=2).astype(np.uint8)
    elif msk.ndim == 2:
        mask_2d = msk.copy()
    else:
        raise ValueError(f"Mask must be 2D or 3D, got {msk.ndim}D")
    
    # Get mask dimensions
    h, w = mask_2d.shape
    
    # Binarize mask (handle different value ranges)
    # Find the maximum value to determine the range
    max_val = mask_2d.max()
    if max_val == 0:
        raise ValueError("Mask is completely empty (all zeros)")
    
    # Binarize: anything > 0 becomes foreground
    binary_mask = (mask_2d > 0).astype(np.uint8)
    
    # Find bounding box coordinates
    # Get all non-zero positions
    nonzero_pos = np.where(binary_mask > 0)
    
    if len(nonzero_pos[0]) == 0:
        raise ValueError("No foreground pixels found in mask")
    
    # Get min/max coordinates
    y_min, y_max = nonzero_pos[0].min(), nonzero_pos[0].max()
    x_min, x_max = nonzero_pos[1].min(), nonzero_pos[1].max()
    
    # Calculate current bbox dimensions
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    
    # Apply margin
    if isinstance(margin, float) and 0 <= margin <= 1:
        # Percentage margin
        margin_x = int(bbox_width * margin)
        margin_y = int(bbox_height * margin)
    else:
        # Pixel margin
        margin_x = margin_y = int(margin)
    
    # Expand bounding box with margin
    x_min_margin = max(0, x_min - margin_x)
    x_max_margin = min(w - 1, x_max + margin_x)
    y_min_margin = max(0, y_min - margin_y)
    y_max_margin = min(h - 1, y_max + margin_y)
    
    # Calculate final bbox dimensions
    final_width = x_max_margin - x_min_margin + 1
    final_height = y_max_margin - y_min_margin + 1
    
    # Calculate center coordinates
    x_center_pixel = x_min_margin + final_width / 2
    y_center_pixel = y_min_margin + final_height / 2
    
    # Normalize to [0, 1] (YOLO format)
    x_center = x_center_pixel / w
    y_center = y_center_pixel / h
    width_norm = final_width / w
    height_norm = final_height / h
    
    # Ensure values are within [0, 1]
    x_center = np.clip(x_center, 0, 1)
    y_center = np.clip(y_center, 0, 1)
    width_norm = np.clip(width_norm, 0, 1)
    height_norm = np.clip(height_norm, 0, 1)
    
    return x_center, y_center, width_norm, height_norm

def get_maskfromRGBA(img):
    """
    Extract the mask from an RGBA image.
    Args:
        img: PIL Image in RGBA format
    Returns:
        mask: PIL Image in grayscale format
    """
    # Convert to numpy array
    img_arr = np.array(img)
    
    # Extract the alpha channel as mask
    mask = img_arr[:, :, 3]
    # Binarize the mask
    mask = (mask > 0).astype(np.uint8) * 255  # Convert to binary mask (0 or 255)
    # Convert mask to PIL Image in grayscale mode
    mask = Image.fromarray(mask.astype(np.uint8), mode='L')
    return mask
def get_RGBAfrommask(img_arr, msk_arr):
    """
    Convert a PIL Image to RGBA format using a reference mask.
    Args:
        img_arr: numpy array RGBA format
        msk: numpy array of 2D or only 1 channel (mask)
    Returns:
        rgba_img: numpy array
    """
    # check if the msk and img is same size in 2D
    assert img_arr.shape[:2] == msk_arr.shape[:2], "Image and mask must have same height and width"
    assert img_arr.ndim == 3, "Image array must be 3D"
    msk_arr = binarize_uint8(msk_arr)  # Ensure mask is binary (0 or 255)
    # if img array did not have alpha channel, add it
    if img_arr.shape[2] != 4:
        # Create an alpha channel based on the mask
        alpha_channel = msk_arr
        # Concatenate the alpha channel to the image array
        rgba_img = np.concatenate((img_arr, alpha_channel[:, :, np.newaxis]), axis=2)
    else:
        # If img_arr already has 4 channels, replace it with the mask
        rgba_img = img_arr.copy()
        rgba_img[:, :, 3] = msk_arr.squeeze()  # Use mask as alpha channel
    
    return rgba_img


def binarize_uint8(arr, threshold=0.5):
    """
    Binarize a numpy array based on a threshold.
    
    Args:
        arr (np.ndarray): Input array to binarize.
        threshold (float): Threshold value for binarization.
        
    Returns:
        np.ndarray: Binarized array with values 0 or 255.
    """
    max_val = arr.max()
    min_val = arr.min()
    threshpoint = (max_val * threshold + min_val * (1 - threshold))/2
    # Binarize the array
    binarized_arr = (arr >= threshpoint).astype(np.uint8) * 255
    return binarized_arr
    
def binarize_tensor(tensor, threshold=0.5):
    """
    Binarize a tensor based on a threshold, applied per channel.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (C, H, W) to binarize.
        threshold (float): Threshold value for binarization (0.0 to 1.0).
        
    Returns:
        torch.Tensor: Binarized tensor with values 0 or 1, same shape as input.
    """
    C, H, W = tensor.shape
    binarized_tensor = torch.zeros_like(tensor)
    
    # Process each channel separately
    for c in range(C):
        channel = tensor[c]  # Shape: (H, W)
        
        # Find min and max values for this channel
        max_val = channel.max()
        min_val = channel.min()
        
        # Calculate threshold point (same formula as reference)
        threshpoint = (max_val * threshold + min_val * (1 - threshold)) / 2
        
        # Binarize: >= threshold becomes 1, < threshold becomes 0
        binarized_tensor[c] = (channel >= threshpoint).float()
    
    return binarized_tensor

class KiboDataset(Dataset):
    """
    KiboDataset if you call it it can autogenerate the dataset for training
    Args:
        datainfo: Dictionary containing dataset information
                  {
                      'Blank_image_size': (width, height),
                      'templateresize': (width, height),
                      'All_itemclass': [list of item classes],
                      'Treasure_item': path to treasure items,
                      'Landmark_item': path to landmark items,
                      'Lost_treasure_rate': float,
                      'Targetitem_rate': float,
                      'Batch_number': int,
                        'Overlap_rate': float,
                  }
        transforminfo: Dictionary containing transformation information
                      {
                          'Augmentation': augmentation parameters,
                          'Max_landmark_items': int,
                          'Outer_effect': float,
                          'Output_transform': transformation parameters
                      }
        In augmentation parameters
            augment_dict: dictionary with the key structure:
            {
                'inner': {
                    'flip': bool,
                    'rotate': bool,
                    'scale': bool,
                    'flip_rate': float,
                    'rotate_range': (min_angle, max_angle),
                    'scale_range': (min_scale, max_scale)
                },
                'outer': {
                    'brightness': float,
                    'contrast': float,
                    'sharpness': float,
                    'gaussian_noise': float,
                    'gaussian_blur': float,
                    'motion_blur': float,
                    'gradient_brightness': float,
                    'perspective': float,
                    'brightness_range': (min_brightness, max_brightness),
                    'contrast_range': (min_contrast, max_contrast),
                    'sharpness_range': (min_sharpness, max_sharpness),
                    'blur_kernel_size': (height, width),
                    'blur_sigma': (min_sigma, max_sigma),
                    'motion_blur_angle_range': (min_angle, max_angle),
                    'motion_blur_distance_range': (min_distance, max_distance),
                    'perspective_distortion': float,
                    'gaussian_noise_list': [ mean, std_min, std_max ]
                }
            }


    """
    def __init__(self, datainfo, transforminfo):
        
        self.blank_image_size = datainfo['Blank_image_size']
        self.template_size = datainfo['templateresize']
        # load the datapath list
        self.all_itemclass = datainfo['All_itemclass']
        self.treasure_items_path = datainfo['Treasure_item']
        self.landmark_items_path = datainfo['Landmark_item']
        
        # set the possibility of treasure item
        self.lost_treasrue_rate = datainfo['Lost_treasure_rate']
        self.targetitem_rate = datainfo['Targetitem_rate']
        self.overlap_rate = datainfo['Overlap_rate'] 

        # set the image total number perbatch
        self.batch_number = datainfo['Batch_number']

        # augmentation info
        self.aug = transforminfo['Augmentation']
        self.maxlndmark = transforminfo['Max_landmark_items']
        assert self.maxlndmark > 0, "Max landmark items must be greater than 0"
        self.outer_effect = transforminfo['Outer_effect']  # float the outer effect happen rate
        
        # transform info currently not used
        self.transforminfo = transforminfo['Output_transform']

        self.all_yolo_list = []
        self.all_result_img = []
        self.all_class_list = []
    


    def generate_target_item(self, landmark_items_path_list, treasure_items_path):
        """
        Generate a target item with two landmark items and one treasure item
        Args:
        landmark_items_path_list: List of landmark items path
        treasure_items_path: Path of the treasure item

        Returns:
            target_yolo_list: List of tuples containing (item_id, x_center, y_center, width, height, item_class, final_img_arr, final_msk_arr)
            taeget_result_msk: Resulting mask image, np.ndarray 
            target_result_img: Resulting image, np.ndarray
            target_class_list: List of item classes
        """
        all_itemclass = self.all_itemclass
        target_item_list = []
        target_class_list = []
        aug_dict = self.aug
        AugTrans = Augmentationtransform2D(aug_dict, aug_type='inner')
        # load the treasure item and landmark items
        treasure_item = Image.open(treasure_items_path).convert('RGBA')
        landmark_items = [Image.open(item).convert('RGBA') for item in landmark_items_path_list]

        # Resize the treasure item and landmark items to the template size use bilinear interpolation
        treasure_item = treasure_item.resize(self.template_size, Image.BILINEAR)
        landmark_items = [item.resize(self.template_size, Image.BILINEAR) for item in landmark_items]
        
        fname_list = [os.path.basename(treasure_items_path)] + [os.path.basename(item) for item in landmark_items_path_list]
        img_list = [treasure_item] + landmark_items
        msk_list = [get_maskfromRGBA(item) for item in img_list]

        # make sure the img and msk are all uint8 and have the same size
        assert all(img.size == treasure_item.size for img in img_list), "All images must have the same size"
        assert all(msk.size == treasure_item.size for msk in msk_list), "All masks must have the same size"
        assert all(img.mode == 'RGBA' for img in img_list), "All images must be in RGBA mode"
        assert all(msk.mode == 'L' for msk in msk_list), "All masks must be in L mode"


        # generate the augmented images and masks for each landmark and treasure item
        for img_PIL, msk_PIL, fname in zip(img_list, msk_list, fname_list):
            # resize the img_PIL and msk_PIL to the template size
            
            item_class = fname.split('.')[0]  # Use the filename without extension as class label
            item_id = all_itemclass.index(item_class) 
            target_class_list.append(item_class)
            img_arr = np.array(img_PIL)
            # make img_arr each channel either 0 or 255
            img_arr = binarize_uint8(img_arr)  # Ensure image is binary (0 or 255)
            msk_arr = np.array(msk_PIL)

            # make msk_arr to 3D array
            if msk_arr.ndim == 2:
                # If mask is 2D, convert it to 3D by adding a channel dimension
                msk_arr = msk_arr[:, :, np.newaxis]
            elif msk_arr.ndim == 3 and msk_arr.shape[2] == 1:
                # If mask is already 3D with one channel, no need to change
                pass
            else:
                raise ValueError("Mask array must be 2D or 3D with one channel")
            # check if they are the same shape
            assert img_arr.ndim == 3, "Image array must be 3D"
            assert msk_arr.ndim == 3, "Mask array must be 2D"
            assert img_arr.shape[:2] == msk_arr.shape[:2], "Image and mask must have the same height and width"

            # apply the augmentation
            new_img_arr, new_msk_arr = AugTrans(img_arr, msk_arr)
            # check the new_img_arr is [0, 255] and new_msk_arr is [0, 255] if not, renormalize scale max to 255
            new_img_arr = (new_img_arr - new_img_arr.min()) / (new_img_arr.max() - new_img_arr.min()) * 255
            new_img_arr = binarize_uint8(new_img_arr)  # Convert to binary image (0 or 255)
            new_msk_arr = (new_msk_arr - new_msk_arr.min()) / (new_msk_arr.max() - new_msk_arr.min()) * 255

            # convert the new image and mask array to PIL Image img(uint8) -> rgba msk -> L(uint8) 
            # new_img = Image.fromarray(new_img_arr.astype(np.uint8), mode='RGBA')
            # new_msk = Image.fromarray(new_msk_arr.astype(np.uint8).squeeze(), mode='L')

            # turn new image to RGBA format
            new_img_arr = get_RGBAfrommask(new_img_arr, new_msk_arr)

            # turn the mask to YOLO format
            x_center, y_center, width, height = getbboxfrommsk(new_msk_arr)

            # make new_img and new_msk to np.array
            final_img_arr = new_img_arr
            # final_msk_arr = new_msk_arr
            final_msk = get_maskfromRGBA(new_img_arr)  # Convert back to mask format
            # wrap item_class, x_center, y_center, width, height new_img into a dictionary
            item_data = (item_id, x_center, y_center, width, height, item_class, final_img_arr, final_msk)
            target_item_list.append(item_data)
        

        # Start to put the item in itemlist into the blank image (RGB)
        blank_image = 255*np.ones((self.blank_image_size[0], self.blank_image_size[1], 3), dtype=np.uint8)
        new_yolo_list, result_msk, result_img = fitintoblank(blank_image, target_item_list, 
                                                            overlap= False)

        result_imgrgb = cv2.cvtColor(result_img, cv2.COLOR_RGBA2RGB)

        target_yolo_list = new_yolo_list
        taeget_result_msk = result_msk
        target_result_img = result_imgrgb


        return target_yolo_list, taeget_result_msk, target_result_img, target_class_list
        

    def generate_lost_item(self, lost_item_path, lost_item_number, treasure_items_path = None):
        """
        Generate a lost item with one landmark item
        Args:
            lost_items_path: List of lost items path
            lost_item_number: Number of lost items to generate
            treasure_items_path: Path of the treasure item (optional)
        """
        all_itemclass = self.all_itemclass
        lost_item_list = []
        lost_class_list = []
        aug_dict = self.aug
        AugTrans = Augmentationtransform2D(aug_dict, aug_type='inner')
        
        # load the lost item_path
        lost_item = Image.open(lost_item_path).convert('RGBA')
        # Resize the lost item to the template size
        lost_item = lost_item.resize(self.template_size, Image.BILINEAR)
        # item_class = os.path.basename(lost_item_path).split('.')[0]  # Use the filename without extension as class label
        # item_id = all_itemclass.index(item_class)  # Get the index of the item class

        # fname_list = [os.path.basename(treasure_items_path)]*lost_item_number
        landmark_img_PIL = copy.deepcopy(lost_item)
        landmark_msk_PIL = get_maskfromRGBA(landmark_img_PIL)

        item_class = os.path.basename(lost_item_path).split('.')[0]  # Use the filename without extension as class label
        item_id = all_itemclass.index(item_class)

        assert landmark_img_PIL.mode == 'RGBA' , "All images must be in RGBA mode"
        assert landmark_msk_PIL.mode == 'L' , "All masks must be in L mode"

        # generate the augmented images and masks for each landmark and treasure item
        for _ in range(lost_item_number):
            lost_class_list.append(item_class)
            img_arr = np.array(landmark_img_PIL)
            # Binarize the image array (0 or 255)
            img_arr = binarize_uint8(img_arr)  # Convert to binary image (0 or 255)                
            msk_arr = np.array(landmark_msk_PIL)



            # make msk_arr to 3D array
            if msk_arr.ndim == 2:
                # If mask is 2D, convert it to 3D by adding a channel dimension
                msk_arr = msk_arr[:, :, np.newaxis]
            elif msk_arr.ndim == 3 and msk_arr.shape[2] == 1:
                # If mask is already 3D with one channel, no need to change
                pass
            else:
                raise ValueError("Mask array must be 2D or 3D with one channel")
            # check if they are the same shape
            assert img_arr.ndim == 3, "Image array must be 3D"
            assert msk_arr.ndim == 3, "Mask array must be 2D"
            assert img_arr.shape[:2] == msk_arr.shape[:2], "Image and mask must have the same height and width"

            # apply the augmentation
            new_img_arr, new_msk_arr = AugTrans(img_arr, msk_arr)
            # check the new_img_arr is [0, 255] and new_msk_arr is [0, 255] if not, renormalize scale max to 255
            new_img_arr = (new_img_arr - new_img_arr.min()) / (new_img_arr.max() - new_img_arr.min()) * 255
            new_img_arr = binarize_uint8(new_img_arr)  # Convert to binary image (0 or 255)
            new_msk_arr = (new_msk_arr - new_msk_arr.min()) / (new_msk_arr.max() - new_msk_arr.min()) * 255
            # new_msk_arr = (new_msk_arr > 0).astype(np.uint8) * 255  # Convert to binary mask (0 or 255)


            # # convert the new image and mask array to PIL Image img(uint8) -> rgba msk -> L(uint8) 
            # new_img = Image.fromarray(new_img_arr.astype(np.uint8), mode='RGBA')
            # new_msk = Image.fromarray(new_msk_arr.astype(np.uint8).squeeze(), mode='L')

            # turn new image to RGBA format
            new_img_arr = get_RGBAfrommask(new_img_arr, new_msk_arr)

            # turn the mask to YOLO format
            x_center, y_center, width, height = getbboxfrommsk(new_msk_arr)
            # make new_img and new_msk to np.array
            final_img_arr = new_img_arr
            # final_msk_arr = new_msk_arr
            final_msk = get_maskfromRGBA(new_img_arr)  # Convert back to mask format

            # wrap item_class, x_center, y_center, width, height new_img into a dictionary
            item_data = (item_id, x_center, y_center, width, height, item_class, final_img_arr, final_msk)
            lost_item_list.append(item_data)

        blank_image = 255* np.ones((self.blank_image_size[0], self.blank_image_size[1], 3), dtype=np.uint8)
        landmark_yolo_list, landmark_result_msk, landmark_result_img = fitintoblank(blank_image, lost_item_list, 
                                                                                    overlap= random.random() < self.overlap_rate)
        landmark_result_imgrgb = cv2.cvtColor(landmark_result_img, cv2.COLOR_RGBA2RGB)

        

        # if the treasure_items_path is provided, load the treasure item
        if treasure_items_path is not None:
            # if treasure_items_path is provided, load the treasure item
            treasure_item = Image.open(treasure_items_path).convert('RGBA')
            # Resize the treasure item to the template size
            treasure_item = treasure_item.resize(self.template_size, Image.BILINEAR)
            # add the treasure item to the result image
            treasure_img_PIL = copy.deepcopy(treasure_item)
            treasure_msk_PIL = get_maskfromRGBA(treasure_img_PIL)

            treasure_item_class = os.path.basename(treasure_items_path).split('.')[0]  # Use the filename without extension as class label
            treasure_item_id = all_itemclass.index(treasure_item_class)  # Get the index of the item class
            lost_class_list.append(treasure_item_class)
            img_arr = np.array(treasure_img_PIL)
            img_arr = binarize_uint8(img_arr)  # Convert to binary image (0 or 255)
            msk_arr = np.array(treasure_msk_PIL)

            if msk_arr.ndim == 2:
                # If mask is 2D, convert it to 3D by adding a channel dimension
                msk_arr = msk_arr[:, :, np.newaxis]
            elif msk_arr.ndim == 3 and msk_arr.shape[2] == 1:
                # If mask is already 3D with one channel, no need to change
                pass
            else:
                raise ValueError("Mask array must be 2D or 3D with one channel")
            # check if they are the same shape
            assert img_arr.ndim == 3, "Image array must be 3D"
            assert msk_arr.ndim == 3, "Mask array must be 2D"
            assert img_arr.shape[:2] == msk_arr.shape[:2], "Image and mask must have the same height and width"

            # apply the augmentation
            new_img_arr, new_msk_arr = AugTrans(img_arr, msk_arr)
            # check the new_img_arr is [0, 255] and new_msk_arr is [0, 255] if not, renormalize scale max to 255
            new_img_arr = (new_img_arr - new_img_arr.min()) / (new_img_arr.max() - new_img_arr.min()) * 255
            new_img_arr = binarize_uint8(new_img_arr)
            new_msk_arr = (new_msk_arr - new_msk_arr.min()) / (new_msk_arr.max() - new_msk_arr.min()) * 255
            # convert the new image and mask array to PIL Image img(uint8) -> rgba msk -> L(uint8)
            # new_img = Image.fromarray(new_img_arr.astype(np.uint8), mode='RGBA')
            # new_msk = Image.fromarray(new_msk_arr.astype(np.uint8).squeeze(), mode='L')
            # turn new image to RGBA format
            new_img_arr = get_RGBAfrommask(new_img_arr, new_msk_arr)
            # turn the mask to YOLO format
            x_center, y_center, width, height = getbboxfrommsk(new_msk_arr)

            # wrap item_class, x_center, y_center, width, height new_img into a dictionary
            final_img_arr = new_img_arr
            # final_msk_arr = new_msk_arr
            final_msk = get_maskfromRGBA(new_img_arr)  # Convert back to mask format

            treasure_item_data = (treasure_item_id, x_center, y_center, width, height, treasure_item_class, final_img_arr, final_msk)
            # make treasure item list
            treasure_item_list = [treasure_item_data]
            # fit the treasure item into the blank image
            treasure_yolo_list, treasure_result_msk, treasure_result_img = fitintoblank(landmark_result_imgrgb,
                                                                                        treasure_item_list, 
                                                                                        overlap=False,
                                                                                        existing_mask=landmark_result_msk,
                                                                                        existing_yolo_list= landmark_yolo_list)
            
            treasure_result_imgrgb = cv2.cvtColor(treasure_result_img, cv2.COLOR_RGBA2RGB)
            lost_yolo_list = treasure_yolo_list
            lost_treasure_result_msk = treasure_result_msk
            lost_treasure_result_img = treasure_result_imgrgb

        else:
            lost_yolo_list = landmark_yolo_list
            lost_treasure_result_msk = landmark_result_msk
            lost_treasure_result_img = landmark_result_imgrgb
        
        return lost_yolo_list, lost_treasure_result_msk, lost_treasure_result_img, lost_class_list
        
    @staticmethod
    def dealwith_outer_msk(result_msk):
        # get bbox from each layer of the result_msk
        assert result_msk.ndim == 3, "Result mask must be 3D"
        x_centers, y_centers, widths, heights = [], [], [], []
        if result_msk.shape[2] == 1:
            x_center, y_center, width, height = getbboxfrommsk(result_msk)
            x_centers.append(x_center)
            y_centers.append(y_center)
            widths.append(width)
            heights.append(height)
        else:
            for i in range(result_msk.shape[2]):
                # keep msk in 3d
                msk = result_msk[:, :, i:i+1]
                x_center, y_center, width, height = getbboxfrommsk(msk)
                x_centers.append(x_center)
                y_centers.append(y_center)
                widths.append(width)
                heights.append(height)

        # output each as tuple
        return tuple(x_centers), tuple(y_centers), tuple(widths), tuple(heights)

    def initialize_batchdata(self):
        """
        Create all data for training (raw image only contain the overlapped, translated, rotated, scaled images)
        if the image is target item
        must contain two types landmark items and one treasure item
        if is lost item
        must contain only one type of landmark item might exist one type of landmark item
        """
        self.all_yolo_list = []
        self.all_result_img = []
        self.all_class_list = []
        all_yolo_list = []
        all_result_img = []
        all_class_list = []
        for i in range(self.batch_number):
            # print(f"Generating batch {i+1}/{self.batch_number}...")
            # first check is target item or lost item
            if random.random() < self.targetitem_rate:
                print("Generating target item as " , i, " th image")
                # target item (must not overlap between each item)
                # print("Generate target item")
                # pick the random treasure item to load for the target item
                treasure_item = random.choice(self.treasure_items_path)
                # Pick two randomm landmark items to load for the target item 
                landmark_items = random.sample(self.landmark_items_path, 2)

                # Generate the image with the treasure item and landmark to make the image we use 
                yolo_list, result_msk, result_img, class_list = self.generate_target_item(landmark_items, treasure_item)

            else:
                # Lost item
                # print("Generate lost item")
                print("Generating lost item as " , i, " th image")
                landmark_item = random.choice(self.landmark_items_path)
                # pick the random lost item number
                lost_item_number = random.randint(1, self.maxlndmark)
                if random.random() < self.lost_treasrue_rate:
                    # if the lost item has a treasure item, pick one
                    treasure_item = random.choice(self.treasure_items_path)
                    yolo_list, result_msk, result_img, class_list = self.generate_lost_item(landmark_item, lost_item_number, treasure_item)
                else:
                    # if the lost item has no treasure item, just use the landmark item
                    yolo_list, result_msk, result_img, class_list = self.generate_lost_item(landmark_item, lost_item_number)
                # print( result_img.shape, result_msk.shape)


            # """
            # if the outer_effect is not 0, apply the outer augmentation
            if self.outer_effect > 0:
                # apply the outer augmentation
                aug_dict = self.aug
                AugTrans = Augmentationtransform2D(aug_dict, aug_type='outer')
                result_img, result_msk = AugTrans(result_img, result_msk)
                # print("result_msk shape: ", result_msk.shape)
                # get new bounding box from the result mask
                new_x_centers, new_y_centers, new_widths, new_heights = self.dealwith_outer_msk(result_msk)
                # new_yolo_list = copy.deepcopy(yolo_list)
                # update the yolo_list with the new bounding box
                # if type(new_x_centers) is tuple:
                for idx, (item_id, _, _, _, _) in enumerate(yolo_list):
                    # update the yolo_list with the new bounding box
                    yolo_list[idx] = (item_id, new_x_centers[idx], new_y_centers[idx], new_widths[idx], new_heights[idx])

                # convert the result_img_arr and result_msk_arr to PIL Image
                # result_img = Image.fromarray(result_img_arr.astype(np.uint8), mode='RGB')
                # result_msk = Image.fromarray(result_msk_arr.astype(np.uint8), mode='L')
            # else:
            #     # if no outer effect, just convert to PIL Image
            #     result_img = Image.fromarray(result_img.astype(np.uint8), mode='RGB')
            #     result_msk = Image.fromarray(result_msk.astype(np.uint8), mode='L')
            # """
            # store the yolo_list, result_img, result_msk, target_class_list
            all_yolo_list.append(yolo_list)
            all_result_img.append(result_img)
            all_class_list.append(class_list)
        
        self.all_yolo_list = all_yolo_list
        self.all_result_img = all_result_img
        self.all_class_list = all_class_list


    def __getitem__(self, idx):
        
        image = self.all_result_img[idx]
        label = self.all_yolo_list[idx]
        clss = self.all_class_list[idx]
        return image, label, clss

    def __len__(self):
        return len(self.all_result_img)



