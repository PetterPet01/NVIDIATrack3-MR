import base64
import io
import re
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def decode_base64_image(base64_string: str) -> Image.Image:
    """Decodes a base64 string into a PIL Image."""
    image_data = base64.b64decode(re.sub('^data:image/.+;base64,', '', base64_string))
    image = Image.open(io.BytesIO(image_data))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def get_depth_map(raw_image_np: np.ndarray, depth_model, depth_transform) -> np.ndarray:
    """Generates a colorized depth map from an RGB image."""
    if depth_model is None or depth_transform is None:
        raise ValueError("Depth model/transform not loaded.")

    orig_h, orig_w = raw_image_np.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    depth_input = depth_transform({"image": raw_image_np / 255.0})["image"]
    depth_input = torch.from_numpy(depth_input).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_depth = depth_model(depth_input)

    raw_depth = F.interpolate(raw_depth[None], (orig_h, orig_w), mode="bilinear", align_corners=False)[0, 0]
    raw_depth = raw_depth.cpu().numpy()

    min_val, max_val = raw_depth.min(), raw_depth.max()
    if max_val - min_val > 1e-6:
        raw_depth = (raw_depth - min_val) / (max_val - min_val) * 255.0
    else:
        raw_depth = np.zeros_like(raw_depth)
    
    raw_depth = raw_depth.astype(np.uint8)
    # Return 3-channel grayscale for VLM consistency
    return cv2.cvtColor(raw_depth, cv2.COLOR_GRAY2RGB)

def segment_using_boxes(raw_image_np: np.ndarray, boxes: list, sam_predictor) -> list:
    """Generates segmentation masks from bounding boxes using SAM."""
    if sam_predictor is None:
        raise ValueError("SAM predictor not loaded.")
    if not boxes:
        return []
    
    bboxes_np = np.array(boxes)
    seg_masks = []
    
    sam_predictor.set_image(raw_image_np)
    for bbox in bboxes_np:
        masks_sam, scores, _ = sam_predictor.predict(box=bbox, multimask_output=True)
        # Return a binary (0 or 1) uint8 mask, which is expected by the inference logic
        best_mask = masks_sam[np.argmax(scores)].astype(np.uint8)
        seg_masks.append(best_mask)
        
    return seg_masks