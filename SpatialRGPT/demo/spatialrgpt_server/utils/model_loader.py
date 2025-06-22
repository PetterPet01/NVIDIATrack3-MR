import os
import sys
import torch
import logging

# --- Environment Variable Checks ---
DEPTH_ANYTHING_PATH = os.environ.get("DEPTH_ANYTHING_PATH")
SAM_CKPT_PATH = os.environ.get("SAM_CKPT_PATH")
MODEL_PATH_DEFAULT = os.environ.get("SPATIALRGPT_MODEL_PATH")
MODEL_NAME_DEFAULT = os.environ.get("SPATIALRGPT_MODEL_NAME")

logging.basicConfig(level=logging.INFO)

def load_all_models():
    """Loads all models and returns them in a dictionary."""
    models = {
        "depth_model": None, "depth_transform": None,
        "sam_predictor": None,
        "tokenizer": None, "spatialrgpt_model": None, "image_processor": None,
        "context_len": None
    }
    
    # logging.info("Loading DepthAnything V2 model...")
    # models["depth_model"], models["depth_transform"] = load_depth_predictor()
    
    # logging.info("Loading SAM model...")
    # models["sam_predictor"] = load_sam_predictor()
    
    logging.info(f"Loading SpatialRGPT model '{MODEL_NAME_DEFAULT}'...")
    models["tokenizer"], models["spatialrgpt_model"], models["image_processor"], models["context_len"] = load_spatialrgpt_model(MODEL_PATH_DEFAULT, MODEL_NAME_DEFAULT)

    # Validate that all essential models are loaded
    if not all([models["tokenizer"], models["spatialrgpt_model"], models["image_processor"]]):
         raise RuntimeError("Failed to load the main SpatialRGPT model. The server cannot start.")
         
    logging.info("All models loaded successfully.")
    return models

def load_depth_predictor():
    if not DEPTH_ANYTHING_PATH or not os.path.isdir(DEPTH_ANYTHING_PATH):
        logging.error(f"DEPTH_ANYTHING_PATH ('{DEPTH_ANYTHING_PATH}') is not set or not a valid directory.")
        return None, None
    
    original_sys_path = list(sys.path)
    if DEPTH_ANYTHING_PATH not in sys.path:
        sys.path.insert(0, DEPTH_ANYTHING_PATH)

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
        from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize
        from torchvision.transforms import Compose
        import cv2

        model_configs = {'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}}
        depth_model_path = os.path.join(DEPTH_ANYTHING_PATH, "checkpoints", "depth_anything_v2_vitl.pth")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        depth_model = DepthAnythingV2(**model_configs['vitl'])
        depth_model.load_state_dict(torch.load(depth_model_path, map_location=device))
        depth_model = depth_model.to(device).eval()

        depth_transform = Compose([
            Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True,
                   ensure_multiple_of=14, resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        logging.info("Depth model loaded successfully.")
        return depth_model, depth_transform
    except Exception as e:
        logging.error(f"Error loading Depth model: {e}")
        return None, None
    finally:
        sys.path = original_sys_path

def load_sam_predictor():
    if not SAM_CKPT_PATH or not os.path.isfile(SAM_CKPT_PATH):
        logging.error(f"SAM_CKPT_PATH ('{SAM_CKPT_PATH}') is not set or not a valid file.")
        return None
    try:
        from segment_anything_hq import SamPredictor, sam_model_registry
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT_PATH).to(device).eval()
        sam_predictor = SamPredictor(sam)
        logging.info("SAM model loaded successfully.")
        return sam_predictor
    except Exception as e:
        logging.error(f"Error loading SAM model: {e}")
        return None

def load_spatialrgpt_model(model_path, model_name):
    if not model_path or not model_name:
        logging.error("SPATIALRGPT_MODEL_PATH or SPATIALRGPT_MODEL_NAME not provided.")
        return None, None, None, None
    try:
        from llava.model.builder import load_pretrained_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_name, device_map=None
        )
        model = model.to(device).eval()
        logging.info(f"SpatialRGPT model '{model_name}' loaded successfully.")
        return tokenizer, model, image_processor, context_len
    except Exception as e:
        logging.error(f"Error loading SpatialRGPT model '{model_name}': {e}")
        return None, None, None, None