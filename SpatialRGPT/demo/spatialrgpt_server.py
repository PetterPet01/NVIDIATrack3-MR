import base64
import io
import time
import uuid
import uvicorn # For running the server

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

# --- Imports from the original Streamlit app ---
import copy
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add spatialrgpt main path
try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

    from utils.markdown import process_markdown
    from utils.som import draw_mask_and_number_on_image

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, process_images, process_regions, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
except ImportError as e:
    print(
        f"Failed to import necessary modules from MealsRetrieval project. Ensure correct project structure and PYTHONPATH. Error: {e}")
    sys.exit(1)

# --- Define default colors for markdown ---
DEFAULT_MARKDOWN_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (0, 128, 128), (128, 0, 128), (255, 128, 0), (255, 0, 128), (0, 255, 128),
    (128, 255, 0), (0, 128, 255), (128, 0, 255), (255, 128, 128), (128, 255, 128),
    (128, 128, 255), (255, 255, 128), (128, 255, 255), (255, 128, 255)
]

# --- Environment Variable Checks (SAM and SpatialRGPT only) ---
SAM_CKPT_PATH = os.environ.get("SAM_CKPT_PATH")
SPATIALRGPT_MODEL_PATH_ENV = os.environ.get("SPATIALRGPT_MODEL_PATH", "a8cheng/SpatialRGPT-VILA1.5-8B")
SPATIALRGPT_MODEL_NAME_ENV = os.environ.get("SPATIALRGPT_MODEL_NAME", "SpatialRGPT-VILA1.5-8B")

# --- Global model placeholders ---
sam_predictor_global = None
spatialrgpt_tokenizer_global = None
spatialrgpt_model_global = None
spatialrgpt_image_processor_global = None
spatialrgpt_context_len_global = None

# --- Pydantic Models for OpenAI Compatibility (defined earlier for type hints) ---
class ChatMessageContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ChatMessageContentItem]]

# --- Model Loading Functions (modified for server startup) ---
def load_sam_predictor():
    global sam_predictor_global
    if not SAM_CKPT_PATH:
        print("Warning: SAM_CKPT_PATH environment variable not set. Segmentation with SAM will not be available.")
        return
    if not os.path.isfile(SAM_CKPT_PATH):
        print(f"Error: SAM checkpoint not found at {SAM_CKPT_PATH}. SAM will not be available.")
        return

    try:
        from segment_anything_hq import SamPredictor, sam_model_registry
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT_PATH).to(device).eval()
        sam_predictor_global = SamPredictor(sam)
        print("SAM model loaded successfully!")
    except Exception as e:
        print(f"Error loading SAM model: {e}")

def load_spatialrgpt_model(model_path, model_name):
    global spatialrgpt_tokenizer_global, spatialrgpt_model_global, \
           spatialrgpt_image_processor_global, spatialrgpt_context_len_global

    if not model_path or not model_name:
        print("Model path or model name not provided for MealsRetrieval.")
        return

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading MealsRetrieval model '{model_name}' from '{model_path}' onto device '{device}'...")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_name, device_map=None
        )
        if model.device.type == 'cpu' and device == 'cuda':
             model = model.to(device)
        model = model.eval()

        spatialrgpt_tokenizer_global = tokenizer
        spatialrgpt_model_global = model
        spatialrgpt_image_processor_global = image_processor
        spatialrgpt_context_len_global = context_len
        print(f"MealsRetrieval model '{model_name}' loaded successfully! Device: {model.device}")
    except Exception as e:
        print(f"Error loading MealsRetrieval model '{model_name}': {e}")


# --- Core Logic Functions (Adapted for API) ---
def segment_using_boxes_api(raw_image_np: np.ndarray, boxes: List[List[int]], use_segmentation=True):
    if sam_predictor_global is None and use_segmentation:
        print("Warning: SAM predictor not loaded. Cannot perform SAM segmentation, returning box masks.")
        use_segmentation = False

    orig_h, orig_w = raw_image_np.shape[:2]
    bboxes_np = np.array(boxes)
    seg_masks_for_vlm = []

    if bboxes_np.size == 0:
        return []

    if use_segmentation and sam_predictor_global:
        sam_predictor_global.set_image(raw_image_np)
        for bbox in bboxes_np:
            masks_sam, scores, _ = sam_predictor_global.predict(box=bbox, multimask_output=True)
            seg_masks_for_vlm.append(masks_sam[np.argmax(scores)].astype(np.uint8))
    else:
        for bbox in bboxes_np:
            zero_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, bbox)
            zero_mask[y1:y2, x1:x2] = 1
            seg_masks_for_vlm.append(zero_mask)
    return seg_masks_for_vlm

def inference_vlm_api(
    openai_messages: List[ChatMessage], # MODIFIED: Type hint changed
    raw_image_pil: Optional[Image.Image],
    seg_masks: List[np.ndarray],
    colorized_depth_pil: Optional[Image.Image],
    conv_mode_name: str,
    use_depth_info: bool,
    use_bfloat16: bool,
    temperature: float,
    max_new_tokens: int
):
    if not all([spatialrgpt_tokenizer_global, spatialrgpt_model_global, spatialrgpt_image_processor_global]):
        raise HTTPException(status_code=503, detail="SpatialRGPT model not loaded or not ready.")

    conv = conv_templates[conv_mode_name].copy()
    current_user_prompt_text_for_llava = "" # Text of the last user message as it appears in the conversation
    has_image_token_been_added_to_conv = False # Tracks if IMAGE_TOKEN was added by message processing loop

    for pydantic_msg_obj in openai_messages: # pydantic_msg_obj is a ChatMessage instance
        role = pydantic_msg_obj.role
        raw_content = pydantic_msg_obj.content # Union[str, List[ChatMessageContentItem]]

        processed_text_for_conv = ""
        message_contains_image_trigger = False # Does *this specific message* contain an image_url?

        if isinstance(raw_content, str):
            processed_text_for_conv = raw_content
        elif isinstance(raw_content, list): # raw_content is List[ChatMessageContentItem]
            temp_text_parts = []
            for item in raw_content: # item is ChatMessageContentItem
                if item.type == "text":
                    temp_text_parts.append(item.text or "")
                elif item.type == "image_url" and role == "user": # Image token relevant for user messages
                    message_contains_image_trigger = True
            processed_text_for_conv = "".join(temp_text_parts)
        
        # Prepend image token if this user message has an image and it's the first time we're adding it
        if role == "user" and message_contains_image_trigger and not has_image_token_been_added_to_conv:
            processed_text_for_conv = DEFAULT_IMAGE_TOKEN + "\n" + processed_text_for_conv
            has_image_token_been_added_to_conv = True

        # Add to conversation
        if role == "system":
            conv.set_system_message(processed_text_for_conv)
        elif role == "user":
            conv.append_message(conv.roles[0], processed_text_for_conv)
            current_user_prompt_text_for_llava = processed_text_for_conv # This message's text is the latest user prompt
        elif role == "assistant":
            conv.append_message(conv.roles[1], processed_text_for_conv)
        else:
            print(f"Warning: Unsupported role '{role}' in message history. Ignoring.")

    # After the loop, if no image token was added via message content,
    # but an image is provided (raw_image_pil), and the last message in conv is user's:
    if not has_image_token_been_added_to_conv and raw_image_pil:
        if conv.messages and conv.messages[-1][0] == conv.roles[0]: # Last message in conv is user
            last_user_msg_text_in_conv = conv.messages[-1][1]
            if DEFAULT_IMAGE_TOKEN not in last_user_msg_text_in_conv:
                modified_last_user_msg = DEFAULT_IMAGE_TOKEN + "\n" + last_user_msg_text_in_conv
                conv.messages[-1] = (conv.roles[0], modified_last_user_msg)
                current_user_prompt_text_for_llava = modified_last_user_msg # Update as this is now the effective last user prompt

    # final_user_input_str_for_regions: original text of the last user message for region tag extraction
    final_user_input_str_for_regions = ""
    if openai_messages and openai_messages[-1].role == "user":
        last_message_raw_content = openai_messages[-1].content
        if isinstance(last_message_raw_content, str):
            final_user_input_str_for_regions = last_message_raw_content
        elif isinstance(last_message_raw_content, list): # List[ChatMessageContentItem]
            for item in last_message_raw_content:
                if item.type == "text":
                    final_user_input_str_for_regions += (item.text or "")
    
    # query_base_for_llava: Derived from the last user message text in the conversation (current_user_prompt_text_for_llava)
    if use_depth_info:
        query_base_for_llava = re.sub(r"<region\d+>", "<mask> <depth>", current_user_prompt_text_for_llava)
    else:
        query_base_for_llava = re.sub(r"<region\d+>", "<mask>", current_user_prompt_text_for_llava)

    # Update the last user message in the conversation if it was modified for region/depth tokens
    if conv.messages and conv.messages[-1][0] == conv.roles[0] and conv.messages[-1][1] != query_base_for_llava:
        conv.messages[-1] = (conv.roles[0], query_base_for_llava)

    conv.append_message(conv.roles[1], None) # Add placeholder for assistant's response
    prompt_for_tokenizer = conv.get_prompt()

    device = spatialrgpt_model_global.device
    selected_dtype = torch.float16
    if use_bfloat16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        selected_dtype = torch.bfloat16
    elif use_bfloat16 and torch.cuda.is_available():
        print("Warning: bfloat16 selected but not supported on this GPU. Using float16.")

    original_model_dtype = next(spatialrgpt_model_global.parameters()).dtype
    if original_model_dtype != selected_dtype and str(device) != 'cpu':
        spatialrgpt_model_global.to(dtype=selected_dtype)

    images_tensor = None
    if raw_image_pil:
        images_tensor = process_images([raw_image_pil], spatialrgpt_image_processor_global, spatialrgpt_model_global.config).to(
            device, dtype=selected_dtype if str(device) != 'cpu' else torch.float32)

    depths_tensor = None
    if use_depth_info and colorized_depth_pil:
        depths_tensor = process_images([colorized_depth_pil], spatialrgpt_image_processor_global, spatialrgpt_model_global.config).to(
            device, dtype=selected_dtype if str(device) != 'cpu' else torch.float32)
    elif use_depth_info and not colorized_depth_pil:
        print("Error: use_depth_info is true, but no colorized_depth_pil provided to inference_vlm_api. Depth will not be used.")

    current_input_region_tags = re.findall(r"<region(\d+)>", final_user_input_str_for_regions)
    current_input_region_indices_int = [int(tag) for tag in current_input_region_tags]

    final_masks_for_model = None
    if len(seg_masks) > 0 and len(current_input_region_indices_int) > 0:
        np_masks_for_processing = [m * 255 for m in seg_masks] # Ensure masks are 0-255 for process_regions
        _masks_tensor_all_available = process_regions(
            np_masks_for_processing, # Expects list of PIL Images or np arrays (H, W) or (H, W, C)
            spatialrgpt_image_processor_global,
            spatialrgpt_model_global.config
        ).to(device, dtype=selected_dtype if str(device) != 'cpu' else torch.float32)

        actual_mask_indices_to_pass_to_model = []
        for r_idx in current_input_region_indices_int:
            if 0 <= r_idx < _masks_tensor_all_available.size(0):
                actual_mask_indices_to_pass_to_model.append(r_idx)
            else:
                print(f"Warning: Region index {r_idx} in current prompt is out of bounds. It will be ignored.")
        if actual_mask_indices_to_pass_to_model:
            final_masks_for_model = _masks_tensor_all_available[actual_mask_indices_to_pass_to_model]

    input_ids = tokenizer_image_token(prompt_for_tokenizer, spatialrgpt_tokenizer_global, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, spatialrgpt_tokenizer_global, input_ids)

    with torch.inference_mode():
        output_ids = spatialrgpt_model_global.generate(
            input_ids,
            images=[images_tensor] if images_tensor is not None else None,
            depths=[depths_tensor] if depths_tensor is not None else None,
            masks=[final_masks_for_model] if final_masks_for_model is not None else None,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    if original_model_dtype != selected_dtype and str(device) != 'cpu':
        spatialrgpt_model_global.to(dtype=original_model_dtype)

    outputs_raw = spatialrgpt_tokenizer_global.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs_stripped = outputs_raw.strip()
    if outputs_stripped.endswith(stop_str):
        outputs_stripped = outputs_stripped[:-len(stop_str)]
    outputs_final_model = outputs_stripped.strip()

    mapping_dict = {str(out_idx): str(in_idx_str) for out_idx, in_idx_str in enumerate(current_input_region_tags)}
    remapped_outputs = outputs_final_model
    if mapping_dict:
        try:
            remapped_outputs = re.sub(r"\[([0-9]+)\]", lambda x: f"[{mapping_dict[x.group(1)]}]" if x.group(1) in mapping_dict else x.group(0), outputs_final_model)
        except KeyError as e: # Should ideally not happen if model only refers to provided indices
            print(f"Output remapping warning: Model referred to index {e} which was not in the mapping_dict derived from current prompt's regions {current_input_region_tags}.")
    return remapped_outputs

# --- FastAPI App Initialization ---
app = FastAPI(title="MealsRetrieval OpenAI-Compatible API")

@app.on_event("startup")
async def startup_event():
    print("Server startup: Loading models...")
    load_sam_predictor() # SAM is optional for segmentation
    load_spatialrgpt_model(SPATIALRGPT_MODEL_PATH_ENV, SPATIALRGPT_MODEL_NAME_ENV)
    if not spatialrgpt_model_global: # VLM is critical
        print("CRITICAL: SpatialRGPT model failed to load. API will not be functional.")
    else:
        print("SpatialRGPT model loaded. SAM model status printed above.")


# --- Pydantic Models for OpenAI Compatibility (already defined ChatMessage, ChatMessageContentItem) ---
class ImageOptions(BaseModel):
    regions_boxes: Optional[List[List[int]]] = Field(None, description="List of [x1,y1,x2,y2] boxes for <regionX> tags.")
    use_sam_segmentation: bool = Field(True, description="Use SAM for segmentation if boxes and SAM model are available.")
    process_provided_depth: bool = Field(False, description="Set to true if a depth image is provided in `depth_image_url` and should be used by the VLM.")
    depth_image_url: Optional[Dict[str, str]] = Field(None, description="URL for the client-provided depth image (e.g., data:image/png;base64,...). Required if `process_provided_depth` is true.")
    provided_seg_masks: Optional[List[str]] = Field(None, description="List of base64 encoded single-channel PNG strings for direct segmentation masks.")

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 512
    image_options: Optional[ImageOptions] = None
    conv_mode: Optional[str] = Field("llama_3", description="LLaVA conversation mode template name.")
    use_bfloat16: Optional[bool] = Field(True, description="Use bfloat16 for inference if supported.")

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

def decode_image_from_message_content(content_item: ChatMessageContentItem) -> Optional[Image.Image]:
    if content_item.type == "image_url" and content_item.image_url:
        url_data = content_item.image_url.get("url") # Use .get() for safety
        if url_data and url_data.startswith("data:image"):
            try:
                header, encoded = url_data.split(",", 1)
                image_data = base64.b64decode(encoded)
                return Image.open(io.BytesIO(image_data)).convert('RGB')
            except Exception as e:
                print(f"Error decoding base64 image from image_url: {e}")
                return None
    return None

def decode_mask_from_base64(base64_str: str) -> Optional[np.ndarray]:
    try:
        header, encoded = base64_str.split(",", 1)
        image_data = base64.b64decode(encoded)
        pil_image = Image.open(io.BytesIO(image_data))
        if pil_image.mode != 'L':
            pil_image = pil_image.convert('L')
        mask_np = np.array(pil_image)
        # Ensure mask values are appropriate (e.g., 0 or 1, or 0 or 255)
        # LLaVA's process_regions will handle PIL Images.
        # If masks are 0/1, they will be multiplied by 255 before process_regions in inference_vlm_api
        # If they are already 0/255, it's also fine.
        # Let's assume they come as 0/255 from client or are binarized.
        # For simplicity, we assume process_regions can handle the 'L' mode image as is.
        # The line `np_masks_for_processing = [m * 255 for m in seg_masks]` in inference_vlm_api
        # assumes input masks are 0/1. If they are already 0/255, this will make them too large.
        # Let's make sure `decode_mask_from_base64` returns 0/1 masks.
        mask_np = (mask_np > 127).astype(np.uint8) # Binarize to 0 or 1
        return mask_np
    except Exception as e:
        print(f"Error decoding base64 mask: {e}")
        return None

# --- API Endpoint ---
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest = Body(...)):
    if not all([spatialrgpt_tokenizer_global, spatialrgpt_model_global, spatialrgpt_image_processor_global]):
         raise HTTPException(status_code=503, detail="Core VLM models not loaded. Server not ready.")

    req_model_name = request.model
    conv_mode = request.conv_mode
    if conv_mode not in conv_templates:
        raise HTTPException(status_code=400, detail=f"Invalid conv_mode '{conv_mode}'. Available: {list(conv_templates.keys())}")

    raw_image_pil: Optional[Image.Image] = None
    # Iterate through messages to find the main image, usually in the last user message.
    # LLaVA typically expects one image associated with the conversation.
    # The image is extracted from the *first* image_url encountered in user messages (from latest to oldest).
    for msg in reversed(request.messages): # Check from latest message first
        if msg.role == "user" and isinstance(msg.content, list):
            for item in msg.content:
                if item.type == "image_url":
                    img = decode_image_from_message_content(item)
                    if img:
                        raw_image_pil = img
                        break # Found the main image
        if raw_image_pil:
            break

    seg_masks_for_vlm = [] # List of np.ndarray (H, W), values 0 or 1
    client_provided_depth_pil: Optional[Image.Image] = None
    raw_image_np = None
    use_depth_for_vlm_flag = False

    if raw_image_pil:
        raw_image_np = np.array(raw_image_pil)

    boxes_from_request = []
    use_sam_flag = True # Default from ImageOptions

    # Handle image_options
    if request.image_options:
        use_sam_flag = request.image_options.use_sam_segmentation # User preference for SAM

        if request.image_options.provided_seg_masks:
            if not raw_image_pil:
                raise HTTPException(status_code=400, detail="`provided_seg_masks` were given, but no main image was found in messages.")
            
            for i, b64_mask_str in enumerate(request.image_options.provided_seg_masks):
                mask_np = decode_mask_from_base64(b64_mask_str) # Returns 0/1 mask
                if mask_np is None:
                    raise HTTPException(status_code=400, detail=f"Failed to decode provided_seg_masks at index {i}.")
                
                if raw_image_np is not None and mask_np.shape[:2] != raw_image_np.shape[:2]:
                    print(f"Warning: Provided mask {i} shape {mask_np.shape[:2]} differs from image shape {raw_image_np.shape[:2]}. Resizing mask to image size.")
                    mask_pil_for_resize = Image.fromarray(mask_np.astype(np.uint8) * 255) # For resize, needs 0-255
                    resized_mask_pil = mask_pil_for_resize.resize((raw_image_np.shape[1], raw_image_np.shape[0]), Image.NEAREST)
                    mask_np = (np.array(resized_mask_pil) > 127).astype(np.uint8) # Back to 0/1

                seg_masks_for_vlm.append(mask_np) # Add 0/1 mask
            print(f"Using {len(seg_masks_for_vlm)} directly provided segmentation masks.")
        
        elif request.image_options.regions_boxes: # Only use boxes if no direct masks are given
            boxes_from_request = request.image_options.regions_boxes
            for box in boxes_from_request:
                if not (isinstance(box, list) and len(box) == 4 and all(isinstance(c, int) for c in box)):
                    raise HTTPException(status_code=400, detail="Invalid format for regions_boxes.")
            if raw_image_np is not None:
                if not sam_predictor_global and use_sam_flag:
                    print("Warning: SAM segmentation requested (or default) but SAM model not available. Will use bounding boxes as masks.")
                # segment_using_boxes_api returns 0/1 masks if SAM used, or 0/1 box masks
                seg_masks_for_vlm = segment_using_boxes_api(raw_image_np, boxes_from_request, use_segmentation=use_sam_flag)
            else: # No image, but boxes provided - this is an issue.
                raise HTTPException(status_code=400, detail="`regions_boxes` provided, but no main image found in messages.")
        
        # Depth processing
        if request.image_options.process_provided_depth:
            if not request.image_options.depth_image_url:
                raise HTTPException(status_code=400, detail="`process_provided_depth` is true, but `depth_image_url` was not provided.")
            if not raw_image_pil:
                raise HTTPException(status_code=400, detail="Depth processing requested, but no main image was found.")

            depth_content_item = ChatMessageContentItem(type="image_url", image_url=request.image_options.depth_image_url)
            decoded_depth_img = decode_image_from_message_content(depth_content_item)
            if not decoded_depth_img:
                raise HTTPException(status_code=400, detail="Failed to decode provided `depth_image_url`.")
            client_provided_depth_pil = decoded_depth_img
            use_depth_for_vlm_flag = True
        elif request.image_options.depth_image_url:
            print("Warning: `depth_image_url` provided, but `process_provided_depth` is false. Depth image will be ignored.")

    # Validate prompt consistency if no image is present
    if not raw_image_pil:
        last_user_prompt_text = ""
        if request.messages and request.messages[-1].role == "user":
            content = request.messages[-1].content
            if isinstance(content, str): last_user_prompt_text = content
            elif isinstance(content, list):
                for item_content in content:
                    if item_content.type == "text": last_user_prompt_text += (item_content.text or "")
        
        if re.search(r"<region\d+>", last_user_prompt_text) and not seg_masks_for_vlm: # If regions in prompt but no masks (from boxes or direct)
            raise HTTPException(status_code=400, detail="Prompt contains <regionX> tags, but no image was provided or no regions were defined via boxes/masks.")
        if use_depth_for_vlm_flag: # This implies process_provided_depth was true
            raise HTTPException(status_code=400, detail="Depth processing was enabled, but no main image was provided.")

    try:
        model_response_text = inference_vlm_api(
            openai_messages=request.messages, # Pass Pydantic models
            raw_image_pil=raw_image_pil,
            seg_masks=seg_masks_for_vlm, # List of 0/1 np.ndarrays
            colorized_depth_pil=client_provided_depth_pil,
            conv_mode_name=conv_mode,
            use_depth_info=use_depth_for_vlm_flag,
            use_bfloat16=request.use_bfloat16 if request.use_bfloat16 is not None else True,
            temperature=request.temperature,
            max_new_tokens=request.max_tokens
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error during VLM inference: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during VLM inference: {str(e)}")

    response_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    # Placeholder for actual token counting if needed
    usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    assistant_message = ChatMessage(role="assistant", content=model_response_text)
    choice = Choice(index=0, message=assistant_message, finish_reason="stop")

    return ChatCompletionResponse(
        id=response_id,
        created=created_time,
        model=req_model_name, # Echo back the requested model name
        choices=[choice],
        usage=usage
    )

if __name__ == "__main__":
    if not SAM_CKPT_PATH:
        print("INFO: SAM_CKPT_PATH environment variable is not set. SAM-based segmentation will not be available.")
    if not SPATIALRGPT_MODEL_PATH_ENV or not SPATIALRGPT_MODEL_NAME_ENV:
        print("CRITICAL: SPATIALRGPT_MODEL_PATH or SPATIALRGPT_MODEL_NAME environment variables not set. API cannot start.")
        sys.exit(1)
        
    uvicorn.run(app, host="0.0.0.0", port=8001)