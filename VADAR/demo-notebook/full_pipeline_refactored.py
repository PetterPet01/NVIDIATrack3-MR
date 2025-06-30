import os
import sys
import numpy as np
import json
from PIL import Image as PILImage, ImageDraw
from io import BytesIO
import base64
import time
import torch
from IPython.display import Markdown, display, Code, HTML
from rich.console import Console
from rich.syntax import Syntax
from rich.padding import Padding
from rich.style import Style
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
import pandas as pd
import re
import textwrap
import multiprocessing
from typing import Optional
import inspect
import sys
import traceback
# import sys
# sys.setrecursionlimit(100)
# import groundingdino.datasets.transforms as T_gd
# from groundingdino.util.inference import load_model, predict
from unik3d.models import UniK3D
import torchvision.transforms as TV_T
from transformers import AutoModelForCausalLM, AutoTokenizer
import google.generativeai as genai
import open3d as o3d

import pandas as pd
import google.generativeai as genai
import requests
import io
import queue
import threading
import contextlib
from PIL import Image
from google.api_core.exceptions import InternalServerError

import queue
from pathlib import Path
import spacy
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from wordfreq import word_frequency
import requests # For making HTTP requests
from PIL import Image # For image handling
import textwrap
from openai import OpenAI
from transformers import AutoModel
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import math
import cv2 # For resizing masks
import tempfile
import traceback
import signal
import shutil
import runpy
import linecache
import io
import concurrent.futures
import queue
import threading
import queue
import threading
import contextlib
import requests
import pandas as pd
import ast
import autopep8
import black

# import astor  # Use 'ast.unparse' instead if Python 3.9+

from typing import Union, Dict, List, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Add VADAR root to path
# Assuming the script is in 'VADAR/demo-notebook', vadar_root should be 'VADAR'
vadar_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if vadar_root not in sys.path:
    sys.path.insert(0, vadar_root)

# Import prompts and modules
from resources.prompts.predef_api import MODULES_SIGNATURES
from resources.prompts.signature_agent import SIGNATURE_PROMPT
from resources.prompts.api_agent import API_PROMPT
from resources.prompts.program_agent import PROGRAM_PROMPT
from resources.prompts.vqa_prompts import VQA_PROMPT

from stage1_ver2 import GeneralizedSceneGraphGenerator, DetectedObject
from database_simple import LocalVectorManager

console = Console(highlight=False, force_terminal=False)

# gemini_api_key = 'AIzaSyCJHta9VITkW9ZNnTOuqpvPPIrpSSgCqXg'
gemini_model_name_25 = 'gemini-2.5-flash'
gemini_model_name_20 = 'gemini-2.0-flash'
MISTRAL_URL = "http://0.0.0.0:8001"
MISTRAL_CODE_MODEL_NAME = "codestral-2501"
MISTRAL_VISION_MODEL_NAME = "pixtral-large-latest"
MISTRAL_FIX_MODEL_NAME = "devstral-small-latest"

detected_object_class_string = """
class DetectedObject:
  def __init__(self,
               class_name: str,
               description: str,
               segmentation_mask_2d: np.ndarray,
               rle_mask_2d: str,
               bounding_box_2d: np.ndarray | None,
               point_cloud_3d: o3d.geometry.PointCloud,
               bounding_box_3d_oriented: o3d.geometry.OrientedBoundingBox,
               bounding_box_3d_axis_aligned: o3d.geometry.AxisAlignedBoundingBox,
               image_crop_pil: PILImage.Image | None = None):
    self.class_name = class_name
    self.description = description
    self.segmentation_mask_2d = segmentation_mask_2d
    self.rle_mask_2d = rle_mask_2d
    self.bounding_box_2d = bounding_box_2d
    self.point_cloud_3d = point_cloud_3d
    self.bounding_box_3d_oriented = bounding_box_3d_oriented
    self.bounding_box_3d_axis_aligned = bounding_box_3d_axis_aligned
    self.image_crop_pil = image_crop_pil
"""

detected_object_definition = """
\"\"\"
Represents a detected object in both 2D and 3D spaces with associated metadata.

Args:
    class_name (str): The class label of the detected object (e.g., \"car\", \"person\").
    description (str): A textual description of the object.
    segmentation_mask_2d (np.ndarray): A binary mask array representing the 2D segmentation of the object.
    rle_mask_2d (str): Run-length encoded representation of the 2D mask.
    bounding_box_2d (np.ndarray | None): The 2D bounding box of the object, typically in [x1, y1, x2, y2] format. Optional.
    point_cloud_3d (o3d.geometry.PointCloud): The 3D point cloud corresponding to the object.
    bounding_box_3d_oriented (o3d.geometry.OrientedBoundingBox): Oriented 3D bounding box of the object.
    bounding_box_3d_axis_aligned (o3d.geometry.AxisAlignedBoundingBox): Axis-aligned 3D bounding box of the object.
    image_crop_pil (PILImage.Image | None): Optional cropped image of the object in PIL format.
\"\"\"
class DetectedObject:
"""

def normalize_indentation(code_block: str, indent_with: str = "    ") -> str:
    """
    Normalizes the indentation of a code block.

    This function takes a string containing a block of Python code,
    which may have inconsistent leading whitespace, and reformats it.
    It removes any common leading whitespace from every line, then
    indents the entire block with a specified string (e.g., 4 spaces).
    This correctly preserves nested indentation.

    Args:
        code_block (str): A string containing the Python code.
        indent_with (str): The string to use for indenting each line.
                           Defaults to "    " (4 spaces).

    Returns:
        str: The correctly indented code block.
    """
    if not code_block:
        return ""

    # textwrap.dedent removes the common leading whitespace from every line.
    # This is the key to preserving relative indentation.
    dedented_code = textwrap.dedent(code_block)

    # Now, add our own consistent indent to each line.
    indented_lines = []
    for line in dedented_code.strip().split('\n'):
        indented_lines.append(f"{indent_with}{line}")

    return "\n".join(indented_lines)

# Helper to encode image to base64
def encode_image_to_base64(image_path_or_pil: Union[str, Image.Image]) -> str:
    """Encodes an image (from path or PIL.Image object) to a base64 string."""
    if isinstance(image_path_or_pil, str):
        if not os.path.exists(image_path_or_pil):
            raise FileNotFoundError(f"Image path not found: {image_path_or_pil}")
        with open(image_path_or_pil, "rb") as image_file:
            img_byte_arr = image_file.read()
    elif isinstance(image_path_or_pil, Image.Image):
        buffered = io.BytesIO()
        # Determine format, default to PNG if not obvious or if it has alpha
        fmt = image_path_or_pil.format if image_path_or_pil.format else 'PNG'
        if image_path_or_pil.mode == 'RGBA' and fmt.upper() not in ['PNG', 'WEBP']:
            fmt = 'PNG' # Ensure alpha is preserved
        image_path_or_pil.save(buffered, format=fmt)
        img_byte_arr = buffered.getvalue()
    else:
        raise ValueError("Input must be a file path string or a PIL.Image object.")

    base64_encoded_data = base64.b64encode(img_byte_arr)
    base64_message = base64_encoded_data.decode('utf-8')
    return base64_message

def get_image_mime_type(image_path_or_pil: Union[str, Image.Image]) -> str:
    """Determines a plausible MIME type for the image."""
    if isinstance(image_path_or_pil, str):
        ext = os.path.splitext(image_path_or_pil)[1].lower()
        if ext == ".jpg" or ext == ".jpeg":
            return "image/jpeg"
        elif ext == ".png":
            return "image/png"
        elif ext == ".gif":
            return "image/gif"
        elif ext == ".webp":
            return "image/webp"
        else:
            return "image/png" # Default
    elif isinstance(image_path_or_pil, Image.Image):
        fmt = image_path_or_pil.format
        if fmt:
            return f"image/{fmt.lower()}"
        return "image/png" # Default for PIL if format unknown
    return "image/png" # Overall default


class SpatialRGPTClient:
    """
    Client for interacting with the OpenAI-compatible SpatialRGPT FastAPI server.
    """
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initializes the client.

        Args:
            base_url (str): The base URL of the FastAPI server (e.g., "http://localhost:8000").
        """
        self.base_url = base_url.rstrip('/')
        self.chat_completions_url = f"{self.base_url}/v1/chat/completions"
        self.session = requests.Session() # Use a session for potential connection pooling

    def _prepare_image_content_item(self, image_path_or_pil: Union[str, Image.Image]) -> Dict[str, Any]:
        """Prepares the image_url dictionary for an image."""
        base64_image = encode_image_to_base64(image_path_or_pil)
        mime_type = get_image_mime_type(image_path_or_pil)
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
        }

    def encode_mask_to_base64_png(self, mask_data: Union[Image.Image, np.ndarray]) -> str:
        """Encodes a single-channel mask (PIL Image or NumPy array) to a base64 PNG string."""
        pil_image: Image.Image
        if isinstance(mask_data, np.ndarray):
            if mask_data.ndim == 3 and mask_data.shape[2] == 1:
                mask_data = mask_data.squeeze(axis=2)
            if mask_data.dtype != np.uint8:
                # Normalize and convert if not uint8 (e.g. bool or 0/1 float/int)
                if mask_data.max() <= 1.0: # handles bool, float 0-1, int 0-1
                    pil_image = Image.fromarray((mask_data * 255).astype(np.uint8), mode='L')
                else: # assume it's already 0-255 range but wrong dtype
                    pil_image = Image.fromarray(mask_data.astype(np.uint8), mode='L')
            else: # uint8
                pil_image = Image.fromarray(mask_data, mode='L')
        elif isinstance(mask_data, Image.Image):
            if mask_data.mode != 'L':
                pil_image = mask_data.convert('L')
            else:
                pil_image = mask_data
        else:
            raise ValueError("Unsupported mask type. Must be PIL.Image or NumPy array.")

        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model_name: str = "SpatialRGPT-VILA1.5-8B",
        main_image_path_or_pil: Optional[Union[str, Image.Image]] = None,
        depth_image_path_or_pil: Optional[Union[str, Image.Image]] = None,
        segmentation_masks_data: Optional[List[Union[Image.Image, np.ndarray]]] = None, # NEW
        region_boxes: Optional[List[List[int]]] = None,
        use_sam_segmentation: bool = True,
        process_provided_depth: Optional[bool] = None,
        conv_mode: str = "llama_3",
        temperature: float = 0.2,
        max_tokens: int = 512,
        use_bfloat16: bool = True,
        timeout: int = 120
    ) -> Dict[str, Any]:
        payload_messages = [dict(m) for m in messages]

        image_options: Dict[str, Any] = {} # Initialize empty
        # Explicitly set use_sam_segmentation, as server defaults to True if not present
        image_options["use_sam_segmentation"] = use_sam_segmentation


        if segmentation_masks_data:
            print(f'Size of segmentation masks: {len(segmentation_masks_data)}')
            encoded_masks = []
            for mask_obj in segmentation_masks_data:
                encoded_masks.append(self.encode_mask_to_base64_png(mask_obj)) # Using the new helper
            if encoded_masks:
                image_options["provided_seg_masks"] = encoded_masks
            # If direct masks are provided, `use_sam_segmentation` for boxes is less relevant for *these* masks.
            # However, regions_boxes might still be sent for annotation/indexing.
            # The server will prioritize provided_seg_masks.

        # This needs to be after provided_seg_masks, so it's not overwritten if both are conceptually set
        # For box-based segmentation, if provided_seg_masks is NOT set:
        if "provided_seg_masks" not in image_options:
             image_options["use_sam_segmentation"] = use_sam_segmentation

        if region_boxes:
            image_options["regions_boxes"] = region_boxes

        _process_depth = False
        if depth_image_path_or_pil is not None:
            if process_provided_depth is None: # Auto-set if not explicitly given
                _process_depth = True
            else:
                _process_depth = process_provided_depth

            if _process_depth:
                image_options["process_provided_depth"] = True
                depth_image_content_item = self._prepare_image_content_item(depth_image_path_or_pil)
                image_options["depth_image_url"] = depth_image_content_item["image_url"]
            elif "process_provided_depth" not in image_options : # if explicitly False, make sure it's in options
                 image_options["process_provided_depth"] = False

        elif process_provided_depth is True: # process_provided_depth is True but no depth image
            raise ValueError("`process_provided_depth` is True, but no `depth_image_path_or_pil` was provided.")
        elif process_provided_depth is False: # No depth image, and explicitly told not to process depth
             image_options["process_provided_depth"] = False


        if main_image_path_or_pil:
            if not payload_messages or payload_messages[-1]["role"] != "user":
                payload_messages.append({"role": "user", "content": []})

            main_image_content = self._prepare_image_content_item(main_image_path_or_pil)
            last_user_message = payload_messages[-1]

            if last_user_message["role"] != "user":
                 raise ValueError("Internal error: Last message not 'user' after image preparation.")

            if isinstance(last_user_message["content"], str):
                last_user_message["content"] = [
                    {"type": "text", "text": last_user_message["content"]},
                    main_image_content
                ]
            elif isinstance(last_user_message["content"], list):
                last_user_message["content"].append(main_image_content)
            else: # E.g. if content was None or other type
                raise ValueError(f"Last user message content must be a string or a list. Got: {type(last_user_message['content'])}")

        payload = {
            "model": model_name,
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "conv_mode": conv_mode,
            "use_bfloat16": use_bfloat16,
        }
        if image_options: # Only add if not empty
            payload["image_options"] = image_options

        try:
            response = self.session.post(self.chat_completions_url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            err_detail = f"HTTP Error: {e.response.status_code} for URL {e.request.url if e.request else self.chat_completions_url}"
            try:
                err_detail += f"\nServer response: {json.dumps(e.response.json(), indent=2)}"
            except json.JSONDecodeError:
                err_detail += f"\nServer response (non-JSON): {e.response.text}"
            print(err_detail) # Or log
            raise requests.exceptions.HTTPError(err_detail, response=e.response) from e
        except requests.exceptions.RequestException as e:
            print(f"Request Exception: {e}") # Or log
            raise

    def simple_query(
        self,
        prompt: str,
        image_path: str,
        model_name: str = "SpatialRGPT-VILA1.5-8B",
        **kwargs # Pass other chat parameters
    ) -> Optional[str]:
        """
        A simplified method for a common use case: one image, one text prompt.

        Args:
            prompt (str): The text prompt.
            image_path (str): Path to the image file.
            model_name (str): Model name.
            **kwargs: Additional arguments for the `chat` method (e.g., temperature).

        Returns:
            Optional[str]: The assistant's response text, or None if an error occurs or no choice.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            response_data = self.chat(
                messages=messages,
                model_name=model_name,
                main_image_path_or_pil=image_path,
                **kwargs
            )
            if response_data and response_data.get("choices"):
                return response_data["choices"][0].get("message", {}).get("content")
            return None
        except Exception as e:
            print(f"Error in simple_query: {e}")
            return None

    def query_with_regions_and_depth(
        self,
        prompt: str, # e.g., "Describe <region0> relative to <region1>"
        main_image_path: str,
        depth_image_path: str,
        region_boxes: List[List[int]], # [[x1,y1,x2,y2], [x1,y1,x2,y2]]
        model_name: str = "SpatialRGPT-VILA1.5-8B",
        **kwargs
    ) -> Optional[str]:
        """
        Simplified method for querying with an image, depth map, and region boxes.

        Args:
            prompt (str): The text prompt, potentially using <regionX> tags.
            main_image_path (str): Path to the main image file.
            depth_image_path (str): Path to the depth image file.
            region_boxes (List[List[int]]): Bounding boxes for regions.
            model_name (str): Model name.
            **kwargs: Additional arguments for the `chat` method.

        Returns:
            Optional[str]: The assistant's response text, or None.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            response_data = self.chat(
                messages=messages,
                model_name=model_name,
                main_image_path_or_pil=main_image_path,
                depth_image_path_or_pil=depth_image_path,
                region_boxes=region_boxes,
                process_provided_depth=True, # Explicitly use the provided depth
                **kwargs
            )
            if response_data and response_data.get("choices"):
                return response_data["choices"][0].get("message", {}).get("content")
            return None
        except Exception as e:
            print(f"Error in query_with_regions_and_depth: {e}")
            return None

class GeneratorLocal:
    def __init__(self, model_name="SalonbusAI/GLM-4-32B-0414-FP8", temperature=0.7, device_preference=None, max_new_tokens=1024):
        self.temperature = temperature
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        if device_preference:
            self.device = torch.device(device_preference)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        print(f"Qwen3 Generator: Selected device: {self.device}")
        print(f"Qwen3 Generator: Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Set pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("Qwen3 Generator: Added new pad token '[PAD]'.")

        print(f"Qwen3 Generator: Loading model {self.model_name} onto device {self.device}...")

        # Use bfloat16 if supported (better performance on A100)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"🧠 Using precision: {dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            # load_in_8bit=True,               # 🔥 Enable 8-bit loading
            # attn_implementation="flash_attention_2"  # Flash Attention 2 for faster attention
        )

        self.model.eval()

    def remove_substring(self, output, substring):
        return output.replace(substring, "") if substring in output else output

    def remove_think_tags(self, text):
        """Remove <think> and </think> tags and their content using regex"""
        # This will remove everything between <think> and </think> including the tags
        pattern = r'<think>.*?</think>'
        return re.sub(pattern, '', text, flags=re.DOTALL).strip()

    def generate(self, prompt=None, messages=None, enable_thinking=False):
        current_conversation = []
        if messages:
            current_conversation = list(messages)
        elif prompt:
            current_conversation.append({"role": "user", "content": prompt})
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")
        try:
            input_ids = self.tokenizer.apply_chat_template(
                current_conversation,
                add_generation_prompt=True,
                tokenize=False,
                return_tensors="pt",
                enable_thinking=enable_thinking
            )
            inputs = self.tokenizer(input_ids, return_tensors="pt").to(self.model.device)
        except Exception as e:
            print(f"Qwen3 Generator: Error applying chat template: {e}")
            raise
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            gen_kwargs.update({
                "temperature": self.temperature,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 20
            })
        try:
            outputs = self.model.generate(**inputs, **gen_kwargs)
            response_ids = outputs[0][len(inputs.input_ids[0]):]
            result = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        except Exception as e:
            print(f"Qwen3 Generator: Error during generation: {e}")
            raise

        # Remove code blocks and think tags
        result = self.remove_substring(result, "```python")
        result = self.remove_substring(result, "```")
        result = self.remove_think_tags(result)  # Add this line to remove think tags
        print(result)

        new_conversation_history = list(current_conversation)
        new_conversation_history.append({"role": "assistant", "content": result})
        return result, new_conversation_history


# class Generator:
#     def __init__(self, model_name="SalonbusAI/GLM-4-32B-0414-FP8", base_url=None, api_key="dummy", max_new_tokens=1024):
#         """
#         Initialize the Generator with OpenAI client

#         Args:
#             model_name: Model name to use
#             temperature: Temperature for generation
#             base_url: Custom base URL for OpenAI-compatible API
#             api_key: API key (can be dummy for local servers)
#             max_new_tokens: Maximum tokens to generate
#         """
#         # self.temperature = temperature
#         self.model_name = model_name
#         self.max_new_tokens = max_new_tokens

#         # Initialize OpenAI client with custom URL
#         if base_url:
#             self.client = OpenAI(
#                 base_url=base_url,
#                 api_key=api_key
#             )
#             print(f"OpenAI Generator: Initialized with custom base URL: {base_url}")
#         else:
#             self.client = OpenAI(api_key=api_key)
#             print("OpenAI Generator: Initialized with default OpenAI API")

#         print(f"OpenAI Generator: Using model: {self.model_name}")

#     def remove_substring(self, output, substring):
#         """Remove a substring from the output"""
#         return output.replace(substring, "") if substring in output else output

#     def remove_think_tags(self, text):
#         """Remove <think> and </think> tags and their content using regex"""
#         pattern = r'<think>.*?</think>'
#         return re.sub(pattern, '', text, flags=re.DOTALL).strip()

#     def generate(self, prompt=None, messages=None, enable_thinking=False, model_name=None, temperature=0.7):
#         """
#         Generate text using OpenAI API

#         Args:
#             prompt: Single prompt string
#             messages: List of message dictionaries
#             enable_thinking: Whether to enable thinking (handled in prompt injection)

#         Returns:
#             tuple: (generated_text, conversation_history)
#         """
#         current_conversation = []

#         # Prepare initial message list
#         if messages:
#             current_conversation = list(messages)
#         elif prompt:
#             current_conversation.append({"role": "user", "content": prompt})
#         else:
#             raise ValueError("Either 'prompt' or 'messages' must be provided.")

#         # Inject `/no_think` if required
#         # if not enable_thinking:
#         #     # Inject into first message if it's user or system
#         #     first_msg = current_conversation[0]
#         #     if first_msg["role"] in ("user", "system") and "/no_think" not in first_msg["content"]:
#         #         first_msg["content"] = "/no_think\n" + first_msg["content"]

#         try:
#             response = self.client.chat.completions.create(
#                 model=model_name if model_name is not None else self.model_name,
#                 messages=current_conversation,
#                 temperature=temperature,
#                 max_tokens=self.max_new_tokens,
#                 top_p=0.9 if temperature > 0 else None,
#             )

#             result = response.choices[0].message.content

#         except Exception as e:
#             print(f"OpenAI Generator: Error during generation: {e}")
#             raise

#         # Post-process
#         print(result)
#         if "```python" in result:
#             result = self.remove_substring(result, "```python")
#             result = self.remove_substring(result, "```")
#         result = self.remove_think_tags(result)

#         new_conversation_history = list(current_conversation)
#         new_conversation_history.append({"role": "assistant", "content": result})

#         return result, new_conversation_history

import base64
import mimetypes
import re
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Optional, Tuple, Union

# It's good practice to define the expected message format
Message = Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]

# It's good practice to define the expected message format
Message = Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]

class Generator:
    """
    An optimized client specifically designed for your Mistral API proxy server.
    Supports both text and multimodal (text and image) generation with intelligent
    model selection and comprehensive error handling.
    """
    def __init__(self,
                 base_url: str = "http://localhost:8001",
                 api_key: str = "dummy",
                 model_name: str = "mistral-large-latest",
                 temperature: float = 0.7,
                 max_new_tokens: int = 8192,
                 timeout: int = 60):
        """
        Initialize the Generator for your Mistral API proxy server.

        Args:
            base_url: URL of your Mistral API proxy server.
            api_key: API key (can be dummy for your server setup).
            model_name: Default model name to use.
            temperature: Default temperature for generation.
            max_new_tokens: Default maximum number of new tokens to generate.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout

        # Initialize OpenAI client with your server's endpoint
        self.client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=api_key,
            timeout=timeout
        )
        
        print(f"✅ Generator Initialized: Connected to Mistral API proxy at {self.base_url}")
        
        # Test connection and get server info
        self._test_connection()

    def _test_connection(self) -> None:
        """Test connection to the server and display capabilities."""
        try:
            # Test health endpoint
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"🏥 Server Health: {health_data.get('status', 'unknown')}")
                print(f"🔑 Keys Available: {health_data.get('keys_in_pool', 'unknown')}")
                
                if health_data.get('vision_support'):
                    print("📷 Vision Support: Enabled")
                    vision_models = health_data.get('supported_vision_models', [])
                    if vision_models:
                        print(f"🎯 Vision Models: {', '.join(vision_models)}")
                else:
                    print("📷 Vision Support: Disabled")
                    
        except Exception as e:
            print(f"⚠️ Could not fetch server health info: {e}")
            print("🔄 Proceeding with basic configuration...")

    def get_available_models(self) -> List[str]:
        """Get list of available models from the server."""
        try:
            response = self.client.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            print(f"⚠️ Could not fetch models: {e}")
            return []

    def _encode_image(self, image_path: str) -> str:
        """
        Loads an image file, encodes it to Base64, and formats it as a data URI.
        Optimized for your server's image processing pipeline.
        """
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found at {image_path}")

            # Your server supports common image formats
            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type or not mime_type.startswith('image'):
                # Default to common formats if mime type detection fails
                ext = path.suffix.lower()
                if ext in ['.jpg', '.jpeg']:
                    mime_type = 'image/jpeg'
                elif ext == '.png':
                    mime_type = 'image/png'
                elif ext == '.gif':
                    mime_type = 'image/gif'
                elif ext == '.webp':
                    mime_type = 'image/webp'
                else:
                    raise ValueError(f"Unsupported image format: {ext}")

            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            raise IOError(f"Error processing image {image_path}: {e}") from e

    def _prepare_api_messages(self, 
                            prompt: Optional[str], 
                            messages: Optional[List[Message]], 
                            images: Optional[List[str]]) -> Tuple[List[Message], List[Message], bool]:
        """
        Prepares messages for your Mistral API server, handling text and images.
        Returns (api_messages, full_history, has_images)
        """
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Create a mutable copy to avoid side effects
        full_history = list(messages) if messages else []

        if not full_history:
            if prompt is None:
                raise ValueError("A 'prompt' must be provided if 'messages' is empty.")
            full_history.append({"role": "user", "content": prompt})
        
        # The last message is the one we will potentially add images to
        last_message = full_history[-1]
        
        # Ensure the last message is from the user to attach images
        if last_message["role"] != "user":
             raise ValueError("Images can only be added to the most recent 'user' message.")
        
        # Extract text content from the last message
        text_content = last_message.get("content", "")
        if not isinstance(text_content, str):
            raise TypeError("The last message content must be a string when providing new images.")
            
        has_images = False
        
        # If no images, keep it simple (your server handles text-only efficiently)
        if not images:
            api_messages = full_history
        else:
            # Build multimodal content for the final user message
            final_user_content = [{"type": "text", "text": text_content}]
            
            for img_path in images:
                try:
                    base64_image_uri = self._encode_image(img_path)
                    final_user_content.append({
                        "type": "image_url",
                        "image_url": {"url": base64_image_uri}
                    })
                    has_images = True
                except (IOError, FileNotFoundError, ValueError) as e:
                    # Add error message to text content
                    error_msg = f"[Error: Could not load image at path '{img_path}'. Reason: {e}]"
                    final_user_content[0]["text"] = f"{error_msg}\n{final_user_content[0]['text']}"
                    print(f"⚠️  {error_msg}")

            # Build the API messages
            api_messages = full_history[:-1]
            api_messages.append({"role": "user", "content": final_user_content})
        
        return api_messages, full_history, has_images

    def _handle_server_error(self, error: Exception) -> str:
        """Handle errors specific to your Mistral API server."""
        error_str = str(error)
        
        # Handle your server's specific error types
        if "No API keys available" in error_str:
            return "Server is at capacity - no API keys available. Please try again later."
        elif "Request timeout" in error_str:
            return "Request timed out on the server. Consider reducing complexity or trying again."
        elif "image_processing_error" in error_str:
            return "Server encountered an error processing images. Check image format and size."
        elif "model_error" in error_str:
            return "Requested model is not available or encountered an error."
        elif "auth_error" in error_str:
            return "Authentication error with the upstream API."
        elif "vision" in error_str.lower() and "not supported" in error_str.lower():
            return "Vision features are not available. Try with text-only input."
        else:
            return f"Server error: {error_str}"

    def generate(self,
                 prompt: Optional[str] = None,
                 messages: Optional[List[Message]] = None,
                 images: Optional[List[str]] = None,
                 model_name: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_new_tokens: Optional[int] = None,
                 **kwargs) -> Tuple[str, List[Message]]:
        """
        Generates a response using your Mistral API proxy server.
        
        The server automatically:
        - Selects appropriate models based on content (vision models for images)
        - Handles OpenAI -> Mistral model mapping
        - Provides real-time metrics and monitoring
        - Manages API key rotation and load balancing

        Args:
            prompt: A single string prompt. Used if 'messages' is not provided.
            messages: A list of message dictionaries (OpenAI format).
            images: A list of local file paths to images to include.
            model_name: The model to use (server will auto-select vision models if needed).
            temperature: The temperature for this request.
            max_new_tokens: The max tokens for this request.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            A tuple containing:
            - The generated text response (str).
            - The complete conversation history including the new response (list).
        """
        api_messages, conversation_history, has_images = self._prepare_api_messages(
            prompt, messages, images
        )

        # Use provided model or default, let server handle vision model selection
        selected_model = model_name if model_name is not None else self.model_name
        
        if has_images:
            print(f"📷 Sending multimodal request with {len([m for m in api_messages[-1]['content'] if m.get('type') == 'image_url'])} image(s)")
            # Optionally suggest a vision model if none specified
            if selected_model not in ["pixtral-12b-2409", "pixtral-large-latest", "gpt-4o", "gpt-4-vision-preview"]:
                print(f"💡 Note: Server will auto-upgrade '{selected_model}' to a vision model for image processing")

        # Prepare API call parameters
        api_params = {
            "model": selected_model,
            "messages": api_messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in api_params and value is not None:
                api_params[key] = value

        try:
            response = self.client.chat.completions.create(**api_params)
            result = response.choices[0].message.content or ""

            # Log useful information
            if hasattr(response, 'model') and response.model != selected_model:
                print(f"🔄 Server selected model: {response.model} (requested: {selected_model})")
            
            if hasattr(response, 'usage'):
                usage = response.usage
                print(f"📊 Token usage - Prompt: {usage.prompt_tokens}, "
                      f"Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")

        except Exception as e:
            error_message = self._handle_server_error(e)
            print(f"❌ Generation failed: {error_message}")
            raise RuntimeError(error_message) from e

        # Append the assistant's response to the conversation history
        conversation_history.append({"role": "assistant", "content": result})

        return result, conversation_history

    def generate_stream(self, **kwargs):
        """
        Note: Your current server implementation doesn't support streaming.
        This method is provided for future compatibility.
        """
        raise NotImplementedError(
            "Streaming is not supported by your current server implementation. "
            "Use generate() for standard completion."
        )

    def get_server_metrics(self) -> Optional[Dict]:
        """Get real-time metrics from your server's dashboard."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                return response.json().get('metrics', {})
        except Exception as e:
            print(f"⚠️ Could not fetch server metrics: {e}")
        return None

    def print_server_status(self) -> None:
        """Print a comprehensive status report of your server."""
        metrics = self.get_server_metrics()
        if not metrics:
            print("❌ Could not retrieve server status")
            return
            
        print("\n" + "="*50)
        print("🚀 MISTRAL API PROXY SERVER STATUS")
        print("="*50)
        print(f"📊 Total Requests: {metrics.get('total_requests', 0):,}")
        print(f"✅ Success Rate: {metrics.get('success_rate', 0):.1f}%")
        print(f"⚡ Active Requests: {metrics.get('active_count', 0)}")
        print(f"🔑 Keys Available: {metrics.get('keys_available', 0)}")
        print(f"⏱️ Avg Response Time: {metrics.get('avg_response_time', 0)*1000:.0f}ms")
        print(f"📈 Requests/min: {metrics.get('requests_per_minute', 0):.1f}")
        print(f"📷 Image Requests: {metrics.get('multimodal_requests', 0):,}")
        print(f"🎯 Vision Model Usage: {metrics.get('vision_requests', 0):,}")
        print(f"⏰ Uptime: {metrics.get('uptime_formatted', 'Unknown')}")
        print("="*50)

class GeneratorVL:
    def __init__(self, model_name="OpenGVLab/InternVL3-9B", revision="main", temperature=0.7,
                 device_preference=None, max_new_tokens=1024, load_in_8bit=False):
        self.temperature = temperature
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.load_in_8bit = load_in_8bit

        # Device selection logic
        if device_preference:
            self.device = torch.device(device_preference)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        print(f"InternVL3 Generator: Selected device: {self.device}")

        # Image preprocessing constants
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.transform = self.build_transform(448)

        # Load tokenizer and model
        print(f"InternVL3 Generator: Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False, revision=revision)

        print(f"InternVL3 Generator: Loading model {self.model_name} onto device {self.device}...")

        # Determine precision
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"🧠 Using precision: {self.dtype}")

        # Model loading logic
        if torch.cuda.device_count() > 1 and not self.load_in_8bit:
            self.device_map = self.split_model()
        else:
            self.device_map = "auto"

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=self.device_map,
            load_in_8bit=self.load_in_8bit,
            revision=revision
        ).eval()

        # Initialize conversation history
        self.history = []

    # Image preprocessing methods
    def build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set((i, j) for n in range(min_num, max_num + 1)
                          for i in range(1, n + 1) for j in range(1, n + 1)
                          if i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find closest aspect ratio
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = orig_width * orig_height

        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff or (ratio_diff == best_ratio_diff and
                                               area > 0.5 * image_size * image_size * ratio[0] * ratio[1]):
                best_ratio_diff = ratio_diff
                best_ratio = ratio

        target_width = image_size * best_ratio[0]
        target_height = image_size * best_ratio[1]
        blocks = best_ratio[0] * best_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []

        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            processed_images.append(resized_img.crop(box))

        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))

        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [self.transform(image) for image in images]
        return torch.stack(pixel_values)

    def split_model(self):
        device_map = {}
        world_size = torch.cuda.device_count()
        config = self.model.config  # Assuming config is available

        try:
            num_layers = config.llm_config.num_hidden_layers
        except:
            num_layers = config.language_config.num_hidden_layers

        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1

        # Vision components
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.model.rotary_emb'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

        return device_map

    # Video processing
    def get_video_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        return np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])

    # Text cleaning
    def remove_substring(self, output, substring):
        return output.replace(substring, "") if substring in output else output

    def remove_think_tags(self, text):
        """Remove </think> tags and their content using regex"""
        pattern = r'</think>'
        return re.sub(pattern, '', text, flags=re.DOTALL).strip()

    # Main generation method
    def generate(self, prompt=None, messages=None, images=None,
                enable_thinking=False, max_num=12):
        """
        Generate response based on input

        Args:
            prompt: Text prompt
            messages: List of conversation history
            images: Path to image file or list of image paths
            video: Path to video file
            enable_thinking: Enable thinking mode
            max_num: Max number of image tiles
        """
        # Handle conversation history
        current_history = []
        if messages:
            current_history = messages
        elif prompt:
            current_history = [{"role": "user", "content": prompt}]
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Process images
        pixel_values = None
        num_patches_list = None

        if images:
            if isinstance(images, str):
                images = [images]

            all_pixel_values = []
            for img_path in images:
                pixel_values = self.load_image(img_path, max_num=max_num)
                all_pixel_values.append(pixel_values)

            pixel_values = torch.cat(all_pixel_values)
            num_patches_list = [p.shape[0] for p in all_pixel_values]
            pixel_values = pixel_values.to(self.dtype).to(self.model.device)

        # Generation config
        gen_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True if self.temperature > 0 else False
        }

        if self.temperature > 0:
            gen_config.update({
                "temperature": self.temperature,
                "top_p": 0.9,
                "top_k": 20
            })

        # Generate response
        try:
            # Pass conversation history explicitly for stateless operation
            history_to_pass = current_history[:-1] if len(current_history) > 1 else None

            if pixel_values is not None:
                response, new_history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    current_history[-1]["content"],
                    generation_config=gen_config,
                    num_patches_list=num_patches_list,
                    history=history_to_pass,
                    return_history=True
                )
            else:
                response, new_history = self.model.chat(
                    self.tokenizer,
                    None,
                    current_history[-1]["content"],
                    generation_config=gen_config,
                    history=history_to_pass,
                    return_history=True
                )

            # Clean response
            response = self.remove_substring(response, "```python")
            response = self.remove_substring(response, "```")
            response = self.remove_think_tags(response)

            return response, new_history

        except Exception as e:
            print(f"InternVL3 Generator: Error during generation: {e}")
            raise
    
# ==============================================================================
# 1. Custom Exception for Clear Error Handling
# ==============================================================================

class ApiKeyQuotaExceededError(Exception):
    """Custom exception raised when a Gemini API key hits its rate limit or quota."""
    pass


# ==============================================================================
# 2. The Thread-Safe Key Manager
# ==============================================================================

class ApiKeyManager:
    """
    A thread-safe manager for handling a pool of Google Gemini API keys.

    This class loads API keys, provides them on demand, and automatically
    handles temporary key retirement, revival, and permanent removal on repeated failure.
    It also reloads new keys from Google Sheets every 30 minutes.
    """
    def __init__(
        self,
        sheet_url: str = "https://docs.google.com/spreadsheets/d/1gqlLToS3OXPA-CvfgXRnZ1A6n32eXMTkXz4ghqZxe2I/gviz/tq?tqx=out:csv&gid=0",
        revival_delay_seconds: int = 60,
        max_failures: int = 3,
        reload_interval: int = 1800,  # 30 minutes
    ):
        self.sheet_url = sheet_url
        self.revival_delay_seconds = revival_delay_seconds
        self.max_failures = max_failures
        self.reload_interval = reload_interval

        self._key_queue = queue.Queue()
        self._retired_keys_lock = threading.Lock()
        self._retired_keys = []
        self._failure_counts = {}
        self._keys_info = []

        self._stop_event = threading.Event()

        self._load_and_enqueue_keys(initial=True)

        self._revival_thread = threading.Thread(target=self._revive_keys_periodically, daemon=True)
        self._revival_thread.start()

        self._reload_thread = threading.Thread(target=self._reload_keys_periodically, daemon=True)
        self._reload_thread.start()

        console.print(f"[ApiKeyManager] Initialized with [bold green]{self._key_queue.qsize()}[/bold green] active keys.")

    def _load_api_keys_from_sheet(self) -> list:
        console.print("[ApiKeyManager] Fetching API keys from Google Sheets...")
        try:
            response = requests.get(self.sheet_url)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            df.columns = df.columns.str.strip()
            if 'Token' not in df.columns or 'Name' not in df.columns:
                raise ValueError("CSV must contain 'Token' and 'Name' columns.")
            df = df.dropna(subset=['Token', 'Name']).drop_duplicates()
            return df[['Token', 'Name']].rename(columns={'Token': 'token', 'Name': 'name'}).to_dict('records')
        except Exception as e:
            raise RuntimeError(f"Failed to load API keys: {e}")

    def _load_and_enqueue_keys(self, initial=False):
        new_keys = self._load_api_keys_from_sheet()
        seen_tokens = {info["token"] for info in self._keys_info}
        added_count = 0

        for key_info in new_keys:
            token = key_info["token"]
            if token not in seen_tokens:
                self._key_queue.put(key_info)
                self._keys_info.append(key_info)
                seen_tokens.add(token)
                added_count += 1

        if not self._keys_info and initial:
            raise ValueError("No API keys loaded. Please check the sheet URL and content.")

        if added_count > 0:
            console.print(f"[ApiKeyManager] Added [green]{added_count}[/green] new API keys.")

    def _revive_keys_periodically(self):
        while not self._stop_event.wait(5):
            with self._retired_keys_lock:
                now = time.time()
                keys_to_revive = []
                remaining_retired = []

                for retirement_time, key_info in self._retired_keys:
                    key_name = key_info["name"]
                    if now - retirement_time >= self.revival_delay_seconds:
                        # if self._failure_counts.get(key_name, 0) < self.max_failures:
                        keys_to_revive.append(key_info)
                    else:
                        remaining_retired.append((retirement_time, key_info))

                self._retired_keys = remaining_retired

            for key_info in keys_to_revive:
                self._key_queue.put(key_info)
                console.print(f"[ApiKeyManager] ✅ Revived key [cyan]{key_info['name']}[/cyan]. Active keys: {self._key_queue.qsize()}")

    def _reload_keys_periodically(self):
        while not self._stop_event.wait(self.reload_interval):
            try:
                self._load_and_enqueue_keys()
            except Exception as e:
                console.print(f"[ApiKeyManager] [red]Reload error:[/red] {e}")

    @contextlib.contextmanager
    def get_key(self):
        key_info = self._key_queue.get(block=True)
        key_name = key_info["name"]

        try:
            yield key_info

        except ApiKeyQuotaExceededError:
            self._failure_counts[key_name] = self._failure_counts.get(key_name, 0) + 1
            fail_count = self._failure_counts[key_name]

            if fail_count >= self.max_failures:
                console.print(f"[ApiKeyManager] 🔥 Permanently removed key [red]{key_name}[/red] after {fail_count} failures.")
                # Key is permanently removed — not added back
            else:
                with self._retired_keys_lock:
                    self._retired_keys.append((time.time(), key_info))
                active = self._key_queue.qsize()
                cooldown = len(self._retired_keys)
                console.print(f"[ApiKeyManager] Retiring key [red]{key_name}[/red] for {self.revival_delay_seconds}s (failures: {fail_count}). Active: {active}, Cooldown: {cooldown})")

            raise

        except Exception:
            # Put it back on unexpected error
            self._key_queue.put(key_info)
            raise

        else:
            # Successful usage
            self._key_queue.put(key_info)

        finally:
            self._key_queue.task_done()

    def has_available_keys(self) -> bool:
        with self._retired_keys_lock:
            return not self._key_queue.empty() or len(self._retired_keys) > 0

    def get_removed_keys(self) -> list:
        """Returns list of permanently removed key names."""
        return [k for k, count in self._failure_counts.items() if count >= self.max_failures]

    def shutdown(self):
        console.print("[ApiKeyManager] Shutting down...")
        self._stop_event.set()
        self._revival_thread.join()
        self._reload_thread.join()
        console.print("[ApiKeyManager] Shutdown complete.")


# ==============================================================================
# 3. The Refactored, Stateless Gemini Generator
# ==============================================================================

# Replace your old GeneratorGemini class with this one.

class GeneratorGemini:
    """
    A self-sufficient, OpenAI-compatible client for the Google AI Studio (Gemini) API.
    It now manages its own API key acquisition and retries.
    """
    def __init__(self, api_key_manager: ApiKeyManager, model_name="gemini-1.5-pro-latest", temperature=0.2, max_new_tokens=8192):
        self.api_key_manager = api_key_manager
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        console.print("[GeneratorGemini] Initialized with direct access to ApiKeyManager for robust retries.", style="green")

    def _load_image(self, image_path: str) -> Image.Image:
        """Helper to load an image file."""
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Error loading image {image_path}: {e}") from e

    def _prepare_api_request(self, prompt=None, messages=None, images=None):
        """Helper to format the request for the Gemini API."""
        if messages:
            history, last_message = messages[:-1], messages[-1]
            prompt_text = last_message['content']
        elif prompt:
            history, prompt_text = [], prompt
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        formatted_history = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [m["content"]]} for m in history]
        prompt_content = [prompt_text] if prompt_text else []

        if images:
            image_list = [images] if isinstance(images, str) else images
            for img_path in image_list:
                try:
                    prompt_content.append(self._load_image(img_path))
                except IOError as e:
                    prompt_content.insert(0, f"[Error: Could not load image at path: {img_path}]")

        # IMPORTANT: Ensure full_history is always a list for appending
        full_history = list(messages) if messages else []
        return prompt_content, formatted_history, full_history

    def generate(self, prompt: str = None, messages: list = None, images: list = None,
                 model_name: str = None,  # <-- CHANGE 1: Add optional model_name parameter
                 max_new_tokens: int = None, temperature: float = None) -> tuple[str, list]:
        """
        Generates a response from the Gemini API, transparently handling key exhaustion and retries.
        Can optionally override the instance's default model_name for a single call.
        """
        while self.api_key_manager.has_available_keys():
            try:
                with self.api_key_manager.get_key() as key_info:
                    api_key = key_info['token']
                    console.print(f"[GeneratorGemini] Attempting request with key '{key_info['name']}'.", style="yellow")

                    try:
                        genai.configure(api_key=api_key) # Assuming genai is configured

                        generation_config = { # Using a dict for clarity
                            "max_output_tokens": max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
                            "temperature": temperature if temperature is not None else self.temperature
                        }

                        # <-- CHANGE 2: Decide which model to use for this specific call
                        model_to_use = model_name or self.model_name
                        console.print(f"[GeneratorGemini] Using model: {model_to_use}", style="cyan")

                        # <-- CHANGE 3: Use the selected model name
                        model = genai.GenerativeModel(model_to_use, generation_config=generation_config)
                        
                        # --- The rest of your logic remains the same ---
                        prompt_content, api_history, full_history = self._prepare_api_request(prompt, messages, images)

                        chat = model.start_chat(history=api_history)
                        response = chat.send_message(prompt_content)
                        response_text = response.text
                        
                        # Mocking the response for demonstration
                        # response_text = f"This is a mocked response from the {model_to_use} model."
                        # End of mock

                        full_history.append({"role": "assistant", "content": response_text})
                        console.print(f"[GeneratorGemini] Request successful with key '{key_info['name']}'.", style="green")
                        return response_text, full_history

                    except InternalServerError as internal_error:
                        console.print(f"[GeneratorGemini] InternalServerError on attempt. Retrying in 1s...", style="yellow")
                        time.sleep(1)
                        continue

                    except Exception as e:
                        error_str = str(e).lower()
                        if any(keyword in error_str for keyword in ["quota", "rate limit", "exceeded", "expired"]):
                            raise ApiKeyQuotaExceededError(f"API key '{key_info['name']}' has been exhausted.") from e
                        else:
                            console.print(f"[GeneratorGemini] An unexpected API error occurred: {e}", style="bold red")
                            raise

            except ApiKeyQuotaExceededError:
                console.print(f"[GeneratorGemini] Key exhausted. Trying next available key...", style="bold yellow")
                continue

        raise RuntimeError("All Gemini API keys have been exhausted. Cannot complete the request.")

# Modify your VADARContext class to include a placeholder for the current key
class VADARContext:
    def __init__(self, qwen_generator, gemini_generator, unik3d_model, embedding_model, spacy_nlp, device, api_key_manager):
        self.qwen_generator = qwen_generator
        self.gemini_generator = gemini_generator
        self.unik3d_model = unik3d_model
        self.embedding_model = embedding_model
        self.spacy_nlp = spacy_nlp
        self.device = device
        self.milvus_manager = LocalVectorManager()
        self.milvus_manager.load_database('my_vector_database.pkl')
        # The ApiKeyManager is now a direct part of the context
        self.api_key_manager = api_key_manager
        # The concept of a single 'current_gemini_api_key' is no longer needed here.
        # It's now managed inside the generator.

# ==============================================================================
# 4. Example Usage in Your Batch Processor
# ==============================================================================

# In your BatchProcessor or main script, you would initialize the manager once.
# The `GeneratorGemini` instance can also be initialized once.

# --- In your BatchProcessor's __init__ or _initialize_shared_resources ---
# self.api_key_manager = ApiKeyManager()
# self.gemini_generator = GeneratorGemini() # This is now safe to share

# --- In your _consumer_task method ---
# def _consumer_task_example(self): # self refers to the BatchProcessor instance
#     # ... inside the main loop of the consumer ...
#     try:
#         data_for_stage2 = self.stage1_output_queue.get(timeout=1)
#         if data_for_stage2 is None:
#             break

#         # Borrow a key from the manager for this specific job
#         with self.api_key_manager.get_key() as key_info:
#             console.print(f"[Consumer-{threading.get_ident()}] Using key '{key_info['name']}' for index {data_for_stage2['index']}")
            
#             # Now, you can safely call the generate method, passing the borrowed key.
#             # For example, inside your process_query_stage2 function:
#             # final_result, _ = self.gemini_generator.generate(
#             #     api_key=key_info['token'],
#             #     prompt="What do you see?",
#             #     images=[data_for_stage2['image_path']]
#             # )
#             # ... process the result ...
            
#     except ApiKeyQuotaExceededError:
#         # The 'with' block already handled retiring the key.
#         # We can log this and maybe retry the job with a new key later.
#         console.print(f"[Consumer-{threading.get_ident()}] A key was exhausted. The job for index {data_for_stage2['index']} failed but can be retried.")
    
#     except queue.Empty:
#         continue
    
#     except Exception as e:
#         console.print(f"[Consumer-{threading.get_ident()}] A critical error occurred: {e}", style="bold red")

def _instantiate_unik3d_model(model_size_str="Large", device_str="cuda"):
    type_ = model_size_str[0].lower()
    name = f"unik3d-vit{type_}"
    try:
        model = UniK3D.from_pretrained(f"lpiccinelli/{name}")
        model.resolution_level = 9
        model.interpolation_mode = "bilinear"
        model = model.to(device_str).eval()
        print(f"UniK3D model '{name}' initialized on {device_str}")
        return model
    except Exception as e:
        print(f"Error instantiating UniK3D model {name}: {e}")
        raise

# --- Refactoring Step 2: `initialize_modules` now returns a context object ---
def initialize_modules(
    qwen_model_name="codestral-2501",
    qwen_max_new_tokens=8192,
    qwen_device_preference=None,
    qwen_api_base_url="http://localhost:8000",
    api_key_manager=None,
    other_models_device_preference="cuda:0",
    unik3d_model_size="Large"
) -> VADARContext:
    """
    Initializes all necessary models and returns them in a context object.
    This function is the single source of model and configuration instantiation.
    """
    # NO global variables are set here.

    # Device setup for GroundingDINO and UniK3D
    if "cuda" in str(other_models_device_preference) and torch.cuda.is_available():
        try:
            if ":" in other_models_device_preference:
                gpu_id = int(other_models_device_preference.split(":")[1])
                if gpu_id < torch.cuda.device_count():
                    device = torch.device(f"cuda:{gpu_id}")
                else:
                    print(f"Warning: GPU ID {gpu_id} not available. Defaulting to cuda:0.")
                    device = torch.device("cuda:0")
            else:
                device = torch.device("cuda:0")
        except ValueError:
            print(f"Invalid CUDA device string '{other_models_device_preference}'. Defaulting to cuda:0.")
            device = torch.device("cuda:0")
    elif "mps" in str(other_models_device_preference) and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Main device for non-LLM models: {device}")

    if api_key_manager is None:
        api_key_manager = ApiKeyManager()

    # Initialize Qwen3 Generator
    print(f"Initializing Qwen3 Generator with model: {qwen_model_name}")
    try:
        qwen_generator = Generator(
            model_name=qwen_model_name,
            max_new_tokens=qwen_max_new_tokens,
            base_url=qwen_api_base_url
        )
        print("Qwen3 Generator initialized successfully.")
    except Exception as e:
        print(f"Error initializing Qwen3 Generator: {e}")
        raise

    # Initialize Gemini API Client Wrapper
    print(f"Initializing Gemini Client for API with model '{gemini_model_name_20}'")
    try:
        # --- MODIFIED: Pass the manager to the constructor ---
        gemini_generator = GeneratorGemini(
            api_key_manager=api_key_manager,
            model_name=gemini_model_name_20
        )
        print("Gemini API Client Wrapper initialized successfully.")
    except Exception as e:
        print(f"Error initializing Gemini API Client Wrapper: {e}")
        traceback.print_exc()
        raise

    # Initialize UniK3D
    print(f"Initializing UniK3D (Size: {unik3d_model_size})")
    try:
        unik3d_model = _instantiate_unik3d_model(model_size_str=unik3d_model_size, device_str=str(device))
    except Exception as e:
        print(f"Error initializing UniK3D: {e}")
        raise

    # Initialize Sentence Transformer for semantic similarity
    try:
        embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', trust_remote_code=True)
        print("SentenceTransformer embedding model initialized successfully.")
    except Exception as e:
        print(f"Error initializing SentenceTransformer model: {e}")
        raise

    # Initialize SpaCy NLP model
    try:
        spacy_nlp = spacy.load("en_core_web_sm")
        print("SpaCy NLP model initialized successfully.")
    except Exception as e:
        print(f"Error initializing SpaCy model: {e}")
        raise

    # Create and return the context object
    return VADARContext(
        qwen_generator=qwen_generator,
        gemini_generator=gemini_generator,
        unik3d_model=unik3d_model,
        embedding_model=embedding_model,
        spacy_nlp=spacy_nlp,
        device=device,
        api_key_manager=api_key_manager # Pass it here
    )

# -- Utility Functions --

def wrap_generated_program(generated_program):
    return f"""
def solution_program(image, detected_objects):
{generated_program}
    return final_result
final_result = solution_program(image, detected_objects)
"""

def correct_indentation(code_str):
    lines = code_str.split("\n")
    indented_lines = ["    " + line.lstrip() for line in lines]
    return "\n".join(indented_lines)

# def generate(context: VADARContext, prompt: str = None, messages: list = None, enable_thinking=False, model_name=None, temperature=0.2):
#     if not context.qwen_generator:
#         error_msg = "Error: Qwen3 Generator not available in the provided context."
#         print(error_msg)
#         return error_msg, messages or []
#     try:
#         response_text, updated_history = context.qwen_generator.generate(prompt=prompt, messages=messages, enable_thinking=enable_thinking, model_name=model_name, temperature=temperature)
#         return response_text, updated_history
#     except Exception as e:
#         error_msg = f"Error during qwen_generator.generate() call: {e}"
#         print(error_msg)
#         traceback.print_exc()
#         return error_msg, messages or []

def generate(
    context: VADARContext,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict]] = None,
    images: Optional[List[str]] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = 0.7,
    max_new_tokens: Optional[int] = None,
    enable_thinking: Optional[bool] = False
) -> Tuple[str, List[Dict]]:
    """
    Acts as a universal wrapper to call the generator model available in the context.
    It now supports multimodal inputs (text and images).

    Args:
        context: The application context object containing the 'generator' instance.
        prompt: A single string prompt (used if 'messages' is not provided).
        messages: A list of message dictionaries in OpenAI format.
        images: A list of local file paths for images to be included in the prompt.
        model_name: A specific model name to override the generator's default.
        temperature: A specific temperature to override the generator's default.
        max_new_tokens: A specific max token limit to override the generator's default.

    Returns:
        A tuple containing:
        - The generated text response as a string.
        - The complete, updated conversation history as a list of dictionaries.
    """
    # Check if the generator is available in the context
    # if not hasattr(context, 'generator') or not context.generator:
    #     error_msg = "Error: A 'generator' instance is not available in the provided context."
    #     print(error_msg)
    #     return error_msg, messages or []

    try:
        # Call the generator's generate method, passing all relevant parameters
        response_text, updated_history = context.qwen_generator.generate(
            prompt=prompt,
            messages=messages,
            images=images,
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        return response_text, updated_history

    except Exception as e:
        # Provide a detailed error message if the generation fails
        error_msg = f"Error during generator.generate() call: {e}"
        print(error_msg)
        traceback.print_exc()
        return error_msg, messages or []

def generate_wrapped(context: VADARContext,
                    messages,
                    max_new_tokens,
                    temperature,
                    do_sample,
                    return_full_text,
                    tokenizer,
                    pipeline,
                    logger):
    return generate(context, messages=messages)[0]

def generate_vl(context: VADARContext, prompt: str = None, messages: list = None, images=None, enable_thinking=False):
    # This function uses an 'internvl_generator', which is not initialized.
    # To make this runnable, we'll gracefully handle its absence or swap with another model.
    # For now, let's assume `context` would have an `internvl_generator` if it were initialized.
    if not hasattr(context, 'internvl_generator') or not context.internvl_generator:
        error_msg = "Error: InternVL3 Generator not available in the provided context."
        print(error_msg)
        return error_msg, messages or []

    internvl_generator = context.internvl_generator
    # The rest of the original function logic remains the same, but using the generator from context.
    try:
        pixel_values = None
        num_patches_list = None
        if images:
            if isinstance(images, str):
                images = [images]
            all_pixel_tensors = []
            for img_path in images:
                single_image_tensor = internvl_generator.load_image(img_path)
                all_pixel_tensors.append(single_image_tensor)
            pixel_values_tensor = torch.cat(all_pixel_tensors)
            num_patches_list = [p.shape[0] for p in all_pixel_tensors]
            pixel_values_tensor = pixel_values_tensor.to(internvl_generator.dtype).to(internvl_generator.device)

        if messages:
            internvl_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            last_message = internvl_messages[-1]["content"]
            history = internvl_messages[:-1] if len(internvl_messages) > 1 else None
        elif prompt:
            last_message = prompt
            history = None
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        gen_config = {
            "max_new_tokens": internvl_generator.max_new_tokens,
            "do_sample": internvl_generator.temperature > 0
        }
        if internvl_generator.temperature > 0:
            gen_config.update({"temperature": internvl_generator.temperature, "top_p": 0.9, "top_k": 20})

        if pixel_values_tensor is not None:
            response, new_history = internvl_generator.model.chat(
                internvl_generator.tokenizer, pixel_values_tensor, last_message,
                generation_config=gen_config, num_patches_list=num_patches_list,
                history=history, return_history=True
            )
        else:
            response, new_history = internvl_generator.model.chat(
                internvl_generator.tokenizer, None, last_message,
                generation_config=gen_config, history=history, return_history=True
            )

        response = internvl_generator.remove_substring(internvl_generator.remove_substring(response, "```python"), "```")
        response = internvl_generator.remove_think_tags(response)
        return response, new_history
    except Exception as e:
        print(f"InternVL3 Generator: Error during generation: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}", messages or []


def generate_spatial_vlm_response(
    context: VADARContext,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    rgb_image: Optional[Union[str, Image.Image]] = None,
    depth_image: Optional[Union[str, Image.Image]] = None,
    segmentation_masks: Optional[List[Union[Image.Image, np.ndarray]]] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    model_name: str = "SpatialRGPT-VILA1.5-8B",
    conv_mode: str = "llama_3",
    use_bfloat16: bool = True
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generates a response from the SpatialRGPT model via the SpatialRGPTClient,
    passing user-provided segmentation masks directly.
    """
    client = context.spatial_rgpt_client
    if not client:
        err_msg = "Failed to find SpatialRGPTClient in context."
        return err_msg, (messages if messages is not None else [])

    active_messages: List[Dict[str, Any]]
    if messages:
        active_messages = [dict(m) for m in messages]
    elif prompt:
        active_messages = [{"role": "user", "content": prompt}]
    else:
        err_msg = "Error: Either 'prompt' or 'messages' must be provided."
        return err_msg, (messages if messages is not None else [])

    if segmentation_masks and not rgb_image:
        err_msg = "Error: If segmentation_masks are provided, an rgb_image must also be provided."
        return err_msg, active_messages

    process_depth_flag: Optional[bool] = depth_image is not None

    try:
        messages_before_api_call = [dict(m) for m in active_messages]
        response_data = client.chat(
            messages=active_messages,
            model_name=model_name,
            main_image_path_or_pil=rgb_image,
            depth_image_path_or_pil=depth_image,
            segmentation_masks_data=segmentation_masks,
            region_boxes=None,
            use_sam_segmentation=False,
            process_provided_depth=process_depth_flag,
            conv_mode=conv_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            use_bfloat16=use_bfloat16
        )

        if response_data and response_data.get("choices"):
            choice = response_data["choices"][0]
            assistant_message_obj = choice.get("message", {})
            assistant_response_content = assistant_message_obj.get("content")

            if assistant_response_content is None:
                err_msg = "Error: Assistant response content is missing from server payload."
                return err_msg, messages_before_api_call

            assistant_final_message: Dict[str, Any] = {"role": "assistant", "content": assistant_response_content}
            updated_history = messages_before_api_call + [assistant_final_message]
            return assistant_response_content, updated_history
        else:
            err_msg = f"Error: Invalid or empty response structure from server: {response_data}"
            return err_msg, messages_before_api_call

    except (ValueError, requests.exceptions.RequestException) as e:
        err_msg = f"Client or API Call Error: {str(e)}"
        return err_msg, active_messages
    except Exception as e:
        err_trace = traceback.format_exc()
        err_msg = f"An unexpected error occurred: {str(e)}\nTrace: {err_trace}"
        return err_msg, active_messages

def enforce_python_code_output(context: VADARContext, raw_response):
    """
    Reinforces the model output to strictly contain only Python code.
    """
    correction_prompt = (
        "You are a strict Python code formatter. Your task is to take the following text "
        "and extract or rewrite it as a valid Python code block without any extra explanation, "
        "markdown, or text outside of the code. Preserve the original logic and structure."
        "If the code is not valid Python, fix it to ensure it runs correctly. Else, keep it as is / make only minimal changes. "
        "Ensure that the code is properly indented and formatted according to Python standards. "
        "Please remove any type definition / enforcement for the input and output parameters of all function declaration in the code."
        "Look closely at where the code might be faulty. Modify ONLY the parts where you are absolutely sure it's faulty by considering the natural logic of the code. "
        "You must slightly modify the local variables' names in the functions' code if the name is exactly like the function's name being called.\n\n"
        "Input:\n"
        f"{raw_response}\n\n"
        "Output:\n"
    )

    print("Qwen3 Generator: Enforcing Python code format...")
    correction_input = [{"role": "user", "content": correction_prompt}]

    try:
        corrected_output, _ = context.qwen_generator.generate(messages=correction_input, temperature=0.1, model_name=MISTRAL_CODE_MODEL_NAME)
    except Exception as e:
        print(f"Qwen3 Generator: Error during Python code enforcement: {e}")
        raise

    code_block_match = re.search(r'```python\n(.*?)\n```', corrected_output, re.DOTALL)
    if code_block_match:
        print("Qwen3 Generator: Extracted cleaned Python code after refinement.")
        return code_block_match.group(1).strip()
    else:
        print("Qwen3 Generator: No code block found; returning full output as potential code.")
        return corrected_output.strip()

# -- Visual Reasoning Tools (now accept context) --
def save_visualized_image(img, output_path="visualized_scene.png"):
    try:
        img.save(output_path)
        print(f"Visualized image saved to {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Error saving visualized image: {e}")

def loc(context: VADARContext, image, object_prompt):
    # This function uses 'grounding_dino', which is commented out.
    # The refactoring will assume `context.grounding_dino` would exist if it were active.
    # It will fail gracefully if it's not present.
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    trace_html = []

    if not hasattr(context, 'grounding_dino') or not context.grounding_dino:
        trace_html.append("<p>Error: GroundingDINO not available in context for loc().</p>")
        return [], trace_html

    grounding_dino = context.grounding_dino
    gd_device_str = str(context.device)
    # The rest of the original function logic remains the same
    original_object_prompt = object_prompt
    width, height = image.size
    prompt_gd = object_prompt.replace(' ', '.') + " ."

    _, img_gd_tensor = transform_image(image) # Assumes transform_image is available
    is_cpu = gd_device_str == "cpu"

    with torch.autocast(device_type=gd_device_str.split(":")[0], enabled=not is_cpu, dtype=torch.float16 if not is_cpu else torch.float32):
        boxes_tensor, logits, phrases = predict(
            model=grounding_dino,
            image=img_gd_tensor,
            caption=prompt_gd,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=gd_device_str,
        )
    bboxes = _parse_bounding_boxes(boxes_tensor, width, height)

    if not bboxes:
        trace_html.append(f"<p>Locate '{original_object_prompt}': No objects found</p>")
        return [], trace_html

    trace_html.append(f"<p>Locate: {original_object_prompt}</p>")
    boxed_pil_image = box_image(image, bboxes)
    save_visualized_image(boxed_pil_image, "loc_output.png")
    trace_html.append(html_embed_image(boxed_pil_image))
    display_prompt = original_object_prompt + ("s" if len(bboxes) > 1 and not original_object_prompt.endswith("s") else "")
    trace_html.append(f"<p>{len(bboxes)} {display_prompt} found</p>")
    trace_html.append(f"<p>Boxes: {bboxes}</p>")

    return bboxes, trace_html

def extract_2d_bounding_box(detected_object: DetectedObject) -> Tuple[np.ndarray, List[str]]:
    trace_html = []
    bbox = detected_object.bounding_box_2d.astype(int)
    trace_html.append(f"<h4>Extract 2D Bounding Box</h4>")
    trace_html.append(f"<p>From object: '<b>{detected_object.description}</b>'</p>")
    trace_html.append(f"<p>Result: <b>{bbox.tolist()}</b></p>")
    return bbox, trace_html

def extract_3d_bounding_box(detected_object: DetectedObject) -> Tuple[o3d.utility.Vector3dVector, List[str]]:
    trace_html = []
    points_vector = detected_object.bounding_box_3d_oriented.get_box_points()
    trace_html.append(f"<h4>Extract 3D Bounding Box Points</h4>")
    trace_html.append(f"<p>From object: '<b>{detected_object.description}</b>'</p>")
    trace_html.append(f"<p>Result: Extracted <b>{len(points_vector)}</b> points for the oriented bounding box.</p>")
    return points_vector, trace_html

def is_similar_text(context: VADARContext, text1: str, text2: str) -> Tuple[bool, List[str]]:
    trace_html = []
    threshold = 0.5
    prompt_embedding = context.embedding_model.encode([text1], convert_to_numpy=True)
    class_embedding = context.embedding_model.encode([text2], convert_to_numpy=True)
    similarity = cosine_similarity(prompt_embedding, class_embedding)[0][0]
    answer = similarity > threshold
    trace_html.append(f"<h4>Is Similar Text</h4><p>Comparing '{text1}' and '{text2}'.</p>")
    trace_html.append(f"<p>Similarity Score: <b>{similarity:.4f}</b> (Threshold: >{threshold})</p>")
    trace_html.append(f"<p>Result: <b>{answer}</b></p>")
    return answer, trace_html

def retrieve_objects(context: VADARContext, detected_objects: List[DetectedObject], object_prompt: str) -> Tuple[List[DetectedObject], List[str]]:
    trace_html = []
    if not detected_objects:
        trace_html.append("<p>Retrieve Objects: Called with an empty list of detected_objects.</p>")
        return [], trace_html

    trace_html.append(f"<h4>Retrieve Objects</h4><p>Prompt: '<b>{object_prompt}</b>'</p>")

    if object_prompt.strip().lower() in ("object", "objects", "items", "things"):
        trace_html.append(f"<p>Generic prompt detected. Returning all {len(detected_objects)} objects.</p>")
        return detected_objects, trace_html

    class_names = [obj.class_name for obj in detected_objects]
    prompt_embedding = context.embedding_model.encode([object_prompt], convert_to_numpy=True)
    class_embeddings = context.embedding_model.encode(class_names, convert_to_numpy=True)
    similarities = cosine_similarity(prompt_embedding, class_embeddings)[0]

    scored_objects = sorted([(s, o) for s, o in zip(similarities, detected_objects) if s > 0.9], reverse=True, key=lambda x: x[0])
    result_objects = [obj for score, obj in scored_objects]

    trace_html.append(f"<p>Found <b>{len(result_objects)}</b> matching objects (similarity > 0.9) from a total of {len(detected_objects)}.</p>")
    if scored_objects:
        table = "<table border='1' style='border-collapse: collapse; width: 100%;'><tr><th>Similarity</th><th>Class Name</th><th>Description</th></tr>"
        for score, obj in scored_objects:
            table += f"<tr><td>{score:.3f}</td><td>{obj.class_name}</td><td>{obj.description}</td></tr>"
        table += "</table>"
        trace_html.append(table)

    return result_objects, trace_html

def get_3D_object_size(detected_object: DetectedObject) -> Tuple[Tuple[float, float, float], List[str]]:
    trace_html = []
    obb = detected_object.bounding_box_3d_oriented
    extent = obb.extent
    width, length, height = float(extent[0]), float(extent[1]), float(extent[2])
    result_tuple = (width, height, length)
    trace_html.append(f"<h4>Get 3D Object Size</h4><p>From object: '<b>{detected_object.description}</b>'</p>")
    trace_html.append(f"<p>OBB Extent (W, D, H): [{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}] meters</p>")
    trace_html.append(f"<p>Result (Width, Height, Length): <b>{result_tuple}</b> meters</p>")
    return result_tuple, trace_html

# In _vqa_predict
def _vqa_predict(context: VADARContext, img, question, holistic=False):
    try:
        prompt = VQA_PROMPT.format(question=question)
        full_prompt = f"<image> {prompt}"
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, "temp_image.png")
            img.save(temp_path)
            # The call is now simpler. No context=context needed.
            # output, _ = context.qwen_generator.generate(prompt=full_prompt, images=[temp_path], model_name=MISTRAL_VISION_MODEL_NAME)
            try:
                output, _ = context.gemini_generator.generate(prompt=full_prompt, images=[temp_path], model_name=gemini_model_name_20)
            except Exception as e:
                print(f"Gemini VQA prediction failed: {e}")
                print('Falling back to Qwen VQA model.')
                output, _ = context.qwen_generator.generate(prompt=full_prompt, images=[temp_path], model_name=MISTRAL_VISION_MODEL_NAME)
        answer_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
        return answer_match.group(1).strip().lower() if answer_match else output.strip().lower()
    except Exception as e:
        print(f"Error in VQA prediction: {e}")
        return f"Error: {str(e)}"
        
def vqa(context: VADARContext, image: PILImage.Image, question: str, object: DetectedObject):
    trace_html = []
    bbox, mask_2d = object.bounding_box_2d, object.segmentation_mask_2d
    is_holistic = False
    img_for_vqa_display = image

    if bbox is None or (bbox[0] <= 0 and bbox[1] <= 0 and bbox[2] >= image.width - 1 and bbox[3] >= image.height - 1):
        is_holistic = True
        trace_html.append(f"<p>VQA (holistic query): {question}</p>" + html_embed_image(image, 300))
    else:
        try:
            cmin, rmin, cmax, rmax = [int(c) for c in bbox]
            if cmax > cmin and rmax > rmin:
                cropped_image_pil = image.crop((cmin, rmin, cmax, rmax))
                cropped_mask_np = mask_2d[rmin:rmax, cmin:cmax]
                cropped_image_rgba = cropped_image_pil.convert('RGBA')
                cropped_image_np = np.array(cropped_image_rgba)
                cropped_image_np[:, :, 3] = (cropped_mask_np * 255).astype(np.uint8)
                img_for_vqa_display = PILImage.fromarray(cropped_image_np, 'RGBA')
                trace_html.append(f"<p>VQA (region query): {question}</p>")
                trace_html.append("<p>Query region on original image:</p>" + html_embed_image(box_image(image, [bbox]), 300))
                trace_html.append("<p>Masked & Cropped region (for VLM):</p>" + html_embed_image(img_for_vqa_display, 200))
            else:
                is_holistic = True
                trace_html.append(f"<p>VQA (holistic due to invalid crop {bbox}): {question}</p>" + html_embed_image(image, 300))
        except (ValueError, TypeError):
            is_holistic = True
            trace_html.append(f"<p>VQA (holistic due to invalid bbox format {bbox}): {question}</p>" + html_embed_image(image, 300))

    answer = _vqa_predict(context, img_for_vqa_display, remake_query(question), holistic=is_holistic)
    trace_html.append(f"<p>VQA Final Answer (from _vqa_predict): {answer}</p>")
    return answer.lower(), trace_html

def _vqa_predict2(context: VADARContext, img, depth, masks, question):
    try:
        full_prompt = VQA_PROMPT.format(question=question)
        output, _ = generate_spatial_vlm_response(context, prompt=full_prompt, rgb_image=img, depth_image=depth, segmentation_masks=masks, temperature=0.2)
        answer_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
        return answer_match.group(1).strip().lower() if answer_match else output.strip().lower()
    except Exception as e:
        print(f"Error in VQA prediction: {e}")
        return f"Error: {str(e)}"

def remake_query(query, tag=''):
    counter = [0]
    def replacer(match):
        replacement = f"<region{counter[0]}{tag}>"
        counter[0] += 1
        return replacement
    return re.sub(r'<mask>', replacer, query)

def invert_query(query):
    return re.sub(r'<region\d+>', '<mask>', query)

def vqa2(context: VADARContext, image, depth, question, objects):
    trace_html = []
    refined_question = remake_query(question)
    trace_html.append(f"<p>VQA Question: {refined_question}</p>")
    trace_html.append(str([obj.description for obj in objects]))
    print(f"VQA Question: {refined_question}")
    answer = _vqa_predict2(context, image, depth, [det_obj.segmentation_mask_2d for det_obj in objects], refined_question)
    trace_html.append(f"<p>VQA Final Answer (from _vqa_predict): {answer}</p>")
    return answer.lower(), trace_html

def _get_iou(box1, box2):
    if not (isinstance(box1, (list, tuple, np.ndarray)) and len(box1) == 4 and all(isinstance(c, (int, float, np.integer, np.floating)) for c in box1) and
            isinstance(box2, (list, tuple, np.ndarray)) and len(box2) == 4 and all(isinstance(c, (int, float, np.integer, np.floating)) for c in box2)):
        print(f"Warning: Invalid box format for IoU calculation. Box1: {box1}, Box2: {box2}")
        return 0.0
    x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
    area_inter = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    area_union = area_box1 + area_box2 - area_inter
    return area_inter / area_union if area_union > 1e-6 else 0.0

def same_object(image, bbox1, bbox2):
    iou_val = _get_iou(bbox1, bbox2)
    answer = iou_val > 0.92
    trace_html = []
    boxes_to_draw = [b for b in [bbox1, bbox2] if isinstance(b, (list, tuple)) and len(b) == 4]
    if boxes_to_draw:
        trace_html.append(f"<p>Same Object Check (Bbox1: {bbox1}, Bbox2: {bbox2})</p>" + html_embed_image(box_image(image, boxes_to_draw), 300))
    else:
        trace_html.append(f"<p>Same Object Check with invalid bboxes (Bbox1: {bbox1}, Bbox2: {bbox2}). Cannot draw.</p>")
    trace_html.append(f"<p>IoU: {iou_val:.3f}, Same object: {answer}</p>")
    return answer, trace_html

def find_overlapping_regions(parent_region: DetectedObject, countable_regions: List[DetectedObject]) -> Tuple[List[int], List[str]]:
    """
    Find regions that overlap with parent region using full image masks
    
    Args:
        parent_region: Parent region to check overlaps against
        countable_regions: List of regions that can be counted
        
    Returns:
        Tuple of (overlapping_region_indices, trace_html)
    """
    trace_html = [
        f"<h4>Find Overlapping Regions</h4>",
        f"<p>Parent region: '<b>{parent_region.description}</b>'</p>",
        f"<p>Checking against <b>{len(countable_regions)}</b> candidate regions.</p>"
    ]
    
    overlap_threshold = 0.2  # Updated to match RegionCounter default
    parent_mask = parent_region.segmentation_mask_2d
    overlapping_regions = []
    overlap_details = []

    for i, region in enumerate(countable_regions):
        # Skip parent region itself
        if region.description == parent_region.description:
            continue
            
        region_mask = region.segmentation_mask_2d
        
        # Calculate overlap on full image (key improvement from RegionCounter)
        overlap = np.logical_and(parent_mask, region_mask)
        overlap_area = np.sum(overlap)
        
        if overlap_area > 0:
            # Calculate overlap percentage relative to the child region's TOTAL area
            total_region_area = np.sum(region_mask)
            overlap_percentage = overlap_area / total_region_area if total_region_area > 0 else 0
            
            trace_html.append(
                f"<p>Region '<b>{region.description}</b>': {overlap_area} overlap pixels, "
                f"{total_region_area} total pixels, {overlap_percentage:.2%} overlap</p>"
            )
            
            # Check if overlap meets threshold
            if overlap_percentage >= overlap_threshold:
                # Extract region index from description
                match = re.search(r'\d+', region.description)
                if match:
                    region_idx = int(match.group())
                    overlapping_regions.append(region_idx)
                    overlap_details.append((region.description, overlap_percentage))
                    trace_html.append(f"<p style='color: green;'>✓ Region {region_idx} meets threshold ({overlap_percentage:.2%} >= {overlap_threshold:.1%})</p>")
                else:
                    trace_html.append(f"<p style='color: orange;'>⚠ Could not extract index from '{region.description}'</p>")
            else:
                trace_html.append(f"<p style='color: red;'>✗ Below threshold ({overlap_percentage:.2%} < {overlap_threshold:.1%})</p>")

    trace_html.append(
        f"<p><b>Result:</b> Found <b>{len(overlapping_regions)}</b> overlapping regions "
        f"with indices: <b>{overlapping_regions}</b></p>"
    )
    
    if overlap_details:
        trace_html.append("<p><b>Overlap Details:</b></p><ul>")
        for desc, pct in overlap_details:
            trace_html.append(f"<li>{desc}: {pct:.2%} overlap</li>")
        trace_html.append("</ul>")

    return overlapping_regions, trace_html

def calculate_3d_distance(obj1: DetectedObject, obj2: DetectedObject) -> Tuple[float, List[str]]:
    trace_html = []
    center1, center2 = obj1.bounding_box_3d_oriented.get_center(), obj2.bounding_box_3d_oriented.get_center()
    distance = np.linalg.norm(center1 - center2)
    adjusted_distance = distance + distance * 0.26
    trace_html.append(f"<h4>Calculate 3D Distance</h4>")
    trace_html.append(f"<p>Object 1: '<b>{obj1.description}</b>' (Center: {np.round(center1, 3).tolist()})</p>")
    trace_html.append(f"<p>Object 2: '<b>{obj2.description}</b>' (Center: {np.round(center2, 3).tolist()})</p>")
    trace_html.append(f"<p>Euclidean distance: <b>{distance:.4f}</b> meters</p>")
    trace_html.append(f"<p>Adjusted distance (+22%): <b>{adjusted_distance:.4f}</b> meters</p>")
    return adjusted_distance, trace_html

# def calculate_3d_distance(obj1: DetectedObject, obj2: DetectedObject) -> Tuple[float, List[str]]:
#     """
#     Calculates the minimum surface distance between two 3D oriented bounding boxes using Open3D.
#     Returns the distance and trace information for debugging/logging.
#     """
#     import open3d as o3d
#     import numpy as np
#     from typing import Tuple, List
    
#     trace_html = []
#     obb1 = obj1.bounding_box_3d_oriented
#     obb2 = obj2.bounding_box_3d_oriented
    
#     # Get centers for trace output
#     center1 = obb1.get_center()
#     center2 = obb2.get_center()
    
#     trace_html.append(f" Calculate 3D Distance ")
#     trace_html.append(f" Object 1: '**{obj1.description}**' (Center: {np.round(center1, 3).tolist()}) ")
#     trace_html.append(f" Object 2: '**{obj2.description}**' (Center: {np.round(center2, 3).tolist()}) ")
    
#     try:
#         def create_transformed_box(obb):
#             box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
#             box.compute_vertex_normals()
#             vertices = np.asarray(box.vertices)
#             scaled_vertices = vertices * obb.extent
#             box.vertices = o3d.utility.Vector3dVector(scaled_vertices)
#             # Apply rotation and translation
#             T = np.eye(4)
#             T[:3, :3] = obb.R
#             T[:3, 3] = obb.center
#             box.transform(T)
#             return box

#         mesh1 = create_transformed_box(obb1)
#         mesh2 = create_transformed_box(obb2)
        
#         # Sample points from mesh2 and measure distance to mesh1
#         pcd2 = mesh2.sample_points_uniformly(number_of_points=1000)
#         distances = np.asarray(mesh1.compute_point_cloud_distance(pcd2))
#         min_surface_distance = float(np.min(distances))
        
#         trace_html.append(f" Surface-to-surface distance: **{min_surface_distance:.4f}** meters ")
        
#         # Apply adjustment factor
#         adjusted_distance = min_surface_distance + min_surface_distance * 0.25
#         trace_html.append(f" Adjusted distance (+25%): **{adjusted_distance:.4f}** meters ")
        
#         return adjusted_distance, trace_html
        
#     except Exception as e:
#         trace_html.append(f" Error in surface distance calculation: {str(e)} ")
#         trace_html.append(f" Falling back to corner-to-corner distance ")
        
#         # Fall back to min distance between corners
#         points1 = np.asarray(obb1.get_box_points())
#         points2 = np.asarray(obb2.get_box_points())
#         distances = np.linalg.norm(points1[:, None, :] - points2[None, :, :], axis=2)
#         min_corner_distance = float(np.min(distances))
        
#         trace_html.append(f" Corner-to-corner distance: **{min_corner_distance:.4f}** meters ")
        
#         # Apply adjustment factor
#         adjusted_distance = min_corner_distance + min_corner_distance * 0.25
#         trace_html.append(f" Adjusted distance (+25%): **{adjusted_distance:.4f}** meters ")
        
#         return adjusted_distance, trace_html

def get_2D_object_size(image, bbox):
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(c, (int, float)) for c in bbox)):
        return (0, 0), [f"<p>Error: Invalid bbox format for get_2D_object_size(): {bbox}.</p>"]
    width, height = abs(bbox[0] - bbox[2]), abs(bbox[1] - bbox[3])
    trace_html = [f"<p>2D Object Size for Bbox: {bbox}</p>", html_embed_image(box_image(image, [bbox]), 300), f"<p>Width: {width}, Height: {height}</p>"]
    return (width, height), trace_html

def wrap_solution_code(solution_program_code):
  indented_code = "\n".join("    " + line if line.strip() != "" else line for line in solution_program_code.splitlines())
  return f"\ndef solution_program(image, detected_objects):\n{indented_code}\n    return final_result\nfinal_result = solution_program(image, detected_objects)"

# NEW, MORE ROBUST PROMPT FOR CODE CORRECTION
PROGRAM_CORRECTION_PROMPT_V2 = """
You are an expert Python debugger for an agentic system. The system failed to execute a program composed of two main parts: a `generated_api` containing helper functions, and a `solution_program` which calls those functions to achieve a goal.

Your task is to analyze the traceback and the provided code components, identify the precise location of the bug, and provide a corrected version of **only the faulty component**.

**Execution Context:**
- The `solution_program` is the main entry point.
- The `solution_program` calls functions defined in the `generated_api`.
- The `generated_api` functions, in turn, may call pre-defined, trusted tools (not shown here, but assume they are correct).
- The bug could be in the `generated_api` (e.g., a bug in a helper function's logic) or in the `solution_program` (e.g., calling an API function with wrong arguments, or incorrect logic).

**Provided Information:**

**0. Custom types definitions:**
```python
{custom_types}
```

**1. Generated API (Potentially Faulty):**
```python
{api_code}
```

**2. Solution Program (Potentially Faulty):**
```python
{program_code}
```

**3. Full Error Traceback:**
```
{traceback}
```

**Your Task & Required Output Format:**

Analyze the information above and respond with a single JSON object. Do not include any other text or explanations outside of the JSON block. The JSON object must have the following structure:

```json
{{
  "analysis": "A brief, one-sentence explanation of the root cause of the error. For example: 'The solution_program passed a list to calculate_3d_distance, which expects a single DetectedObject.' or 'The is_red function in the API incorrectly checked for the color blue.'",
  "faulty_component": "api" or "solution_program",
  "corrected_code": "The full, corrected code for ONLY the component you identified as faulty. Do not include the code for the other, non-faulty component."
}}
```

**Example 1: Bug in `solution_program`**
If the traceback shows a `TypeError` when calling a function, your response should look like this:
```json
{{
  "analysis": "The solution program incorrectly passed a list of objects to `get_3D_object_size`, which requires a single `DetectedObject`.",
  "faulty_component": "solution_program",
  "corrected_code": "final_result = None\\nall_objects = retrieve_objects(detected_objects, 'object')\\nif all_objects:\\n    # Correctly iterate and call the function on the first object\\n    size = get_3D_object_size(all_objects[0])\\n    final_result = size"
}}
```

**Example 2: Bug in `api`**
If the traceback shows an error inside one of the API functions, your response should look like this:
```json
{{
  "analysis": "The `find_largest_object` function had a logic error, initializing max_size to a large number instead of zero, causing it to always return None.",
  "faulty_component": "api",
  "corrected_code": "def find_largest_object(objects):\\n    if not objects:\\n        return None\\n    largest_object = None\\n    max_size = 0  # Correctly initialized to 0\\n    for obj in objects:\\n        size = get_3D_object_size(obj)[0] # Assume width is size\\n        if size > max_size:\\n            max_size = size\\n            largest_object = obj\\n    return largest_object"
}}
```

Now, provide your diagnosis and correction for the provided code and traceback.
"""

def get_tool_function_hint_from_traceback(tb) -> Optional[str]:
    """
    Walks a traceback to find a call to '_trace_and_run' and extracts the
    signature of the tool function that was being called.

    This provides a crucial hint for debugging, especially for type-related errors,
    by showing what signature the failing function expected.

    Args:
        tb: A traceback object from `sys.exc_info()`.

    Returns:
        A formatted string with the hint, or None if the specific frame isn't found.
    """
    hint = None
    # Walk the traceback frames from the innermost (where the error occurred)
    # to the outermost.
    for frame, lineno in traceback.walk_tb(tb):
        # We are looking for the specific wrapper function that calls all tools.
        if frame.f_code.co_name == '_trace_and_run':
            # The tool function itself is stored in the 'func' local variable of this frame.
            if 'func' in frame.f_locals:
                tool_func = frame.f_locals['func']
                try:
                    # Use inspect to get the function's signature.
                    sig = inspect.signature(tool_func)
                    # Format it into a readable string for the hint.
                    hint = (f"HINT: The error occurred inside the tool function "
                            f"'{tool_func.__name__}' which has the signature: {str(sig)}")
                    # We found our hint, so we can stop searching.
                    break
                except (ValueError, TypeError):
                    # Fallback if the object is not a standard inspect-able function.
                    hint = (f"HINT: The failing tool function was '{tool_func.__name__}', "
                            "but its signature could not be inspected.")
                    break
    return hint

def execute_program(context: VADARContext, program: str, image: PILImage.Image, depth, detected_objects: List[DetectedObject], api: str) -> Tuple[Any, str]:
    """
    Executes the generated program with robust, structured error correction and
    adds contextual hints to tracebacks.
    """
    max_retries = 2
    current_program_code = program
    current_api_code = api
    full_html_trace = []

    header_lines = [
        "import math", "from typing import Tuple, List, Dict, Optional",
        "from PIL import Image as PILImage, ImageDraw", "import numpy as np",
        "import open3d as o3d", "import io, base64, sys, os, re, tempfile, json, time, torch",
        "from pathlib import Path", "import PIL", ""
    ]
    header_str = "\n".join(header_lines)
    api_methods = re.findall(r"def (\w+)\s*\(.*\):", current_api_code)

    for attempt in range(max_retries + 1):
        full_html_trace.append(f"<h2>Execution Attempt {attempt + 1}</h2>")
        
        full_html_trace.append("<h4>API Code for This Attempt:</h4>" + f"<pre style='background-color:#f0f0f0; border:1px solid #ccc; padding:10px; border-radius:5px;'><code>{current_api_code}</code></pre>")
        full_html_trace.append("<h4>Solution Program for This Attempt:</h4>" + f"<pre style='background-color:#f0f0f0; border:1px solid #ccc; padding:10px; border-radius:5px;'><code>{current_program_code}</code></pre>")

        wrapped_program = wrap_solution_code(current_program_code)
        executable_program = f"{header_str}\n\n{current_api_code}\n\n{wrapped_program}"
        
        program_lines = executable_program.split("\n")
        def get_line(line_no):
            return program_lines[line_no - 1] if 0 <= line_no - 1 < len(program_lines) else ""
        
        attempt_trace = []
        def trace_lines(frame, event, arg):
            if frame.f_code.co_name == 'trace_lines': return trace_lines
            if event == "line":
                method_name, line_no = frame.f_code.co_name, frame.f_lineno
                if method_name == "solution_program" or method_name in api_methods:
                    line = get_line(line_no).strip()
                    if line: attempt_trace.append(f"<p><code>[{method_name}] Line {line_no}: {line}</code></p>")
            return trace_lines

        namespace = {
            "DetectedObject": DetectedObject, "image": image, "detected_objects": detected_objects,
            "loc": lambda *a, **kw: _trace_and_run(loc, attempt_trace, context, *a, **kw),
            "retrieve_objects": lambda *a, **kw: _trace_and_run(retrieve_objects, attempt_trace, context, *a, **kw),
            "vqa": lambda *a, **kw: _trace_and_run(vqa, attempt_trace, context, *a, **kw),
            "extract_2d_bounding_box": lambda *a, **kw: _trace_and_run(extract_2d_bounding_box, attempt_trace, *a, **kw),
            "extract_3d_bounding_box": lambda *a, **kw: _trace_and_run(extract_3d_bounding_box, attempt_trace, *a, **kw),
            "get_3D_object_size": lambda *a, **kw: _trace_and_run(get_3D_object_size, attempt_trace, *a, **kw),
            "find_overlapping_regions": lambda *a, **kw: _trace_and_run(find_overlapping_regions, attempt_trace, *a, **kw),
            "calculate_3d_distance": lambda *a, **kw: _trace_and_run(calculate_3d_distance, attempt_trace, *a, **kw),
            "is_similar_text": lambda *a, **kw: _trace_and_run(is_similar_text, attempt_trace, context, *a, **kw),
        }
        
        sys.settrace(trace_lines)
        try:
            exec(executable_program, namespace)
            sys.settrace(None)
            final_result = namespace.get("final_result", "Error: final_result not found.")
            full_html_trace.append("<h3><span style='color:green;'>Success</span></h3>" + "".join(attempt_trace))
            console.print(f"--- Program execution successful on attempt {attempt + 1} ---", style="bold green")
            return final_result, "".join(full_html_trace)
        except Exception:
            sys.settrace(None)
            # --- START OF MODIFIED EXCEPTION HANDLING ---
            exc_type, exc_value, exc_traceback = sys.exc_info()
            
            # Get the standard traceback string
            traceback_str_original = traceback.format_exc()

            # Get our custom contextual hint
            tool_hint = get_tool_function_hint_from_traceback(exc_traceback)
            
            # Prepare the augmented traceback string for logging and the LLM
            traceback_for_llm = traceback_str_original
            hint_html = ""
            if tool_hint:
                traceback_for_llm += f"\n\n--- CONTEXTUAL HINT ---\n{tool_hint}\n"
                hint_html = f"<div style='background-color:#fff3cd; color:#856404; border:1px solid #ffeeba; padding:10px; border-radius:5px; margin-top:10px;'><strong>{tool_hint}</strong></div>"
            
            # Add the failure information and styled hint to the HTML trace
            full_html_trace.append(
                "<h3><span style='color:red;'>Failed</span></h3>" + "".join(attempt_trace) +
                f"<h4>Error Traceback:</h4><pre style='background-color:#ffebeb; color:red; border:1px solid red; padding:10px; border-radius:5px;'>{traceback_str_original}</pre>" +
                hint_html
            )
            
            console.print(f"--- Program execution failed on attempt {attempt + 1} ---\n{traceback_for_llm}", style="bold red")
            # --- END OF MODIFIED EXCEPTION HANDLING ---

            if attempt >= max_retries:
                full_html_trace.append("<h3>Max retries reached. Aborting.</h3>")
                return "None", "".join(full_html_trace)

            # --- Intelligent Correction Block ---
            console.print("--- Requesting intelligent code correction from Gemini... ---", style="yellow")
            correction_prompt = PROGRAM_CORRECTION_PROMPT_V2.format(
                custom_types=detected_object_definition,
                api_code=current_api_code,
                program_code=current_program_code,
                traceback=traceback_for_llm  # Use the augmented traceback string
            )
            
            try:
                correction_response, _ = context.gemini_generator.generate(prompt=correction_prompt, max_new_tokens=32000, temperature=0.1, model_name=gemini_model_name_25)
            except Exception as e:
                console.print(f"--- Gemini correction failed: {e}. Retrying with Qwen... ---", style="bold red")
                full_html_trace.append(f"<p><strong>Gemini correction call failed: {e}. Retrying with Qwen.</strong></p>")
                # Fallback to Qwen if Gemini fails
                try:
                    correction_response, _ = context.qwen_generator.generate(prompt=correction_prompt, max_new_tokens=32000, temperature=0.1, model_name=MISTRAL_FIX_MODEL_NAME)
                except Exception as qwen_e:
                    console.print(f"--- Qwen correction also failed: {qwen_e}. Returning original code. ---", style="bold red")
                    full_html_trace.append(f"<p><strong>Qwen correction call failed: {qwen_e}. Returning original code.</strong></p>")
                    continue  # Continue to the next attempt without changing the code
            
            # Log the LLM correction response
            console.print(f"[bold cyan]LLM Correction Response:[/bold cyan] {correction_response}")
            full_html_trace.append(f"<h4>LLM Correction Response:</h4><pre style='background-color:#f0f0f0; border:1px solid #ccc; padding:10px; border-radius:5px;'>{correction_response}</pre>")
            try:
                # Use a low temperature for deterministic fixes
                # correction_response, _ = context.qwen_generator.generate(prompt=correction_prompt, max_new_tokens=32000, temperature=0.1, model_name=MISTRAL_FIX_MODEL_NAME)

                # Extract the JSON from the response
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', correction_response, re.DOTALL)
                if not json_match:
                    raise ValueError("LLM response did not contain a valid JSON block.")
                
                correction_data = json.loads(json_match.group(1))
                analysis = correction_data.get("analysis", "No analysis provided.")
                faulty_component = correction_data.get("faulty_component")
                corrected_code = correction_data.get("corrected_code")

                if not faulty_component or not corrected_code:
                    raise ValueError("LLM response JSON is missing 'faulty_component' or 'corrected_code'.")

                console.print(f"[bold cyan]LLM Diagnosis:[/bold cyan] {analysis}")
                full_html_trace.append(f"<h4>LLM Diagnosis:</h4><p><em>{analysis}</em></p>")

                # Surgically apply the fix
                if faulty_component == "api":
                    console.print("--- Applying correction to [bold]Generated API[/bold] and retrying... ---", style="yellow")
                    current_api_code = corrected_code
                elif faulty_component == "solution_program":
                    console.print("--- Applying correction to [bold]Solution Program[/bold] and retrying... ---", style="yellow")
                    current_program_code = corrected_code
                else:
                    raise ValueError(f"Unknown 'faulty_component': {faulty_component}")

            except (Exception, json.JSONDecodeError, ValueError) as llm_e:
                console.print(f"--- LLM-based correction failed: {llm_e}. Retrying without changes... ---", style="bold red")
                full_html_trace.append(f"<p><strong>LLM correction call failed: {llm_e}. Retrying with previous code.</strong></p>")
                # Continue to the next attempt without changing the code
                continue

    return "None", "".join(full_html_trace)
    
def _trace_and_run(func, trace_list, *args, **kwargs):
    result, html = func(*args, **kwargs)
    if html: trace_list.extend(html)
    return result

def display_result(final_result, image_pil, question, ground_truth):
    result_html_parts = [f"<h3>Question:</h3><p>{question}</p>", "<h3>Input Image:</h3>", html_embed_image(image_pil, 300), f"<h3>Program Result:</h3><p>{final_result}</p>"]
    if ground_truth is not None: result_html_parts.append(f"<h3>Ground Truth:</h3><p>{ground_truth}</p>")
    return "\n".join(result_html_parts)

def extract_descriptions(context: VADARContext, query: str, objects: list[str], similarity_threshold=0.8) -> dict:
    doc = context.spacy_nlp(query)
    result, deduped = defaultdict(list), {}

    for token in doc:
        if token.text.lower() in objects:
            for child in token.children:
                if child.dep_ in ("amod", "compound"): result[token.text.lower()].append(child.text)
            for left in token.lefts:
                if left.pos_ == "ADJ": result[token.text.lower()].append(left.text)

    for obj, adjs in result.items():
        if not adjs:
            deduped[obj] = []
            continue
        embeddings, used, clusters = context.embedding_model.encode(adjs), set(), []
        for i in range(len(adjs)):
            if i in used: continue
            cluster = [adjs[i]]
            used.add(i)
            for j in range(i + 1, len(adjs)):
                if j in used: continue
                if util.cos_sim(embeddings[i], embeddings[j]).item() >= similarity_threshold:
                    cluster.append(adjs[j])
                    used.add(j)
            representative = min(cluster, key=lambda w: -word_frequency(w, 'en'))
            clusters.append(representative)
        deduped[obj] = clusters

    return deduped

def generate_description_functions(context: VADARContext, query: str, objects: list[str]):
    descriptions = extract_descriptions(context, query, objects)
    function_lines = []
    for obj, descs in descriptions.items():
        for desc in descs:
            func_name = f"is_{obj}_{desc}".replace(" ", "_")
            function_lines.append(f"def {func_name}(image, detected_object):\n  # Check if {obj} is {desc}\n  return vqa(image, 'Is {obj} <region0> {desc}?', detected_object)\n")
    return "\n".join(function_lines)

def signature_agent(context: VADARContext, predef_api, query):
    prompt = SIGNATURE_PROMPT.format(signatures=predef_api, question=query)
    print("Signature Agent Prompt (first 200 chars):\n", prompt[:200] + "...")
    console.print(Padding(f"[Signature Agent] Query: {query}", (1, 2), style="on blue"))
    output_text, _ = context.qwen_generator.generate(prompt=prompt, temperature=0.3, model_name=MISTRAL_CODE_MODEL_NAME)
    if not isinstance(output_text, str) or output_text.startswith('Error:'):
        print(f"Signature Agent Error or invalid output: {output_text}")
        return [], []
    docstrings = re.findall(r"<docstring>(.*?)</docstring>", output_text, re.DOTALL)
    signatures = re.findall(r"<signature>(.*?)</signature>", output_text, re.DOTALL)
    return signatures, docstrings

def expand_template_to_instruction(json_obj):
    instructions = []
    if explanation := json_obj.get("explanation"): instructions.append(f"\nClarify the request: {explanation}.")
    # for i, check in enumerate(json_obj.get("visual_checks", []), 1):
    #     obj, adj, vqa_call = check.get("object", "object"), check.get("adjective", "property"), check.get("vqa_call", "vqa(...)")
    #     instructions.append(f"{i}. Check visually if the '{obj}' is '{adj}' by calling:\n   → {vqa_call}\n   You need to implement this function or route it to your VQA module.")
    if spatial_steps := json_obj.get("spatial_instructions", []):
        instructions.append("\nThen follow these spatial steps:")
        for i, step in enumerate(spatial_steps, 1): instructions.append(f"  {i}. {step}")
    return "\n".join(instructions)

# --- Updated query_expansion Function ---

def query_expansion(context: VADARContext, img, query: str):
    # 1. Define the static parts of the prompt (header and footer)
    prompt_header = """
# Warehouse Logistics Query Analysis

## Objective
Analyze logistics queries to clarify vague language and provide actionable spatial instructions.

## Process

### 1. Query Analysis
- Identify core request and action needed
- Define vague terms ("best", "optimal", "suitable") with concrete criteria
- Clarify spatial relationships

### 2. Visual Verification (VQA)
**PURPOSE**: Check ONLY intrinsic object properties visible on the object itself. NEVER check relative positions, distances, or external states. NEVER call VQA on pallets.

**ALLOWED** (object's own state):
- Empty/loaded status (transporter ONLY): "Is this <mask_tag> (transporter) empty?"

**FORBIDDEN** (requires external context):
- Spatial relationships: "Is this pallet in the left zone?", "Is this item in the right zone?"
- Comparisons: "Is this the closest/largest/best?"
- System states: "Is this transporter available/ready?"
- Distance-based: "Is this near the exit?"
- Physical condition: "Is this <mask_tag> (machine) damaged?"
- Operational state: "Is this <mask_tag> (conveyor) running?"
- Accessibility: "Is this <mask_tag> (path) blocked?"

**RULE**: VQA must be answerable by looking only at the cropped object without knowing about other objects, zones, systems, or its position in space.

### 3. Spatial Instructions
Provide 3-5 clear action steps for spatial logic and decision-making.

## Output Format
```json
{
  "explanation": "Concrete definition of vague terms (1 sentence max)",
  "visual_checks": [
    {
      "object": "object_name",
      "adjective": "property",
      "vqa_call": "vqa(image, 'Is this <mask_tag> (object) [property]?', reference)"
    }
  ],
  "spatial_instructions": [
    "Clear action steps for spatial logic (3-5 steps max)"
  ]
}
```

## Examples
"""

    prompt_footer = "\nInput:\n"

    # 2. Initialize Milvus and retrieve the top 5 few-shot examples
    # This could also be part of the context object if initialization is expensive.
    # Assuming the search function returns a list of dicts. We take the top 5.
    few_shot_examples = context.milvus_manager.search_fewshot(query, top_k=3)

    print(few_shot_examples)

    # 3. Format the retrieved examples into strings
    formatted_examples = []
    for example in few_shot_examples:
        # Reconstruct the JSON object from the example fields
        example_json_obj = {
            "explanation": example.get("explanation", ""),
            "visual_checks": example.get("visual_checks", []),
            "spatial_instructions": example.get("spatial_instructions", [])
        }
        # Convert the Python dict to a formatted JSON string
        example_json_str = json.dumps(example_json_obj, indent=2)

        # Build the complete example string in the required format
        formatted_example = f'**Query**: "{example["query"]}"\n```json\n{example_json_str}\n```'
        formatted_examples.append(formatted_example)

    # 4. Assemble the final prompt
    # Join the formatted examples with two newlines for separation
    examples_section = "\n\n".join(formatted_examples)
    full_prompt = prompt_header + examples_section + prompt_footer

    # The rest of your function remains the same
    print(f"Calling Gemini with dynamically generated few-shot prompt for query: {query}")
    # The user's query is appended to the full prompt
    # FIX: Added context=context
    output_text, _ = context.qwen_generator.generate(prompt=full_prompt + query, model_name=MISTRAL_VISION_MODEL_NAME)
    output_text = output_text.replace("<mask_tag>", "<mask>")
    prompt_json_refine = """
**Prompt:**

You are a JSON repair assistant. You will be given a possibly malformed or inconsistent JSON object that should match the following structure:

```json
{
  "explanation": "string describing the reasoning or criteria",
  "visual_checks": [
    {
      "object": "string (name of object)",
      "adjective": "string (descriptor like 'empty')",
      "vqa_call": "string (function call of form: vqa(image, 'Is this <mask> (object) adjective?', detected_objects[i]))"
    },
    ...
  ],
  "spatial_instructions": [
    "instruction string",
    ...
  ]
}
```

Your task is to:

1. Correct any JSON syntax errors (e.g., commas, brackets, quoting).
2. Validate and complete all required keys: `explanation`, `visual_checks`, and `spatial_instructions`.
3. Ensure all `vqa_call` strings follow this template:

   ```
   vqa(image, 'Is this <region0> (object) adjective?', detected_objects[i])
   ```

   Replace `object`, `adjective`, and `i` with the corresponding values from the JSON entry.
4. Ensure all <regionX> and <regionX_tag> are replaced with <mask> in the JSON.
5. Ensure the final output is **valid, parsable JSON** with **no comments**, **no explanation text**, and **nothing outside the JSON** — just the corrected JSON object.

Respond with only the corrected JSON, wrapped inside triple backticks with the `json` tag, like this:

```json
{
  ...
}
```

Now, here is the input JSON to fix:

"""
    output_text_refined, _ = context.qwen_generator.generate(prompt=prompt_json_refine + output_text, temperature=0.2)
    print('-Debug-'*20 + f"\n{output_text_refined}\n" + '-Debug-'*20)
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', output_text_refined, re.DOTALL | re.IGNORECASE)
    if match:
        result = json.loads(match.group(1))
        print(result)
        return expand_template_to_instruction(result)
    else:
        print("No JSON found")
        return ""


def api_agent_2(context: VADARContext, predef_signatures, gen_signatures, gen_docstrings):
    if not gen_signatures:
        print("API Agent: No generated signatures to process.")
        return ""
    method_names = []
    for sig in gen_signatures:
        match = re.compile(r"def (\w+)\s*\(.*\):").search(sig)
        method_names.append(match.group(1) if match else f"unknown_method_{len(method_names)}")
    gen_signatures_text = "".join([f"{doc}\n{sig}\n\n" for doc, sig in zip(gen_docstrings, gen_signatures)])
    implementations, error_count = {}, {}
    max_retries_per_signature = 3
    sig_idx = 0
    while sig_idx < len(gen_signatures):
        signature, docstring, current_method_name = gen_signatures[sig_idx], gen_docstrings[sig_idx], method_names[sig_idx]
        console.print(Padding(f"[API Agent] Generating implementation for: {current_method_name}", (1,2), style="on blue"))
        if error_count.get(sig_idx, 0) >= max_retries_per_signature:
            print(f"Skipping implementation generation for '{current_method_name}' after {max_retries_per_signature} retries.")
            implementations[sig_idx] = f"    # Error: Max retries reached for {current_method_name}\n    pass\n"
            sig_idx += 1; continue
        prompt = API_PROMPT.format(predef_signatures=predef_signatures, generated_signatures=gen_signatures_text, docstring=docstring, signature=signature)
        output_text, _ = context.qwen_generator.generate(prompt=prompt, model_name=MISTRAL_CODE_MODEL_NAME)
        if not isinstance(output_text, str) or "Error:" in output_text:
            print(f"API Agent Error for {current_method_name}: {output_text}. Retrying...")
            error_count[sig_idx] = error_count.get(sig_idx, 0) + 1; continue
        implementation_match = re.search(r"<implementation>(.*?)</implementation>", output_text, re.DOTALL)
        if not implementation_match:
            print(f"Warning: No <implementation> tag found for {current_method_name}. Retrying. Output was: {output_text[:200]}...")
            error_count[sig_idx] = error_count.get(sig_idx, 0) + 1; continue
        implementation = implementation_match.group(1).strip()
        lines = implementation.split("\n")
        if lines and lines[0].strip().startswith("def "): implementation = "\n".join(lines[1:])
        implementations[sig_idx] = "\n".join(["    " + line.strip() for line in implementation.split("\n") if line.strip()])
        sig_idx += 1
    api_parts = [f"{gen_docstrings[i]}\n{gen_signatures[i]}\n{implementations.get(i, '    pass # Implementation generation failed')}\n" for i in range(len(gen_signatures))]
    return "\n".join(api_parts).replace("\t", "    ")

def program_agent(context: VADARContext, api, query):
    console.print(Padding(f"[Program Agent] Query: {query}", (1,2), style="on blue"))
    prompt = PROGRAM_PROMPT.format(predef_signatures=MODULES_SIGNATURES, api=api, question=query)
    output_text, _ = context.qwen_generator.generate(prompt=prompt, model_name=MISTRAL_CODE_MODEL_NAME, temperature=0.3)
    if not isinstance(output_text, str) or "Error:" in output_text:
        print(f"Program Agent Error: {output_text}")
        return "final_result = 'Error: Program generation failed'"
    program_match = re.search(r"<program>(.*?)</program>", output_text, re.DOTALL)
    program_code = program_match.group(1).strip() if program_match else output_text.strip()
    return "\n".join(["    " + line for line in program_code.split('\n')]).replace("\t", "    ")

def transform_image(og_image):
    # This function uses T_gd which is from a commented-out import.
    # It will raise a NameError if called. The logic is preserved as requested.
    transform = T_gd.Compose([T_gd.RandomResize([800], max_size=1333), T_gd.ToTensor(), T_gd.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    og_image = og_image.convert("RGB")
    im_t, _ = transform(og_image, None)
    return og_image, im_t

def box_image(img, boxes):
    img1 = img.copy()
    draw = ImageDraw.Draw(img1)
    for box in boxes:
        try:
            x_0, y_0, x_1, y_1 = [int(c) for c in box]
            if x_1 > x_0 and y_1 > y_0: draw.rectangle([x_0, y_0, x_1, y_1], outline="red", width=3)
            else: print(f"Warning: Invalid box coordinates for drawing: {box}")
        except (ValueError, TypeError):
            print(f"Warning: Non-numeric or invalid box coordinates for drawing: {box}")
    return img1

def html_embed_image(img, size=300):
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(depth_to_grayscale(img)) if img.dtype in [np.float32, np.float64] else Image.fromarray(img)
    elif isinstance(img, Image.Image): img_pil = img.copy()
    else: img_pil = Image.new('RGB', (size,size), color='grey')
    img_pil.thumbnail((size, size))
    if img_pil.mode in ["RGBA", "P", "L", "F"]: img_to_save = img_pil.convert("RGB")
    else: img_to_save = img_pil
    with BytesIO() as buffer:
        img_to_save.save(buffer, "jpeg")
        base64_img = base64.b64encode(buffer.getvalue()).decode()
    return f'<img style="vertical-align:middle" src="data:image/jpeg;base64,{base64_img}">'

def depth_to_grayscale(depth_map):
    depth_map_np = np.array(depth_map, dtype=np.float32)
    d_min, d_max = np.min(depth_map_np), np.max(depth_map_np)
    if d_max - d_min < 1e-6: return np.zeros_like(depth_map_np, dtype=np.uint8)
    return ((depth_map_np - d_min) / (d_max - d_min) * 255).astype(np.uint8)

def dotted_image(img, points):
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(depth_to_grayscale(img) if img.ndim == 2 or img.dtype in [np.float32, np.float64] else img).convert("RGB")
    elif isinstance(img, Image.Image): img_pil = img.copy().convert("RGB")
    else: return Image.new('RGB', (100,100), color='grey')
    dot_size = max(1, int(img_pil.size[0] * 0.01))
    draw = ImageDraw.Draw(img_pil)
    for pt in points:
        try:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img_pil.width and 0 <= y < img_pil.height:
                draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), fill="red", outline="black")
        except (ValueError, TypeError, IndexError): print(f"Warning: Invalid point coordinates for dotting: {pt}")
    return img_pil

def _parse_bounding_boxes(boxes, width, height):
    if len(boxes) == 0: return []
    bboxes = []
    for box_tensor in boxes:
        cx, cy, w, h = box_tensor.tolist()
        x1, y1 = (cx - 0.5 * w) * width, (cy - 0.5 * h) * height
        x2, y2 = (cx + 0.5 * w) * width, (cy + 0.5 * h) * height
        bboxes.append([max(0, int(x1)), max(0, int(y1)), min(width -1, int(x2)), min(height -1, int(y2))])
    return bboxes

def load_image(im_pth):
    try: return Image.open(im_pth).convert("RGB")
    except (FileNotFoundError, Exception) as e:
        print(f"Error loading image {im_pth}: {e}")
        return None

def display_predef_api():
    console.print(Syntax(MODULES_SIGNATURES, "python", theme="dracula", line_numbers=False, word_wrap=True))
    return MODULES_SIGNATURES

def display_generated_program(program, api):
    api_lines = len(api.split("\n"))
    console.print(Syntax(program, "python", theme="dracula", line_numbers=True, start_line=api_lines + 2, word_wrap=True))

def display_generated_signatures(generated_signatures, generated_docstrings):
    code_to_display = "".join([f"{docstring}\n{signature}\n\n" for signature, docstring in zip(generated_signatures, generated_docstrings)])
    console.print(Syntax(code_to_display, "python", theme="dracula", line_numbers=False, word_wrap=True))

def display_generated_api(api):
    console.print(Syntax(api, "python", theme="dracula", line_numbers=True, word_wrap=True))

class TimeoutException(Exception): pass
def timeout_handler(signum, frame): raise TimeoutException

def _get_docstring_types_for_pipeline(docstring: str):
    args_pattern = re.compile(r"Args:\s*((?:\s+\w+ \(\w+\): .+\n)+)")
    args_match = args_pattern.search(docstring)
    args_section = args_match.group(1) if args_match else ""
    returns_pattern = re.compile(r"Returns:\s+(\w+): .+")
    returns_match = returns_pattern.search(docstring)
    returns_section = returns_match.group(1) if returns_match else ""
    arg_types = re.findall(r"\s+(\w+) \((\w+)\):", args_section)
    return arg_types, returns_section

def _get_robust_return_code_for_pipeline(docstring, signature, img_width, img_height):
    method_name_match = re.search(r"def\s+([\w_]+)\s*\(", signature)
    if not method_name_match: return "    return None"
    method_name = method_name_match.group(1)
    args_in_signature_match = re.search(r"def\s+[\w_]+\s*\((.*?)\)", signature)
    arg_names = [a.split(':')[0].strip() for a in args_in_signature_match.group(1).split(',') if a.strip()] if args_in_signature_match else []

    if method_name == "is_similar_text": return "    return str(text1).lower() == str(text2).lower()"
    if method_name == "extract_2d_bounding_box": return f"    return detected_object.bounding_box_2d if hasattr(detected_object, 'bounding_box_2d') else np.array([0, 0, min({img_width-1}, 10), min({img_height-1}, 10)])"
    if method_name == "extract_3d_bounding_box": return "    return [tuple(p) for p in np.asarray(detected_object.bounding_box_3d_oriented.get_box_points())] if hasattr(detected_object, 'bounding_box_3d_oriented') else [(0.0,0.0,0.0)]*8"
    if method_name == "retrieve_objects":
        detected_objects_arg_name = arg_names[0] if arg_names else "detected_objects"
        return f"    if str(object_prompt).lower() == 'objects': return list({detected_objects_arg_name})\n    return [{detected_objects_arg_name}[0]] if {detected_objects_arg_name} and hasattr({detected_objects_arg_name}[0], 'class_name') and str(object_prompt).lower() in {detected_objects_arg_name}[0].class_name.lower() else []"
    if method_name == "vqa": return "    if 'color' in str(question).lower(): return 'red'\n    return 'yes' if 'is there' in str(question).lower() or 'are there' in str(question).lower() else 'a mock object'"
    if method_name == "same_object": return "    try:\n        x1_inter, y1_inter = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])\n        x2_inter, y2_inter = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])\n        if x2_inter < x1_inter or y2_inter < y1_inter: return False\n        area1, area2 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1]), (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])\n        area_inter = (x2_inter-x1_inter)*(y2_inter-y1_inter)\n        if area1 > 0 and area_inter / area1 > 0.5: return True\n        if area2 > 0 and area_inter / area2 > 0.5: return True\n        return False\n    except: return False"
    if method_name == "find_overlapping_regions": return "    overlapping = []\n    for idx, region in enumerate(countable_regions):\n        if region.description != parent_region.description and _get_iou(parent_region.bounding_box_2d, region.bounding_box_2d) >= 0.2: overlapping.append(idx)\n    return overlapping"
    if method_name == "get_3D_object_size": return "    return tuple(float(x) for x in detected_object.bounding_box_3d_oriented.extent) if hasattr(detected_object, 'bounding_box_3d_oriented') else (0.1, 0.05, 0.2)"
    _, return_type_str = _get_docstring_types_for_pipeline(docstring)
    if "bool" in return_type_str.lower(): return "    return False"
    if "str" in return_type_str.lower(): return "    return 'stub_string'"
    if "int" in return_type_str.lower(): return "    return 0"
    if "float" in return_type_str.lower(): return "    return 0.0"
    if "list" in return_type_str.lower() or "array" in return_type_str.lower(): return "    return []"
    if "tuple" in return_type_str.lower(): return "    return ()"
    return "    return None"

def _get_docstring_types_for_pipeline(docstring: str):
    args_pattern = re.compile(r"Args:\s*((?:\s*[\w_]+\s*\([\w\s.:|\[\],_]+\)\s*:.+\n)+)", re.IGNORECASE)
    args_match = args_pattern.search(docstring)
    args_section = args_match.group(1) if args_match else ""
    returns_pattern = re.compile(r"Returns:\s*([\w\s.:|\[\],_]+)\s*:.+", re.IGNORECASE)
    returns_match = returns_pattern.search(docstring)
    returns_section = returns_match.group(1).strip() if returns_match else "None"
    arg_entries = re.findall(r"\s*([\w_]+)\s*\(([\w\s.:|\[\],_]+)\)\s*:", args_section)
    def normalize_type(t): return t.strip().replace("array", "np.ndarray").replace("pil.image.image", "PILImage.Image")
    return [(name, normalize_type(type_str)) for name, type_str in arg_entries], normalize_type(returns_section)

def create_mock_detected_object(class_name="mock_item", description="a generic mock detected item", image_width=200, image_height=150, bbox_2d_coords=None, has_point_cloud=True, num_3d_points=100, has_crop=True):
    mask_2d = np.zeros((image_height, image_width), dtype=bool)
    if bbox_2d_coords: x1, y1, x2, y2 = bbox_2d_coords; mask_2d[y1:y2, x1:x2] = True
    else: mask_2d[5:15, 5:15] = True; bbox_2d_coords = np.array([5, 5, 15, 15])
    point_cloud_3d = o3d.geometry.PointCloud()
    if has_point_cloud:
        points = np.random.rand(num_3d_points, 3) * np.array([0.5, 0.5, 0.2])
        point_cloud_3d.points = o3d.utility.Vector3dVector(points)
        point_cloud_3d.colors = o3d.utility.Vector3dVector(np.random.rand(num_3d_points, 3))
    if has_point_cloud and point_cloud_3d.has_points():
        bounding_box_3d_axis_aligned = point_cloud_3d.get_axis_aligned_bounding_box()
        bounding_box_3d_oriented = point_cloud_3d.get_oriented_bounding_box()
    else:
        center, extent = np.array([0.0, 0.0, 0.1]), np.array([0.05, 0.05, 0.05])
        bounding_box_3d_oriented = o3d.geometry.OrientedBoundingBox(center, np.identity(3), extent)
        bounding_box_3d_axis_aligned = o3d.geometry.AxisAlignedBoundingBox(center - extent/2, center + extent/2)
    image_crop_pil = PILImage.new("RGB", (max(1, bbox_2d_coords[2] - bbox_2d_coords[0]), max(1, bbox_2d_coords[3] - bbox_2d_coords[1])), "green") if has_crop and bbox_2d_coords else None
    return DetectedObject(class_name=class_name, description=description, segmentation_mask_2d=mask_2d, rle_mask_2d=f"rle_placeholder_for_shape_{image_height}x{image_width}", bounding_box_2d=np.array(bbox_2d_coords) if bbox_2d_coords else None, point_cloud_3d=point_cloud_3d, bounding_box_3d_oriented=bounding_box_3d_oriented, bounding_box_3d_axis_aligned=bounding_box_3d_axis_aligned, image_crop_pil=image_crop_pil)

def create_mock_image_with_content(width=200, height=150):
    img = PILImage.new("RGB", (width, height), "lightgrey")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 50, 50], fill="red", outline="black")
    draw.ellipse([width - 60, height - 60, width - 10, height - 10], fill="blue", outline="black")
    return img

def _execute_file_for_pipeline(program_executable_path: str, execution_namespace: dict, timeout_seconds: int = 15, inject_globals_dict: dict = None):
    signal.signal(signal.SIGALRM, timeout_handler)
    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = stdout_capture, stderr_capture
        signal.alarm(timeout_seconds)
        script_dir = os.path.dirname(program_executable_path)
        if script_dir not in sys.path: sys.path.insert(0, script_dir)
        runpy.run_path(program_executable_path, init_globals=inject_globals_dict, run_name="__main__")
        signal.alarm(0)
        if script_dir in sys.path and sys.path[0] == script_dir: sys.path.pop(0)
        err_output = stderr_capture.getvalue()
        return (Exception(f"Error during script execution (stderr):\n{err_output}"), err_output) if err_output else (None, None)
    except (TimeoutException, Exception) as e:
        stacktrace = traceback.format_exc() + f"\nStderr: {stderr_capture.getvalue()}"
        return e, stacktrace
    finally:
        signal.alarm(0)
        sys.stdout, sys.stderr = original_stdout, original_stderr

# ==============================================================================
# NEW AND REFACTORED FUNCTIONS FOR ROBUST SUBPROCESS EXECUTION
# ==============================================================================

def _serialize_for_subprocess(data_dict: dict) -> dict:
    """Serializes complex objects into pickle-friendly formats."""
    serialized_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, Image.Image):
            buffer = io.BytesIO()
            value.save(buffer, format='PNG')
            serialized_dict[key] = buffer.getvalue()
        elif isinstance(value, list) and all(isinstance(i, DetectedObject) for i in value):
            serialized_dict[key] = [obj.serialize() for obj in value]
        elif isinstance(value, DetectedObject):
            serialized_dict[key] = value.serialize()
        else:
            serialized_dict[key] = value
    return serialized_dict

def _deserialize_in_subprocess(serialized_data: dict) -> dict:
    """Deserializes data inside the child process."""
    deserialized_dict = {}
    for key, value in serialized_data.items():
        if key.endswith("_bytes"):  # Convention for image bytes
            new_key = key.replace("_bytes", "")
            deserialized_dict[new_key] = Image.open(io.BytesIO(value))
        elif key == "detected_objects" and isinstance(value, list):
            deserialized_dict[key] = [DetectedObject.deserialize(obj_data) for obj_data in value]
        elif key.startswith("mock_detected_object"):
             deserialized_dict[key] = DetectedObject.deserialize(value)
        else:
            deserialized_dict[key] = value
    return deserialized_dict

# ==============================================================================
# CORRECTED AND FINALIZED FUNCTIONS FOR ROBUST API TESTING IN SUBPROCESS
# ==============================================================================

# Serialization/Deserialization helpers remain the same.
# _serialize_for_subprocess, _deserialize_in_subprocess


import textwrap
import re
import ast
from typing import List, Tuple, Optional, Dict

def safe_parse_check(code: str) -> Tuple[bool, str]:
    """
    Safely check if code can be parsed, handling common edge cases.
    """
    if not code.strip():
        return True, ""
    
    try:
        # Try parsing as-is first
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        # Try common fixes before giving up
        
        # Fix 1: Remove trailing/leading empty lines that might cause issues
        cleaned = code.strip()
        if cleaned != code:
            try:
                ast.parse(cleaned)
                return True, "Fixed by removing extra whitespace"
            except SyntaxError:
                pass
        
        # Fix 2: Try wrapping in a function if it looks like function body
        if not re.match(r'^(def|class|import|from)', cleaned):
            try:
                wrapped = f"def temp_function():\n{textwrap.indent(cleaned, '    ')}"
                ast.parse(wrapped)
                return True, "Code appears to be function body"
            except SyntaxError:
                pass
        
        # Fix 3: Try as module-level code with textwrap.dedent
        try:
            dedented = textwrap.dedent(cleaned)
            ast.parse(dedented)
            return True, "Fixed by dedenting"
        except SyntaxError:
            pass
        
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def normalize_indentation_simple(code: str, target_base_indent: int = 4) -> str:
    """
    Simple but robust indentation normalization.
    Uses a heuristic approach that's more forgiving.
    """
    if not code.strip():
        return ' ' * target_base_indent + 'pass'
    
    # Convert tabs to spaces and split into lines
    lines = code.replace('\t', '    ').splitlines()
    
    # Remove completely empty lines at start and end
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    if not lines:
        return ' ' * target_base_indent + 'pass'
    
    # Find the minimum indentation (ignoring empty lines)
    min_indent = float('inf')
    for line in lines:
        if line.strip():  # Only consider non-empty lines
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    
    if min_indent == float('inf'):
        min_indent = 0
    
    # Remove the common indentation and add target base indent
    result_lines = []
    for line in lines:
        if line.strip():
            # Remove common indent and add target indent
            relative_indent = len(line) - len(line.lstrip()) - min_indent
            new_line = ' ' * (target_base_indent + relative_indent) + line.lstrip()
            result_lines.append(new_line)
        else:
            result_lines.append('')  # Keep empty lines as-is
    
    return '\n'.join(result_lines)


def fix_python_blocks(code: str, base_indent: int = 4) -> str:
    """
    Fix Python code blocks by ensuring proper structure.
    More conservative approach that preserves relative indentation.
    """
    if not code.strip():
        return ' ' * base_indent + 'pass'
    
    lines = code.replace('\t', '    ').splitlines()
    
    # Filter out completely empty lines for analysis
    content_lines = [(i, line) for i, line in enumerate(lines) if line.strip()]
    
    if not content_lines:
        return ' ' * base_indent + 'pass'
    
    # Find minimum indentation level
    min_indent = min(len(line) - len(line.lstrip()) for _, line in content_lines)
    
    # Build result preserving relative indentation
    result_lines = []
    for line in lines:
        if line.strip():
            current_indent = len(line) - len(line.lstrip())
            relative_indent = current_indent - min_indent
            new_indent = base_indent + relative_indent
            result_lines.append(' ' * new_indent + line.lstrip())
        else:
            result_lines.append('')
    
    return '\n'.join(result_lines)


def create_minimal_function(signature: str, body: str = None, decorator: str = None) -> str:
    """
    Create a minimal working function, guaranteed to be syntactically valid.
    """
    parts = []
    
    # Add decorator if provided
    if decorator and decorator.strip():
        clean_decorator = decorator.strip()
        if not clean_decorator.startswith('@'):
            clean_decorator = '@' + clean_decorator
        parts.append(clean_decorator)
    
    # Clean up signature
    clean_sig = signature.strip()
    if not clean_sig.startswith('def '):
        clean_sig = 'def ' + clean_sig
    if not clean_sig.endswith(':'):
        clean_sig += ':'
    parts.append(clean_sig)
    
    # Handle body
    if not body or not body.strip():
        parts.append('    pass')
    else:
        try:
            # Try to fix the body
            fixed_body = fix_python_blocks(body, base_indent=4)
            
            # Validate the complete function
            temp_func = '\n'.join(parts + [fixed_body])
            is_valid, _ = safe_parse_check(temp_func)
            
            if is_valid:
                parts.append(fixed_body)
            else:
                # Fall back to safe version
                parts.append('    # Original implementation had syntax issues')
                parts.append('    # ' + body.replace('\n', '\n    # '))
                parts.append('    pass')
        except Exception:
            # Ultimate fallback
            parts.append('    pass')
    
    return '\n'.join(parts)


def robust_function_builder(signature: str, implementation: str, decorator: str = None) -> str:
    """
    Build a function with maximum robustness - will always return valid Python.
    """
    # Start with minimal function approach
    function_code = create_minimal_function(signature, implementation, decorator)
    
    # Validate the result
    is_valid, error_msg = safe_parse_check(function_code)
    
    if is_valid:
        return function_code
    
    # If that failed, create absolute minimal version
    print(f"Function creation failed ({error_msg}), using minimal fallback")
    
    # Extract just the function name for minimal version
    sig_match = re.search(r'def\s+(\w+)', signature)
    func_name = sig_match.group(1) if sig_match else 'generated_function'
    
    minimal_parts = []
    if decorator:
        minimal_parts.append(decorator.strip())
    minimal_parts.extend([
        f'def {func_name}():',
        '    """Generated function with syntax issues in original implementation."""',
        '    pass'
    ])
    
    return '\n'.join(minimal_parts)


def demonstrate_robust_approach():
    """
    Demonstrate the robust approach with problematic inputs.
    """
    test_cases = [
        {
            'name': 'Mixed indentation',
            'signature': 'def process_data(items)',
            'implementation': '''
    matching_objects = retrieve_objects(detected_objects)
        if not matching_objects:
    return None
            for obj in matching_objects:
        process(obj)
''',
            'decorator': '@limited_recursion(10)'
        },
        {
            'name': 'Syntax errors in original',
            'signature': 'def broken_function(x, y)',
            'implementation': '''
    if x > y
        return x
    else
        return y
''',
            'decorator': None
        },
        {
            'name': 'Empty implementation',
            'signature': 'def empty_func()',
            'implementation': '',
            'decorator': '@property'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n=== {test_case['name']} ===")
        print("Input signature:", repr(test_case['signature']))
        print("Input implementation:", repr(test_case['implementation']))
        
        result = robust_function_builder(
            test_case['signature'], 
            test_case['implementation'], 
            test_case['decorator']
        )
        
        print("Generated function:")
        print(result)
        
        # Validate result
        is_valid, error = safe_parse_check(result)
        print(f"Validation: {'✓ VALID' if is_valid else '✗ INVALID: ' + error}")


# Updated version of your original function
def build_function_ultra_safe(signature: str, implementation: str, decorator: str = None) -> str:
    """
    Ultra-safe function builder that never fails.
    Drop-in replacement for your original build_function_safe.
    """
    return robust_function_builder(signature, implementation, decorator)

def _run_api_test_in_subprocess(
    api_info_under_test: dict,
    predef_signatures_text: str, # Kept for potential future use, though not used in mocks
    successfully_implemented_apis: list,
    serialized_mock_data: dict
):
    """
    This function runs in a separate process to safely test a single generated API function.
    It now correctly sets up a mocked environment with tool functions and recursion limits.
    """
    temp_dir = None
    try:
        # 1. Deserialize the mock data needed for the test
        deserialized_data = _deserialize_in_subprocess(serialized_mock_data)

        # 2. Create a mock global namespace that includes mocked tool functions.
        mock_globals = {
            # Standard libraries
            "np": np, "PILImage": PILImage, "o3d": o3d, "re": re,
            "math": math, "ImageDraw": ImageDraw,
            # The class definition itself
            "DetectedObject": DetectedObject,
            # Deserialized mock data
            **deserialized_data,
            # --- Mocked Tool Functions ---
            "vqa": lambda image, question, object: "yes" if "is there" in question.lower() else "red",
            "retrieve_objects": lambda detected_objects, object_prompt: [deserialized_data.get('mock_detected_object_1')] if detected_objects else [],
            "is_similar_text": lambda text1, text2: str(text1).lower() == str(text2).lower(),
            "extract_2d_bounding_box": lambda detected_object: detected_object.bounding_box_2d,
            "extract_3d_bounding_box": lambda detected_object: [tuple(p) for p in np.asarray(detected_object.bounding_box_3d_oriented.get_box_points())],
            "get_3D_object_size": lambda detected_object: tuple(float(x) for x in detected_object.bounding_box_3d_oriented.extent),
            "find_overlapping_regions": lambda parent, countable: [0],
            "calculate_3d_distance": lambda obj1, obj2: 1.5,
        }

        # 3. Build the full script content
        method_name_under_test = api_info_under_test["method_name"]
        script_content = []
        
        # Add imports for readability of the generated code
        script_content.append("import math")
        script_content.append("from typing import Tuple, List, Dict, Optional")
        script_content.append("from PIL import Image as PILImage, ImageDraw")
        script_content.append("import numpy as np")
        script_content.append("import open3d as o3d")
        script_content.append("import io, base64, sys, os, re, tempfile, json, time, torch, PIL")
        script_content.append("from pathlib import Path\n")
        
        # --- CHANGE 1: DEFINE THE DECORATOR AS A STRING ---
        # Define the decorator that will be injected into the script.
        # A max depth of 10 is a reasonable default for testing.
        recursion_decorator_string = """
from functools import wraps

def limited_recursion(max_depth):
  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      # Use a unique attribute name to avoid clashes
      if not hasattr(wrapper, 'depth'):
          wrapper.depth = 0
      if wrapper.depth >= max_depth:
        raise RecursionError(f"Max recursion depth {max_depth} exceeded in {func.__name__}")
      wrapper.depth += 1
      try:
        return func(*args, **kwargs)
      finally:
        wrapper.depth -= 1
    return wrapper
  return decorator
"""

        # --- CHANGE 2: ADD THE DECORATOR DEFINITION TO THE SCRIPT ---
        script_content.append("\n# --- Recursion Limiting Decorator ---")
        script_content.append(recursion_decorator_string)
        script_content.append("# --- End Decorator ---\n")


        # Write DetectedObject Class Definition
        script_content.append("\n# --- DetectedObject Class Definition (copied for test execution context) ---")
        script_content.append(detected_object_class_string)
        script_content.append("# --- End DetectedObject Class Definition ---\n")

        # --- Dependencies ---
        script_content.append("\n# --- Successfully Implemented Generated APIs (Dependencies) ---")
        for api_item in successfully_implemented_apis:
            print(f'HELLOOOO {api_item["method_name"]} <versus> {method_name_under_test}')
            if api_item["method_name"] != method_name_under_test:
                signature = api_item['signature'].strip()
                implementation = api_item['implementation'] or 'pass'

                # Use the safe function builder
                full_function = build_function_ultra_safe(signature, implementation)
                script_content.append(full_function)

        # --- Function under test ---
        signature = api_info_under_test['signature'].strip()
        implementation = api_info_under_test['implementation'] or 'pass'

        function_under_test = build_function_ultra_safe(
            signature, 
            implementation, 
            decorator="@limited_recursion(10)"
        )
        
        script_content.append(f"\n# --- Function under test: {method_name_under_test} ---")
        script_content.append(function_under_test)

        # Create the test call logic
        arg_types, _ = _get_docstring_types_for_pipeline(api_info_under_test["docstring"])
        call_args_str_list = []
        for arg_name, arg_type_str in arg_types:
            norm_type = arg_type_str.lower()
            if "pilimage.image" in norm_type: call_args_str_list.append(f"{arg_name}=image")
            elif "list[detectedobject]" in norm_type: call_args_str_list.append(f"{arg_name}=detected_objects")
            elif "detectedobject" == norm_type: call_args_str_list.append(f"{arg_name}=mock_detected_object_1")
            elif "str" in norm_type: call_args_str_list.append(f"{arg_name}='mock question'")
            else: call_args_str_list.append(f"{arg_name}=None")

        call_string = f"{method_name_under_test}({', '.join(call_args_str_list)})"
        script_content.append(f"\nif __name__ == '__main__':\n    test_result = {call_string}\n    print('Subprocess test completed successfully.')")

        # 4. Write and execute the script
        temp_dir = tempfile.mkdtemp()
        script_path = os.path.join(temp_dir, "test_script.py")
        with open(script_path, "w") as f:
            f.write("\n".join(script_content))

        stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
        original_stdout, original_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = stdout_capture, stderr_capture
            runpy.run_path(script_path, init_globals=mock_globals, run_name="__main__")
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr

        err_output = stderr_capture.getvalue()
        if err_output:
            return {"error": f"Stderr: {err_output}", "stacktrace": err_output}
        
        return {"error": None, "stacktrace": None}

    except Exception as e:
        stacktrace = traceback.format_exc()
        return {"error": f"Outer error in subprocess: {str(e)} at file {script_path}", "stacktrace": stacktrace}
    finally:
        # if temp_dir and os.path.exists(temp_dir):
        #     shutil.rmtree(temp_dir)
        print("Normally would delete tmp folder:", temp_dir)

def _process_wrapper(result_queue, api_info, predef_sigs, successful_apis, mock_data):
    """
    A simple wrapper that calls the target function and puts the result in a queue.
    This is what will be executed in the separate process.
    """
    try:
        result = _run_api_test_in_subprocess(api_info, predef_sigs, successful_apis, mock_data)
        result_queue.put(result)
    except Exception as e:
        # If any unexpected error happens, report it back
        result_queue.put({"error": f"Critical failure in subprocess wrapper: {e}", "stacktrace": traceback.format_exc()})

# `test_individual_api_function` itself is now much cleaner.
# It only prepares data and orchestrates the subprocess.
# Replace the entire existing test_individual_api_function with this new version

def test_individual_api_function(
    api_info_under_test: dict,
    predef_signatures_text: str,
    successfully_implemented_apis: list
):
    """
    Orchestrates the testing of a single API function in a separate, timed-out process.
    This version is thread-safe and avoids nested executors.
    """
    # 1. Prepare mock data for the test (same as before)
    MOCK_IMAGE_WIDTH, MOCK_IMAGE_HEIGHT = 200, 150
    mock_image = create_mock_image_with_content(MOCK_IMAGE_WIDTH, MOCK_IMAGE_HEIGHT)
    mock_do1 = create_mock_detected_object(class_name='red_square_mock', description='a mock red square', image_width=MOCK_IMAGE_WIDTH, image_height=MOCK_IMAGE_HEIGHT, bbox_2d_coords=[10, 10, 50, 50])
    mock_do2 = create_mock_detected_object(class_name='blue_item_mock', description='a mock blue item', image_width=MOCK_IMAGE_WIDTH, image_height=MOCK_IMAGE_HEIGHT, bbox_2d_coords=[70, 70, 120, 120])
    
    mock_data_to_serialize = {
        "image": mock_image,
        "detected_objects": [mock_do1, mock_do2],
        "mock_detected_object_1": mock_do1,
    }
    serialized_data = _serialize_for_subprocess(mock_data_to_serialize)

    # 2. Use multiprocessing.Process for safe, timed execution
    # A multiprocessing.Queue is used to safely pass the result back
    result_queue = multiprocessing.Queue()
    timeout_seconds = 20

    process = multiprocessing.Process(
        target=_process_wrapper,
        args=(
            result_queue,
            api_info_under_test,
            predef_signatures_text,
            successfully_implemented_apis,
            serialized_data
        )
    )

    try:
        process.start()
        process.join(timeout=timeout_seconds) # Wait for the process to finish with a timeout

        if process.is_alive():
            # Process did not finish in time
            process.terminate() # Forcefully stop it
            process.join()      # Wait for termination to complete
            return {"error": f"API test execution timed out after {timeout_seconds} seconds.", "stacktrace": "TimeoutError"}

        # Check if the process exited with an error
        if process.exitcode != 0:
            # Try to get a more specific error from the queue, but have a fallback
            try:
                err_result = result_queue.get_nowait()
                return err_result
            except queue.Empty:
                 return {"error": f"API test process exited with code {process.exitcode}.", "stacktrace": "Unknown error, queue was empty."}

        # Process finished successfully, get the result
        return result_queue.get()

    except Exception as e:
        return {"error": f"Failed to orchestrate API test process: {e}", "stacktrace": traceback.format_exc()}
    finally:
        # Ensure the process is cleaned up if it's still running
        if process.is_alive():
            process.terminate()
            
def fix_indentation(code: str) -> str:
    try:
        # Normalize indentation baseline
        dedented = textwrap.dedent(code)

        # Try Black first (best formatter)
        try:
            return black.format_str(dedented, mode=black.Mode())
        except Exception as e_black:
            print("Black failed:", e_black)

        # Fallback to autopep8
        return autopep8.fix_code(dedented, options={'aggressive': 1})

    except Exception as e:
        print("Failed to fix indentation:", e)
        return code

def remove_non_function_code(code: str) -> str:
  fixed_code = fix_indentation(code)
  try:
    tree = ast.parse(fixed_code)

    # Keep only function (and async function) definitions
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]

    return ast.unparse(tree)  # Python 3.9+
  except Exception as e:
    print("AST parse or unparse failed:", e)
    return code  # or return "" or raise, depending on use case

def api_agent(context: VADARContext, predef_signatures, gen_signatures, gen_docstrings, query):
    if not gen_signatures: return ""
    
    all_method_headers = []
    method_names_ordered = []
    for i, sig in enumerate(gen_signatures):
        match = re.compile(r"def\s+([\w_][\w\d_]*)\s*\((.*?)\)\s*(?:->\s*.*?)?:").search(sig)
        method_name = match.group(1) if match else f"unknown_method_{i}"
        all_method_headers.append({"method_name": method_name, "docstring": gen_docstrings[i], "signature": sig})
        if method_name not in method_names_ordered: method_names_ordered.append(method_name)

    processed_apis, error_counts, llm_messages_history, last_attempted_implementations = [], {n: 0 for n in method_names_ordered}, {n: [] for n in method_names_ordered}, {n: None for n in method_names_ordered}
    MAX_RETRIES, MAX_DEPTH = 3, 5
    
    # ==============================================================================
    # FIX: REMOVED the problematic, thread-unsafe, and unused directory operations.
    # 
    # temp_test_base_dir = "temp_api_test_run"
    # if os.path.exists(temp_test_base_dir): shutil.rmtree(temp_test_base_dir)
    # os.makedirs(temp_test_base_dir, exist_ok=True)
    #
    # The functions called by api_agent already create their own safe, temporary 
    # directories using `tempfile`, so this block was redundant and causing race conditions.
    # ==============================================================================

    processing_queue = list(method_names_ordered)

    def get_header_info(name): return next((h for h in all_method_headers if h["method_name"] == name), None)
    
    idx = 0
    while idx < len(processing_queue):
        current_method_name = processing_queue[idx]
        header_info = get_header_info(current_method_name)
        if not header_info or any(api["method_name"] == current_method_name for api in processed_apis):
            idx += 1
            continue
            
        console.print(Padding(f"[API Agent] Implementing: {current_method_name} (Attempt: {error_counts[current_method_name] + 1})", (1,2), style="on blue"))
        
        console.print()

        if error_counts[current_method_name] >= MAX_RETRIES:
            final_impl = last_attempted_implementations.get(current_method_name) or "    pass # Max retries reached"
            processed_apis.append({"method_name": current_method_name, **header_info, "implementation": fix_indentation(final_impl), "status": "failed_max_retries"})
            idx += 1
            continue

        context_signatures_text = "\n\n".join([h["docstring"]+"\n"+h["signature"] for h in all_method_headers if h["method_name"] != current_method_name])
        current_prompt_text = API_PROMPT.format(predef_signatures=predef_signatures, generated_signatures=context_signatures_text, docstring=header_info["docstring"], signature=header_info["signature"], question=query)
        current_messages = llm_messages_history[current_method_name] or [{"role": "user", "content": current_prompt_text}]
        
        output_text, updated_messages = context.qwen_generator.generate(messages=current_messages, model_name=MISTRAL_FIX_MODEL_NAME, temperature=0.3)
        llm_messages_history[current_method_name] = updated_messages

        if not isinstance(output_text, str) or "Error:" in output_text:
            error_counts[current_method_name] += 1
            llm_messages_history[current_method_name].append({"role": "user", "content": "Generation failed. Please try again."})
            continue
            
        implementation_match = re.search(r"<implementation>(.*?)</implementation>", output_text, re.DOTALL)
        if not implementation_match:
            error_counts[current_method_name] += 1
            llm_messages_history[current_method_name].append({"role": "user", "content": "No <implementation> tag. Provide code in <implementation></implementation>."})
            continue

        raw_impl = implementation_match.group(1).strip()
        raw_impl = remove_non_function_code(raw_impl)  # Clean up the implementation to only include function code
        if not raw_impl:
            error_counts[current_method_name] += 1
            llm_messages_history[current_method_name].append({"role": "user", "content": "Empty implementation. Please provide valid code."})
            continue

        lines = raw_impl.split("\n")
        if lines and lines[0].strip().startswith("def "): raw_impl = "\n".join(lines[1:])
        last_attempted_implementations[current_method_name] = raw_impl
        api_info_for_test = {**header_info, "implementation": raw_impl, "messages": llm_messages_history[current_method_name]}

        test_result = test_individual_api_function(
            api_info_for_test,
            predef_signatures,
            [api for api in processed_apis if "success" in api["status"].lower()] 
        )
        
        if test_result["error"]:
            print(f"Test failed for {current_method_name}: {test_result['error']}")
            was_deferred = False
            undefined_match = re.search(r"name '(\w+)' is not defined", str(test_result["error"]).lower())
            
            if undefined_match:
                dep_name = undefined_match.group(1)
                if dep_name in method_names_ordered and dep_name != current_method_name and not any(api["method_name"] == dep_name and api["status"] == "success" for api in processed_apis):
                    try:
                        current_q_idx = processing_queue.index(current_method_name)
                        if processing_queue.count(dep_name) < MAX_DEPTH:
                            if dep_name in processing_queue: processing_queue.pop(processing_queue.index(dep_name))
                            processing_queue.insert(current_q_idx, dep_name)
                            was_deferred = True
                    except ValueError: pass
            
            if was_deferred:
                llm_messages_history[current_method_name].append({"role": "user", "content": f"Note: Implementation failed due to unresolved dependency '{dep_name}'. I will handle it first."})
            else:
                error_counts[current_method_name] += 1
                llm_messages_history[current_method_name].append({"role": "user", "content": f"Your implementation failed with error:\n{test_result['error']}\n{test_result['stacktrace']}\nPlease provide a corrected implementation."})
            
            continue
        else:
            print(f"Successfully implemented and tested: {current_method_name}")
            processed_apis.append({**header_info, "implementation": fix_indentation(raw_impl), "status": "success"})
            idx += 1

    final_api_parts = []
    for name in method_names_ordered:
        found = next((api for api in processed_apis if api["method_name"] == name), None)
        if found: 
            final_api_parts.append(f"{found['docstring']}\n{found['signature']}\n{found['implementation']}\n")
        else:
            header = get_header_info(name)
            if header: 
                final_api_parts.append(f"{header['docstring']}\n{header['signature']}\n    pass # Implementation failed\n")
    
    return "\n".join(final_api_parts).replace("\t", "    ")

def load_annotations_data():
    actual_json_path = "/workspace/PhysicalAI_Dataset/test/test.json"
    actual_image_dir = "/workspace/PhysicalAI_Dataset/test/images"
    json_file_path = os.environ.get("ANNOTATION_JSON_PATH", actual_json_path)
    try:
        with open(json_file_path, 'r') as f: all_annotations = json.load(f)
        if not isinstance(all_annotations, list) or not all_annotations: raise ValueError("Invalid annotation format.")
        return all_annotations, os.environ.get("IMAGE_BASE_DIR", actual_image_dir), json_file_path
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading annotations from '{json_file_path}': {e}")
        return None, None, None

def display_available_annotations(all_annotations):
    """Displays a formatted table of available annotations for user selection."""
    if not all_annotations:
        console.print("[bold red]No annotations found.[/bold red]")
        return

    table = Table(title="[bold cyan]Available Queries for Inference[/bold cyan]", show_header=True, header_style="bold magenta")
    table.add_column("Index", style="dim", width=6)
    table.add_column("Image ID", style="green")
    table.add_column("Question (start)", justify="left")

    for i, ann in enumerate(all_annotations):
        image_id = extract_image_id(ann.get('image', 'N/A'))
        # Get the 'question' field, default to 'N/A' if not present
        question_text = ann.get('question', 'N/A')
        question_start = question_text[:60] + '...' if len(question_text) > 60 else question_text
        table.add_row(str(i), image_id, question_start)
    
    console.print(table)

def initialize_and_get_generator(context: VADARContext):
    """
    Initializes the GeneralizedSceneGraphGenerator once.
    This function is now only called once by the BatchProcessor.
    """
    # Assuming the config file is in a 'configs' directory relative to the script.
    # Adjust path if necessary.
    CONFIG_DIR, CONFIG_FILE_NAME = "configs", "v2_hf_llm.py"
    CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, CONFIG_FILE_NAME)
    
    if not os.path.exists(CONFIG_FILE_PATH):
        # Create a dummy config if it doesn't exist to avoid crashing
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE_PATH, "w") as f:
            f.write("# Dummy config file\n")
            f.write("model_config = {}\n")
        print(f"Warning: Created dummy config at '{CONFIG_FILE_PATH}'")

    # Wrapper for the text generation function to match expected signature
    def generate_wrapped(messages, max_new_tokens, temperature, do_sample, return_full_text, tokenizer, pipeline, logger):
        return context.qwen_generator.generate(messages=messages, model_name=MISTRAL_CODE_MODEL_NAME)[0]

    return GeneralizedSceneGraphGenerator(
        config_path=CONFIG_FILE_PATH,
        custom_generate_text=generate_wrapped,
        unik3d_model=context.unik3d_model
    )

def prepare_image(image_path):
    try:
        if not os.path.exists(image_path):
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            dummy_img = Image.new('RGB', (640, 480), color=(128, 180, 200))
            draw = ImageDraw.Draw(dummy_img)
            draw.rectangle([100, 100, 200, 200], fill="red", outline="black")
            draw.text((50, 50), "Test Image Qwen (v2)", fill="black")
            dummy_img.save(image_path)
            print(f"Created dummy image: {image_path}")
        return load_image(image_path)
    except Exception as e:
        print(f"Failed to prepare image at {image_path}: {e}")
        return None

def process_annotation_by_index(context: VADARContext, sgg_generator: GeneralizedSceneGraphGenerator, all_annotations: List[Dict], actual_image_dir: str, index: int):
    """
    Performs Stage 1 of the pipeline for a single annotation index.
    This function is now called by the producer thread.
    """
    if not (0 <= index < len(all_annotations)):
        console.print(f"[bold red]Error:[/bold red] Invalid index {index}.")
        return None
        
    selected_annotation = all_annotations[index].copy()
    image_path = str(Path(actual_image_dir) / selected_annotation['image'])
    selected_annotation["image"] = image_path
    
    test_image_pil = load_image(image_path)
    if not test_image_pil:
        console.print(f"[bold red]Error:[/bold red] Could not load image for index {index} at path: {image_path}")
        return None

    try:
        # The generator is passed in, not created here.
        detected_objects_list, refined_query = sgg_generator.process_json_with_llm_classes(selected_annotation)
        console.print(f"[Producer] Stage 1 complete for index {index}. Found {len(detected_objects_list)} objects.")
        
        return {
            'selected_annotation': selected_annotation,
            'test_image_pil': test_image_pil,
            'detected_objects_list': detected_objects_list,
            'refined_query': refined_query,
            'index': index
        }
    except Exception as e:
        console.print(f"[bold red]Error[/bold red] during Stage 1 processing for index {index}: {e}")
        traceback.print_exc()
        return None
        
def load_all_data():
    # Use the new ApiKeyManager and stateless GeneratorGemini
    api_key_manager = ApiKeyManager() 
    gemini_generator = GeneratorGemini(api_key_manager) # Now stateless

    # Pass the stateless generator to initialize_modules
    context = initialize_modules(qwen_api_base_url=MISTRAL_URL
        # The key manager is not part of the context, it lives alongside it
        # The gemini_generator is now the stateless one
        # gemini_generator_instance=gemini_generator 
    )
    if not context: return None

    all_annotations, actual_image_dir, json_file_path = load_annotations_data()
    if all_annotations is None: return None
    
    display_available_annotations(all_annotations)
    
    # Return both the context AND the key manager
    return {
        'context': context, 
        'all_annotations': all_annotations, 
        'actual_image_dir': actual_image_dir, 
        'json_file_path': json_file_path,
        'api_key_manager': api_key_manager # <<< ADD THIS
    }

def extract_image_id(image_path):
    if not image_path or image_path == 'N/A': return 'N/A'
    numeric_chars = re.findall(r'\d', os.path.basename(image_path))
    return ''.join(numeric_chars) or 'N/A'

import re

def extract_answer_from_result(context: VADARContext, query: str, result: str):
    result_lower = result.lower()
    result_lower = re.sub(r'[^a-z0-9\s]', '', result_lower)  # remove special characters

    # 1. Direct keyword detection
    if 'left' in result_lower:
        return 'left'
    if 'right' in result_lower:
        return 'right'

    # 2. Region pattern detection
    region_matches = re.findall(r'region(\d+)', result_lower)
    if region_matches:
        return int(region_matches[0])

    # 3. Use LLM to interpret complex or ambiguous answers
    prompt = (
        f"You are given a query and a machine-generated answer.\n\n"
        f"Query: '{query}'\n"
        f"Answer:\n{result}\n\n"
        "Extract the most relevant response based on the following rules:\n"
        "- If the answer refers to a region like 'region0', return the number (e.g., 0).\n"
        "- If the answer contains a clear number, return that number.\n"
        "- If it clearly says 'left' or 'right', return 'left' or 'right'.\n"
        "- If it clearly says binary answer: 'yes' or 'no', 'true' or 'false', return 'left' or 'right' based on the question in the Query.\n"
        "- If the answer is ambiguous or says 'not found', return None.\n\n"
        "Respond with only one of the following: an integer, 'left', 'right', or None. Do not include any explanation."
    )
    try:
        normalized, _ = context.qwen_generator.generate(
            prompt=prompt, temperature=0.1, model_name=MISTRAL_VISION_MODEL_NAME
        )
        normalized = normalized.strip().lower()
        if 'left' in normalized:
            return 'left'
        if 'right' in normalized:
            return 'right'
        if normalized == 'none':
            return None
        if normalized.isdigit():
            return int(normalized)
        try:
            return float(normalized)
        except ValueError:
            pass
    except Exception as e:
        print(f'LLM extraction failed: {e}')

    # 4. Fallback: extract general numbers if LLM failed or returned unknown format
    numbers = re.findall(r'\d+\.\d+|\d+', result)
    if numbers:
        return [float(n) if '.' in n else int(n) for n in numbers][0]

    return None
    # 1. Check for standard numeric patterns (e.g., "The answer is 5")
    numbers = re.findall(r'\d+\.\d+|\d+', result)
    if numbers:
        return [float(n) if '.' in n else int(n) for n in numbers][0]

    # 2. Check for explicit 'left'/'right'
    result_lower = result.lower()
    if 'left' in result_lower:
        return 'left'
    if 'right' in result_lower:
        return 'right'

    # 3. Check for <regionX> patterns and return first region number found
    region_matches = re.findall(r'region(\d+)', result_lower)
    if region_matches:
        return int(region_matches[0])

    # 4. Fallback to LLM inference for binary or directional answers
    prompt = (
        f"Given a question: '{query}'\n"
        f"And an answer: '{result}'\n\n"
        "Your task is to infer whether the correct directional response should be 'left' or 'right'.\n"
        "- If the answer is binary (i.e., clearly a 'yes' or 'no' answer), use both the question and the answer to determine whether the direction should be 'left' or 'right'.\n"
        "- If the answer is not binary (e.g., contains phrases like 'not found', 'no suitable pallets', 'unknown', etc.), then respond with None.\n\n"
        "Only respond with 'left', 'right', or None. Do not explain or include any other text."
    )
    try:
        normalized, _ = context.qwen_generator.generate(
            prompt=prompt, temperature=0.1, model_name=MISTRAL_VISION_MODEL_NAME
        )
        normalized = normalized.strip().lower()
        if 'left' in normalized:
            return 'left'
        if 'right' in normalized:
            return 'right'
    except Exception as e:
        print(f'An exception occurred: {e}')

    return None

def process_query(context: VADARContext, processed_instance_data):
    if not processed_instance_data: return None
    selected_annotation, test_image_pil, detected_objects_list, refined_query, index = [processed_instance_data.get(k) for k in ['selected_annotation', 'test_image_pil', 'detected_objects_list', 'refined_query', 'index']]
    class_names_list = list(set(det_obj.class_name for det_obj in detected_objects_list))
    image_id = extract_image_id(selected_annotation.get('image', 'N/A'))
    depth_path = selected_annotation['image'].replace('images', 'depths').replace(image_id, f'{image_id}_depth')
    query_expansion_str = query_expansion(context, test_image_pil, remake_query(invert_query(refined_query), '_tag'))
    test_query = refined_query + query_expansion_str + "\nAnswer in an integer, string 'regionX', decimal distance, or 'left'/'right'."
    print(f"\nTest Query: {test_query}")

    predef_api_signatures = display_predef_api()
    # vqa_functions = generate_description_functions(context, test_query, class_names_list)
    html_trace_output, generated_api_code, solution_program_code = None, None, None

    try:
        print("\n--- Running Signature Agent ---")
        # generated_signatures, generated_docstrings = signature_agent(context, predef_api_signatures, test_query, vqa_functions)
        generated_signatures, generated_docstrings = signature_agent(context, predef_api_signatures, test_query)
        if not generated_signatures: raise Exception("Signature agent failed.")
        display_generated_signatures(generated_signatures, generated_docstrings)

        print("\n--- Running API Agent ---")
        generated_api_code = api_agent(context, predef_api_signatures, generated_signatures, generated_docstrings, test_query)
        generated_api_code = enforce_python_code_output(context, generated_api_code)
        if not generated_api_code.strip(): raise Exception("API agent failed.")
        display_generated_api(generated_api_code)

        print("\n--- Running Program Agent ---")
        # solution_program_code = program_agent(context, generated_api_code, test_query, vqa_functions)
        solution_program_code = program_agent(context, generated_api_code, test_query)
        solution_program_code = enforce_python_code_output(context, solution_program_code)
        if not solution_program_code.strip(): raise Exception("Program agent failed.")
        display_generated_program(wrap_solution_code(solution_program_code), generated_api_code)

        print("\n--- Executing Program ---")
        final_result, html_trace_output = execute_program(context, solution_program_code, test_image_pil, depth_path, detected_objects_list, generated_api_code)
        normalized_result = extract_answer_from_result(context, refined_query, str(final_result))
        if normalized_result is None: raise Exception("Answer not found in result.")    
    except Exception as e:
        print(f"Program execution failed: {e}. Falling back to VQA.")
        error_trace_html = f"<p><strong>Program failed:</strong> {e}</p><p><strong>Falling back to VQA.</strong></p>"
        html_trace_output = (html_trace_output or "") + error_trace_html
        try:
            vqa_fallback_question = VQA_PROMPT.format(question=refined_query)
            final_result = _vqa_predict(context, img=test_image_pil, question=vqa_fallback_question)
            html_trace_output += f"<p>VQA Fallback Question: {vqa_fallback_question}</p><p>VQA Fallback Answer: {final_result}</p>"
        except Exception as fallback_e:
            final_result = f"Program failed ({e}), and VQA fallback also failed: {fallback_e}"
            html_trace_output += f"<p><strong>VQA fallback also failed:</strong> {fallback_e}</p>"

    os.makedirs("results", exist_ok=True)
    console.print(f"\nFinal Result: {final_result}")
    console.print(f"Image ID: {image_id}, Index: {index}")
    trace_file_path = f"results/execution_trace_qwen_v2_id_{image_id}_index_{index}.html"
    with open(trace_file_path, "w", encoding="utf-8") as f: f.write(f"<html><body><h1>Trace - ID: {image_id}, Index: {index}</h1>{html_trace_output}</body></html>")
    print(f"HTML trace saved to: {os.path.abspath(trace_file_path)}")
    summary_html = display_result(final_result, test_image_pil, test_query, "Ground truth N/A")
    summary_file_path = f"result_summary_qwen_v2_id_{image_id}_index_{index}.html"
    with open(summary_file_path, "w", encoding="utf-8") as f: f.write(f"<html><body><h1>Summary - ID: {image_id}, Index: {index}</h1>{summary_html}</body></html>")
    print(f"Result summary saved to: {os.path.abspath(summary_file_path)}")

    normalized_result = extract_answer_from_result(context, refined_query, str(final_result))
    return {'id': str(selected_annotation['id']), 'normalized_answer': normalized_result}

def main_processor(loaded_data=None):
    if not loaded_data: print("No loaded data provided to processor."); return None
    result = process_query(loaded_data['context'], loaded_data)
    if result: print("Query processing completed."); return result
    else: print("Query processing failed."); return None

def main_batch_processing(indices_to_process: Optional[List[int]] = None, workers: int = 5):
    """
    Main entry point for running the concurrent batch processor.
    """
    console.print("="*50)
    console.print("[bold cyan]VADAR Batch Processing Mode[/bold cyan]")
    console.print("="*50)

    # If no specific indices are given, process all available annotations
    if indices_to_process is None:
        _, _, json_file_path = load_annotations_data()
        with open(json_file_path, 'r') as f:
            num_annotations = len(json.load(f))
        indices_to_process = list(range(num_annotations))
        console.print(f"No specific indices provided. Processing all {num_annotations} annotations.")

    batch_processor = BatchProcessor(
        indices_to_process=indices_to_process,
        num_consumer_workers=workers
    )
    
    try:
        results = batch_processor.run()
        
        # Save results to a JSON file
        results_file = "batch_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        console.print(f"\n[bold magenta]All batch results saved to '{results_file}'[/bold magenta]")

    except Exception as e:
        console.print("\n[bold red]An unrecoverable error occurred during batch processing:[/bold red]")
        console.print(f"{e}")
        traceback.print_exc()
        batch_processor.stop_event.set() # Attempt to stop all threads
    
    return results

class BatchProcessor:
    """
    Manages the concurrent batch processing of annotations.
    - Stage 1 (Producer): Runs locally intensive model inference sequentially.
    - Stage 2 (Consumers): Runs I/O-bound API calls concurrently.
    """
    def __init__(self, indices_to_process: List[int], num_consumer_workers: int = 5):
        """
        Initializes the batch processor.
        Args:
            indices_to_process (List[int]): A list of annotation indices to process.
            num_consumer_workers (int): The number of concurrent threads for Stage 2.
        """
        self.indices = indices_to_process
        self.num_consumers = num_consumer_workers
        self.total_items = len(indices_to_process)
        
        # Shared resources
        self.context: Optional[VADARContext] = None
        self.sgg_generator: Optional[GeneralizedSceneGraphGenerator] = None
        self.all_annotations: Optional[List[Dict]] = None
        self.image_dir: Optional[str] = None
        
        # Concurrency primitives
        self.stage1_output_queue = queue.Queue(maxsize=self.num_consumers * 2)
        self.results_list = []
        self.producer_thread = None
        self.stop_event = threading.Event()

        self.api_keys = [] # Will be loaded during initialization
        self.api_key_queue = queue.Queue()
        self.key_lock = threading.Lock()
        self.failed_keys = set()



    def _initialize_shared_resources(self):
        """Loads models and data once before starting the workers."""
        console.print("[bold cyan]Initializing shared resources (models, data)...[/bold cyan]")
        
        api_key_manager = ApiKeyManager()
        
        # Initialize context without a specific key; it will be set per-job
        self.context = initialize_modules(api_key_manager=api_key_manager, qwen_api_base_url=MISTRAL_URL) 
        if not self.context:
            raise RuntimeError("Failed to initialize VADAR context.")
        
        self.sgg_generator = initialize_and_get_generator(self.context)
        if not self.sgg_generator:
            raise RuntimeError("Failed to initialize GeneralizedSceneGraphGenerator.")
            
        self.all_annotations, self.image_dir, _ = load_annotations_data()
        if not self.all_annotations:
            raise RuntimeError("Failed to load annotation data.")

        console.print("[bold green]All resources, including ApiKeyManager, initialized successfully.[/bold green]")


    def _producer_task(self, progress, task_id):
        """
        Stage 1: Processes annotations sequentially and puts them in the queue.
        This runs in a single, dedicated thread and updates its own progress bar.
        """
        console.print(f"[Producer] Starting Stage 1 processing for {self.total_items} items.")
        for index in self.indices:
            if self.stop_event.is_set():
                console.print("[Producer] Stop event received. Shutting down.")
                break
            
            processed_data = None
            try:
                processed_data = process_annotation_by_index(
                    self.context, 
                    self.sgg_generator, 
                    self.all_annotations, 
                    self.image_dir, 
                    index
                )
                if processed_data:
                    self.stage1_output_queue.put(processed_data) 
                else:
                    console.print(f"[Producer] [bold yellow]Warning:[/bold yellow] Failed to process index {index} in Stage 1.")
            
            except Exception as e:
                console.print(f"[Producer] [bold red]Error[/bold red] processing index {index}: {e}")
                traceback.print_exc()
            
            finally:
                if processed_data is not None:
                    del processed_data
                
                if self.context and hasattr(self.context.device, 'type') and self.context.device.type == 'cuda':
                    try:
                        torch.cuda.empty_cache()
                    except Exception as cache_e:
                        console.print(f"[Producer] [bold orange]Warning:[/bold orange] Could not empty CUDA cache: {cache_e}")

                progress.update(task_id, advance=1)

        console.print("[Producer] Finished processing all items. Signaling consumers to exit.")
        for _ in range(self.num_consumers):
            self.stage1_output_queue.put(None)

    def _consumer_task(self, progress, task_id):
        """
        Stage 2: Pulls data, safely acquires an API key, sets it on the
        context, runs the job, and cleans up.
        """
        while not self.stop_event.is_set():
            data_for_stage2 = None
            try:
                data_for_stage2 = self.stage1_output_queue.get(timeout=1)
                if data_for_stage2 is None:
                    break # End of queue signal

                index = data_for_stage2.get('index', 'N/A')
                
                # --- CRITICAL FIX ---
                # The 'with api_key_manager.get_key()' block is REMOVED from here.
                # We simply call the processing function.
                console.print(f"[Consumer-{threading.get_ident()}] Starting job for index {index}.")
                
                # The `generate` method within `GeneratorGemini` will handle
                # its own key acquisition and retries.
                result = process_query_stage2(self.context, data_for_stage2)
                
                if result:
                    self.results_list.append(result)

                progress.update(task_id, advance=1)
                console.print(f"[Consumer-{threading.get_ident()}] Finished job for index {index}.")

            except queue.Empty:
                continue

            # This error can now only happen if ALL keys are exhausted during a call.
            except RuntimeError as e:
                if "All Gemini API keys have been exhausted" in str(e):
                    console.print(f"[Consumer-{threading.get_ident()}] [bold red]FATAL:[/bold red] All API keys are exhausted. Stopping worker.")
                    self.stop_event.set() # Signal all other threads to stop
                    break
                else:
                    console.print(f"[Consumer-{threading.get_ident()}] [bold red]CRITICAL ERROR[/bold red] processing index {data_for_stage2.get('index', 'N/A')}: {e}")
                    traceback.print_exc()

            except Exception as e:
                console.print(f"[Consumer-{threading.get_ident()}] [bold red]UNEXPECTED ERROR[/bold red] processing index {data_for_stage2.get('index', 'N/A')}: {e}")
                traceback.print_exc()

            finally:
                if data_for_stage2 is not None:
                    self.stage1_output_queue.task_done()

# In class BatchProcessor:

    def run(self):
        """Starts and manages the entire batch processing pipeline."""
        self._initialize_shared_resources()
        
        with Progress(
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            TextColumn("{task.completed} of {task.total} items"),
            "•",
            TimeElapsedColumn(),
        ) as progress:
            
            stage1_task = progress.add_task("[yellow]Stage 1 (Producing)", total=self.total_items)
            stage2_task = progress.add_task("[cyan]Stage 2 (Consuming)", total=self.total_items)

            self.producer_thread = threading.Thread(target=self._producer_task, args=(progress, stage1_task))
            self.producer_thread.start()

            # The 'with' block is the key. It will handle shutting down the
            # executor and waiting for all threads to finish.
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_consumers, thread_name_prefix='Consumer') as executor:
                # Submit all consumer tasks to the executor.
                # We create a list of futures to potentially check for exceptions later.
                futures = [executor.submit(self._consumer_task, progress, stage2_task) for _ in range(self.num_consumers)]

                # --- THE FIX IS HERE ---
                # REMOVED: concurrent.futures.wait(future_to_worker)
                # The 'with' block's exit will now handle waiting for all tasks to complete.
                
                # (Optional but Recommended) You can check for exceptions as futures complete
                # to make debugging easier if a worker fails.
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()  # This will re-raise any exception that occurred in the thread
                    except Exception as exc:
                        console.print(f'[bold red]A consumer thread generated an exception: {exc}[/bold red]')
                        # You might want to signal other threads to stop here
                        # self.stop_event.set()

            # At this point, the 'with' block has exited, meaning all consumer threads have finished.
            # Now, we just need to wait for the producer thread to finish.
            self.producer_thread.join()

        console.print(f"\n[bold green]Batch processing complete.[/bold green]")
        console.print(f"Successfully processed {len(self.results_list)} out of {self.total_items} items.")
        return self.results_list
        
def process_query_stage2(context: VADARContext, processed_instance_data: Dict):
    """
    Performs Stage 2. It now correctly uses the existing context and executes
    the program directly within the consumer thread, avoiding subprocesses.
    """
    if not processed_instance_data:
        return None

    selected_annotation = processed_instance_data['selected_annotation']
    test_image_pil = processed_instance_data['test_image_pil']
    detected_objects_list = processed_instance_data['detected_objects_list']
    refined_query = processed_instance_data['refined_query']
    index = processed_instance_data['index']
    
    image_path = selected_annotation.get('image', 'N/A')
    image_id = extract_image_id(image_path)
    depth_path = image_path.replace('images', 'depths').replace(image_id, f'{image_id}_depth')

    final_result = None
    html_trace_output = ""
    
    try:
        console.print(f"[Consumer-{threading.get_ident()}] Index {index}: Generating APIs and program...")
        class_names_list = list(set(det_obj.class_name for det_obj in detected_objects_list))
        query_expansion_str = query_expansion(context, test_image_pil, remake_query(invert_query(refined_query), '_tag'))
        test_query = refined_query + query_expansion_str + "\nAnswer in an integer, string 'regionX', decimal distance, or 'left'/'right'."
        
        predef_api_signatures = display_predef_api()
        # vqa_functions = generate_description_functions(context, test_query, class_names_list)
        
        generated_signatures, generated_docstrings = signature_agent(context, predef_api_signatures, test_query)
        if not generated_signatures: raise Exception("Signature agent failed.")
        
        generated_api_code = api_agent(context, predef_api_signatures, generated_signatures, generated_docstrings, test_query)
        generated_api_code = enforce_python_code_output(context, generated_api_code)
        if not generated_api_code.strip(): raise Exception("API agent failed.")
        
        solution_program_code = program_agent(context, generated_api_code, test_query)
        solution_program_code = enforce_python_code_output(context, solution_program_code)
        if not solution_program_code.strip(): raise Exception("Program agent failed.")

        console.print(f"[Consumer-{threading.get_ident()}] Index {index}: Executing program directly...")

        final_result, html_trace_output = execute_program(
            context,
            solution_program_code,
            test_image_pil,
            depth_path,
            detected_objects_list,
            generated_api_code
        )
        normalized_result = extract_answer_from_result(context, refined_query, str(final_result))
        if normalized_result is None: raise Exception("Answer not found in result.")    

    except Exception as e:
        console.print(f"[Consumer-{threading.get_ident()}] Program generation or execution failed for index {index}: {e}. Falling back to VQA.")
        html_trace_output += f"<h2>Fallback to VQA</h2><p><strong>Reason:</strong> {e}</p><pre>{traceback.format_exc()}</pre>"
        try:
            vqa_fallback_question = VQA_PROMPT.format(question=refined_query)
            # FIX: Added context=context
            final_result, _ = context.qwen_generator.generate(prompt=vqa_fallback_question, images=[image_path], model_name=MISTRAL_VISION_MODEL_NAME)
            html_trace_output += f"<p>VQA Fallback Answer: {final_result}</p>"
        except Exception as fallback_e:
            final_result = f"Program failed ({e}), and VQA fallback also failed: {fallback_e}"
            html_trace_output += f"<p><strong>VQA fallback also failed:</strong> {fallback_e}</p>"

    os.makedirs("results", exist_ok=True)
    console.print(f"\nFinal Result: {final_result}")
    console.print(f"Image ID: {image_id}, Index: {index}")
    trace_file_path = f"results/execution_batch_trace_id_{image_id}_index_{index}.html"
    with open(trace_file_path, "w", encoding="utf-8") as f: f.write(f"<html><body><h1>Trace - ID: {image_id}, Index: {index}</h1>{html_trace_output}</body></html>")
    print(f"HTML trace saved to: {os.path.abspath(trace_file_path)}")
    summary_html = display_result(final_result, test_image_pil, test_query, "Ground truth N/A")
    summary_file_path = f"results/result_summary_batch_id_{image_id}_index_{index}.html"
    with open(summary_file_path, "w", encoding="utf-8") as f: f.write(f"<html><body><h1>Summary - ID: {image_id}, Index: {index}</h1>{summary_html}</body></html>")
    print(f"Result summary saved to: {os.path.abspath(summary_file_path)}")


    normalized_result = extract_answer_from_result(context, refined_query, str(final_result))
    
    return {'id': str(selected_annotation['id']), 'normalized_answer': normalized_result, 'index': index}
    
def main_interactive_mode():
    """Main entry point for running the interactive, single-question workflow."""
    console.print("="*60, style="bold blue")
    console.print("[bold cyan]VADAR Interactive Single-Question Mode[/bold cyan]")
    console.print("="*60, style="bold blue")

    # --- Phase 1: Load all models and data ONCE ---
    console.print("\n[Phase 1] Loading all models and data. This may take a moment...")
    try:
        # The initialization now needs to create the ApiKeyManager
        loaded_data = load_all_data() # Ensure load_all_data initializes and returns the manager
        if not loaded_data:
            console.print("[bold red]Failed to load data and models. Exiting.[/bold red]")
            return

        api_key_manager = loaded_data['api_key_manager']
        context = loaded_data['context']

        sgg_generator = initialize_and_get_generator(context)
        if not sgg_generator:
            console.print("[bold red]Failed to initialize Scene Graph Generator. Exiting.[/bold red]")
            return
    except Exception as e:
        console.print("[bold red]An error occurred during initialization:[/bold red]")
        console.print(e)
        traceback.print_exc()
        return

    console.print("[bold green]Models and data loaded successfully.[/bold green]")
    all_annotations = loaded_data['all_annotations']
    
    # --- Phase 2: Interactive Selection Loop ---
    while True:
        console.print("\n" + "="*60)
        console.print("[Phase 2] Select a question to process.")
        display_available_annotations(all_annotations)

        try:
            choice_str = input(f"\n> Enter the index (0 to {len(all_annotations)-1}) to process, or type 'q' to quit: ")
            if choice_str.lower() in ['q', 'quit', 'exit']:
                break

            choice_idx = int(choice_str)
            # --- Stage 1: Scene Graph Generation ---
            console.print("\n--- Running Stage 1: Scene Graph Generation ---")
            processed_instance_data = process_annotation_by_index(
                context, 
                sgg_generator, 
                all_annotations, 
                loaded_data['actual_image_dir'], 
                choice_idx
            )

            if not processed_instance_data:
                console.print(f"[bold red]Failed to complete Stage 1 for index {choice_idx}. Please select another.[/bold red]")
                continue

            # --- The call to Stage 2 is now much simpler ---
            console.print("\n--- Running Stage 2: Agentic Program Generation & Execution ---")

            # No more `with api_key_manager.get_key()` block is needed here.
            # The generator will handle it internally.
            
            # We wrap the call in a try-except to catch the *final* unrecoverable error
            # when ALL keys have been exhausted.
            final_job_result = process_query(context, processed_instance_data)
            
            if final_job_result:
                console.print("\n[bold green on black] Query processing completed for this item. [/bold green on black]")
                console.print(f"Final normalized result: {json.dumps(final_job_result, indent=2)}")
            else:
                console.print("[bold red]Query processing ran but did not produce a valid result.[/bold red]")

            console.print("\nReturning to selection menu...")
            time.sleep(2)

        except RuntimeError as e:
            if "All Gemini API keys have been exhausted" in str(e):
                console.print(f"\n[bold red on white]FATAL ERROR: {e}[/bold red on white]")
                console.print("The application cannot continue. Please add new API keys and restart.")
                break # Exit the main interactive loop.
            else:
                # Handle other unexpected RuntimeErrors
                console.print(f"[bold red]An unexpected runtime error occurred: {e}[/bold red]")
                traceback.print_exc()
        except (ValueError, IndexError):
            console.print("[bold red]Invalid input. Please enter a valid number from the table.[/bold red]\n")
        except Exception as e:
            console.print(f"[bold red]An unexpected and critical error occurred in the main loop: {e}[/bold red]")
            traceback.print_exc()
            console.print("Please try another selection or restart the program.")

def example_usage():
    """Example showing how to use the optimized Generator with your server."""
    
    # Initialize with your server's URL
    generator = Generator(
        base_url="http://localhost:8001",  # Your server URL
        model_name="mistral-large-latest",   # Default model
        temperature=0.7,
        max_new_tokens=2048
    )
    
    # Check server status
    generator.print_server_status()
    
    # Text-only generation (server will use efficient text models)
    print("\n🔤 Text-only generation:")
    response, history = generator.generate(
        prompt="Explain quantum computing in simple terms.",
        temperature=0.3
    )
    print(f"Response: {response[:100]}...")
    
    # Multimodal generation (server will auto-select vision models)
    print("\n📷 Multimodal generation:")
    try:
        response, history = generator.generate(
            prompt="What do you see in this image? Describe it in detail.",
            images=["path/to/your/image.jpg"],  # Replace with actual image path
            messages=history  # Continue the conversation
        )
        print(f"Vision response: {response[:100]}...")
    except Exception as e:
        print(f"Multimodal example failed: {e}")
    
    # Use specific models (server handles OpenAI compatibility)
    print("\n🎯 Specific model usage:")
    response, history = generator.generate(
        prompt="Write a short poem about AI.",
        model_name="gpt-4o",  # Will be mapped to pixtral-large-latest by your server
        temperature=0.8
    )
    print(f"Creative response: {response[:100]}...")
    
    # Final server status
    print("\n📊 Final server status:")
    generator.print_server_status()

if __name__ == "__main__":
    python_code="""
def _identify_empty_transporters(detected_objects):
    empty_transporters = []
    for obj in detected_objects:
        if obj.class_name == 'transporter':
            is_empty = vqa(
                image=image,
                question='Is this object  empty?',
                object=obj)
            if is_similar_text(is_empty, 'True'):
                empty_transporters.append(obj)
    return empty_transporters

def _find_nearest_pallet(transporter, pallets):
    nearest_pallet = None
    min_distance = float('inf')
    for pallet in pallets:
        distance = calculate_3d_distance(transporter, pallet)
        if distance < min_distance:
            min_distance = distance
            nearest_pallet = pallet
    return nearest_pallet

def _determine_optimal_pallet(empty_transporters, pallets):
    if not empty_transporters:
        return None
    optimal_pallet = None
    min_distance = float('inf')
    for transporter in empty_transporters:
        nearest_pallet = _find_nearest_pallet(transporter, pallets)
        if nearest_pallet:
            distance = calculate_3d_distance(transporter, nearest_pallet)
            if distance < min_distance:
                min_distance = distance
                optimal_pallet = nearest_pallet
    return optimal_pallet

empty_transporters = _identify_empty_transporters(detected_objects)
if empty_transporters is None:
    return None
if transporter not in empty_transporters:
    return None
min_distance = float("inf")
nearest_pallet = None
for pallet in pallets:
    distance = calculate_3d_distance(transporter, pallet)
    if distance < min_distance:
        min_distance = distance
        nearest_pallet = pallet
return nearest_pallet

empty_transporters = _identify_empty_transporters(empty_transporters)
if not empty_transporters:
    return None
min_distance = float("inf")
optimal_pallet = None
for transporter in empty_transporters:
    for pallet in pallets:
        distance = calculate_3d_distance(transporter, pallet)
        if distance < min_distance:
            min_distance = distance
            optimal_pallet = pallet
return optimal_pallet
    """
    print(remove_non_function_code(python_code))

    # main_interactive_mode()
        # To run in batch processing mode, comment out main_interactive_mode()
    # and uncomment the main_batch_processing() call below.
    
    # --- Option 1: Interactive Single-Question Mode (Default) ---
    # main_interactive_mode()
    
    # --- Option 2: Batch Processing Mode ---
    # The following code is needed for the batch processor to handle object serialization.
    # print("Starting Batch Processing Mode...")
    # with open(__file__, 'r') as f:
    #     source_code = f.read()
    # do_class_match = re.search(r"class DetectedObject.*?:(?:\n\s+.*)*", source_code, re.DOTALL)
    # if do_class_match:
    #     DetectedObject.__class_code__ = do_class_match.group(0)
    # else:
    #     raise RuntimeError("Could not dynamically find DetectedObject class definition.")
    
    # INDICES_TO_PROCESS = None  # Or specify a list of indices, e.g., [0, 1, 5]
    # NUM_WORKERS = 8
    # main_batch_processing(indices_to_process=INDICES_TO_PROCESS, workers=NUM_WORKERS)
