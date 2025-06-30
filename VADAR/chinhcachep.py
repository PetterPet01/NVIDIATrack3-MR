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

import re
# import groundingdino.datasets.transforms as T_gd
# from groundingdino.util.inference import load_model, predict
from unik3d.models import UniK3D
import torchvision.transforms as TV_T
from transformers import AutoModelForCausalLM, AutoTokenizer
import google.generativeai as genai
import open3d as o3d
import pandas as pd

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

from typing import Union, Dict, List, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Add VADAR root to path
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
from milvus_retrieval import MilvusManager

console = Console(highlight=False, force_terminal=False)

gemini_api_key = 'AIzaSyCJHta9VITkW9ZNnTOuqpvPPIrpSSgCqXg'
gemini_model_name = 'gemini-2.5-flash'


# --- Refactoring Step 1: VADARContext Class Definition ---
# This class will hold all the models and shared resources, eliminating global variables.
class VADARContext:
    def __init__(self, qwen_generator, gemini_generator, unik3d_model, embedding_model, spacy_nlp, device, spatial_rgpt_client):
        self.qwen_generator = qwen_generator
        self.gemini_generator = gemini_generator
        self.unik3d_model = unik3d_model
        self.embedding_model = embedding_model
        self.spacy_nlp = spacy_nlp
        self.device = device
        self.spatial_rgpt_client = spatial_rgpt_client
        # If groundingdino is re-enabled, it would be added here:
        # self.grounding_dino = grounding_dino

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


class Generator:
    def __init__(self, model_name="SalonbusAI/GLM-4-32B-0414-FP8", base_url=None, api_key="dummy", max_new_tokens=1024):
        """
        Initialize the Generator with OpenAI client

        Args:
            model_name: Model name to use
            temperature: Temperature for generation
            base_url: Custom base URL for OpenAI-compatible API
            api_key: API key (can be dummy for local servers)
            max_new_tokens: Maximum tokens to generate
        """
        # self.temperature = temperature
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        # Initialize OpenAI client with custom URL
        if base_url:
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )
            print(f"OpenAI Generator: Initialized with custom base URL: {base_url}")
        else:
            self.client = OpenAI(api_key=api_key)
            print("OpenAI Generator: Initialized with default OpenAI API")

        print(f"OpenAI Generator: Using model: {self.model_name}")

    def remove_substring(self, output, substring):
        """Remove a substring from the output"""
        return output.replace(substring, "") if substring in output else output

    def remove_think_tags(self, text):
        """Remove <think> and </think> tags and their content using regex"""
        pattern = r'<think>.*?</think>'
        return re.sub(pattern, '', text, flags=re.DOTALL).strip()

    def generate(self, prompt=None, messages=None, enable_thinking=False, temperature=0.7):
        """
        Generate text using OpenAI API

        Args:
            prompt: Single prompt string
            messages: List of message dictionaries
            enable_thinking: Whether to enable thinking (handled in prompt injection)

        Returns:
            tuple: (generated_text, conversation_history)
        """
        current_conversation = []

        # Prepare initial message list
        if messages:
            current_conversation = list(messages)
        elif prompt:
            current_conversation.append({"role": "user", "content": prompt})
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        # Inject `/no_think` if required
        # if not enable_thinking:
        #     # Inject into first message if it's user or system
        #     first_msg = current_conversation[0]
        #     if first_msg["role"] in ("user", "system") and "/no_think" not in first_msg["content"]:
        #         first_msg["content"] = "/no_think\n" + first_msg["content"]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=current_conversation,
                temperature=temperature,
                max_tokens=self.max_new_tokens,
                top_p=0.9 if temperature > 0 else None,
            )

            result = response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI Generator: Error during generation: {e}")
            raise

        # Post-process
        print(result)
        if "```python" in result:
            result = self.remove_substring(result, "```python")
            result = self.remove_substring(result, "```")
        result = self.remove_think_tags(result)

        new_conversation_history = list(current_conversation)
        new_conversation_history.append({"role": "assistant", "content": result})

        return result, new_conversation_history


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
            load_in_8bit=load_in_8bit,
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



class GeneratorGemini:
    """
    An OpenAI-compatible client that connects to the Google AI Studio (Gemini) API
    to generate responses from text and image inputs. Automatically rotates API keys
    from a public Google Sheet when one exceeds its usage limit.
    """

    def __init__(self, model_name="gemini-1.5-pro-latest", temperature=0.2,
                 max_new_tokens=4096, rotation_frequency=10,
                 sheet_url="https://docs.google.com/spreadsheets/d/1gqlLToS3OXPA-CvfgXRnZ1A6n32eXMTkXz4ghqZxe2I/gviz/tq?tqx=out:csv&gid=0"):
        self.temperature = temperature
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.rotation_frequency = rotation_frequency
        self.api_key_index = 0
        self.call_count = 0
        self.api_keys = self._load_api_keys_from_sheet(sheet_url)

        if not self.api_keys:
            raise ValueError("No API keys found in the Google Sheet.")
        # Default generation config template
        self.default_generation_config = genai.types.GenerationConfig(
            max_output_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        self._set_api_key(self.api_keys[self.api_key_index])



    def _load_api_keys_from_sheet(self, sheet_url):
        """
        Downloads the Google Sheet as CSV and extracts 'Token' and 'Name' columns.
        """
        print("Gemini Generator: Downloading API keys from Google Sheets...")
        try:
            response = requests.get(sheet_url)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))

            # Fix column names if there are leading/trailing spaces
            df.columns = df.columns.str.strip()

            if 'Token' not in df.columns or 'Name' not in df.columns:
                raise ValueError("CSV must contain 'Token' and 'Name' columns.")

            return df[['Token', 'Name']].rename(columns={'Token': 'token', 'Name': 'name'}).to_dict('records')

        except Exception as e:
            raise RuntimeError(f"Failed to load API keys from Google Sheet: {e}")


    def _set_api_key(self, key_entry):
        genai.configure(api_key=key_entry['token'])
        print(f"Gemini Generator: Using API key ({key_entry['name']})")
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.default_generation_config
        )

    def _load_image(self, image_path: str) -> Image.Image:
        try:
            image = Image.open(image_path).convert('RGB')
            print(f"Gemini Generator: Loaded image '{image_path}' ({image.size[0]}x{image.size[1]})")
            return image
        except FileNotFoundError:
            print(f"Gemini Generator: Error - Image file not found at {image_path}")
            raise
        except Exception as e:
            print(f"Gemini Generator: Error loading image {image_path}: {e}")
            raise

    def _prepare_api_request(self, prompt=None, messages=None, images=None):
        if messages:
            history = messages
            last_message = history[-1]
            api_history = [msg for msg in history[:-1]]
            prompt_text = last_message['content']
        elif prompt:
            api_history = []
            history = [{"role": "user", "content": prompt}]
            prompt_text = prompt
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        formatted_history = []
        for msg in api_history:
            role = "model" if msg["role"] == "assistant" else "user"
            formatted_history.append({"role": role, "parts": [msg["content"]]})

        prompt_content = []
        if prompt_text:
            prompt_content.append(prompt_text)

        if images:
            if isinstance(images, str):
                images = [images]

            for img_path in images:
                try:
                    pil_image = self._load_image(img_path)
                    prompt_content.append(pil_image)
                except FileNotFoundError:
                    error_text = f"[Error: Could not load image at path: {img_path}]"
                    prompt_content.insert(0, error_text)

        return prompt_content, formatted_history, history

    def generate(self, prompt: str = None, messages: list = None, images: list = None, temperature: float = None):
        """
        Generates a response from the Gemini API. Automatically switches API key
        if a key exceeds its quota or rate limit.
        """
        prompt_content, api_history, full_history = self._prepare_api_request(
            prompt=prompt, messages=messages, images=images
        )
        effective_temperature = temperature if temperature is not None else self.temperature

        current_generation_config = genai.types.GenerationConfig(
            max_output_tokens=self.max_new_tokens,
            temperature=effective_temperature,
        )

        # Try available keys in a session
        remaining_keys = list(range(len(self.api_keys)))

        while remaining_keys:
            try:
                print(f"Gemini Generator: Trying API key #{self.api_key_index + 1} ({self.api_keys[self.api_key_index]['name']})")
                chat = self.model.start_chat(history=api_history)
                response = chat.send_message(
                    prompt_content,
                    generation_config=current_generation_config
                )
                response_text = response.text
                full_history.append({"role": "assistant", "content": response_text})
                self.call_count += 1
                return response_text, full_history

            except Exception as e:
                error_str = str(e).lower()
                if "quota" in error_str or "rate limit" in error_str or "exceeded" in error_str:
                    print(f"Gemini Generator: API key #{self.api_key_index + 1} exceeded limit. Trying next key...")
                    remaining_keys.remove(self.api_key_index)
                    if remaining_keys:
                        self.api_key_index = remaining_keys[0]
                        self._set_api_key(self.api_keys[self.api_key_index])
                    else:
                        raise RuntimeError("All API keys exceeded their limits.")
                else:
                    print(f"Gemini Generator: Error - {e}")
                    raise



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
    qwen_model_name="Qwen/Qwen3-8B",
    qwen_max_new_tokens=1536,
    qwen_device_preference=None,
    gemini_api_key=None,
    spatialrgpt_api_base_url="http://localhost:8001",
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

    # Initialize Qwen3 Generator
    print(f"Initializing Qwen3 Generator with model: {qwen_model_name}")
    try:
        qwen_generator = Generator(
            model_name=qwen_model_name,
            max_new_tokens=qwen_max_new_tokens,
            base_url="http://localhost:8000/v1"
        )
        print("Qwen3 Generator initialized successfully.")
    except Exception as e:
        print(f"Error initializing Qwen3 Generator: {e}")
        raise

    # Initialize Gemini API Client Wrapper
    print(f"Initializing Gemini Client for API with model '{gemini_model_name}'")
    try:
        gemini_generator = GeneratorGemini(
            # api_key=gemini_api_key,
            model_name=gemini_model_name
        )
        print("Gemini API Client Wrapper initialized successfully.")
    except Exception as e:
        print(f"Error initializing Gemini API Client Wrapper: {e}")
        traceback.print_exc()
        raise

    # Initialize SpatialRGPT Client
    try:
        spatial_rgpt_client = SpatialRGPTClient(base_url=spatialrgpt_api_base_url)
        print(f"SpatialRGPT Client initialized for server at {spatialrgpt_api_base_url}")
    except Exception as e:
        print(f"Error initializing SpatialRGPT Client: {e}")
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
        spatial_rgpt_client=spatial_rgpt_client
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

def generate(context: VADARContext, prompt: str = None, messages: list = None, enable_thinking=False, temperature=0.2):
    if not context.qwen_generator:
        error_msg = "Error: Qwen3 Generator not available in the provided context."
        print(error_msg)
        return error_msg, messages or []
    try:
        response_text, updated_history = context.qwen_generator.generate(prompt=prompt, messages=messages, enable_thinking=enable_thinking, temperature=temperature)
        return response_text, updated_history
    except Exception as e:
        error_msg = f"Error during qwen_generator.generate() call: {e}"
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
        corrected_output, _ = generate(context, messages=correction_input, enable_thinking=False, temperature=0.2)
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

def depth(context: VADARContext, image, bbox):
    trace_html = []
    if not context.unik3d_model:
        trace_html.append("<p>Error: UniK3D model not available in context for depth().</p>")
        return 0.0, trace_html
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(c, (int, float)) for c in bbox)):
        trace_html.append(f"<p>Error: Invalid bbox format for depth(): {bbox}. Expected [x0,y0,x1,y1].</p>")
        return 0.0, trace_html

    x_mid, y_mid = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    unik3d_device_str = str(context.device)
    with torch.no_grad():
        image_np_rgb = np.array(image.convert("RGB"))
        if image_np_rgb.ndim == 2:
            image_np_rgb = np.stack((image_np_rgb,) * 3, axis=-1)
        image_tensor = (torch.from_numpy(image_np_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0).to(unik3d_device_str)
        outputs = context.unik3d_model.infer(image_tensor, camera=None, normalize=True)
        points_3d_per_pixel = outputs["points"].squeeze(0).permute(1, 2, 0).cpu().numpy()
        depth_map_preds = points_3d_per_pixel[:, :, 2]
        h_preds, w_preds = depth_map_preds.shape
        safe_y_mid = min(max(0, int(y_mid * h_preds / image.height)), h_preds - 1)
        safe_x_mid = min(max(0, int(x_mid * w_preds / image.width)), w_preds - 1)
        depth_val = depth_map_preds[safe_y_mid, safe_x_mid]

    trace_html.append(f"<p>Depth query at original image coordinates: ({x_mid:.1f}, {y_mid:.1f})</p>")
    trace_html.append(f"<p>Corresponding point on predicted depth map (scaled): ({safe_x_mid}, {safe_y_mid})</p>")
    trace_html.append("<p>Point on original image:</p>" + html_embed_image(dotted_image(image, [[x_mid, y_mid]])))
    trace_html.append("<p>Point on predicted depth map (Z from UniK3D):</p>" + html_embed_image(dotted_image(depth_map_preds, [[safe_x_mid, safe_y_mid]])))
    trace_html.append(f"<p>Depth value: {depth_val:.4f}</p>")
    return float(depth_val), trace_html

def _vqa_predict(context: VADARContext, img, question, holistic=False):
    try:
        prompt = VQA_PROMPT.format(question=question)
        full_prompt = f"<image> {prompt}"
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, "temp_image.png")
            img.save(temp_path)
            output, _ = context.gemini_generator.generate(prompt=full_prompt, images=[temp_path])
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
    trace_html = [f"<h4>Find Overlapping Regions</h4><p>Parent region: '<b>{parent_region.description}</b>'</p>", f"<p>Checking against <b>{len(countable_regions)}</b> candidate regions.</p>"]
    padding, overlap_threshold = 20, 0.2
    parent_mask = parent_region.segmentation_mask_2d
    x, y, w, h = parent_region.bounding_box_2d
    x_min, y_min = max(0, int(x - padding)), max(0, int(y - padding))
    x_max, y_max = min(parent_mask.shape[1], int(x + w + padding)), min(parent_mask.shape[0], int(y + h + padding))
    overlapping_regions = []

    for region in countable_regions:
        if region.description == parent_region.description: continue
        crop_parent = parent_mask[y_min:y_max, x_min:x_max]
        crop_region = region.segmentation_mask_2d[y_min:y_max, x_min:x_max]
        overlap_area = np.sum(np.logical_and(crop_parent, crop_region))
        if overlap_area > 0:
            region_area = np.sum(crop_region)
            if region_area > 0 and (overlap_area / region_area) >= overlap_threshold:
                match = re.search(r'\d+', region.description)
                if match: overlapping_regions.append(int(match.group()))

    trace_html.append(f"<p>Result: Found <b>{len(overlapping_regions)}</b> overlapping regions with indices: <b>{overlapping_regions}</b></p>")
    return overlapping_regions, trace_html

def calculate_3d_distance(obj1: DetectedObject, obj2: DetectedObject) -> Tuple[float, List[str]]:
    trace_html = []
    center1, center2 = obj1.bounding_box_3d_oriented.get_center(), obj2.bounding_box_3d_oriented.get_center()
    distance = np.linalg.norm(center1 - center2)
    adjusted_distance = distance + distance * 0.22
    trace_html.append(f"<h4>Calculate 3D Distance</h4>")
    trace_html.append(f"<p>Object 1: '<b>{obj1.description}</b>' (Center: {np.round(center1, 3).tolist()})</p>")
    trace_html.append(f"<p>Object 2: '<b>{obj2.description}</b>' (Center: {np.round(center2, 3).tolist()})</p>")
    trace_html.append(f"<p>Euclidean distance: <b>{distance:.4f}</b> meters</p>")
    trace_html.append(f"<p>Adjusted distance (+22%): <b>{adjusted_distance:.4f}</b> meters</p>")
    return adjusted_distance, trace_html

def get_2D_object_size(image, bbox):
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(c, (int, float)) for c in bbox)):
        return (0, 0), [f"<p>Error: Invalid bbox format for get_2D_object_size(): {bbox}.</p>"]
    width, height = abs(bbox[0] - bbox[2]), abs(bbox[1] - bbox[3])
    trace_html = [f"<p>2D Object Size for Bbox: {bbox}</p>", html_embed_image(box_image(image, [bbox]), 300), f"<p>Width: {width}, Height: {height}</p>"]
    return (width, height), trace_html

PROGRAM_CORRECTION_PROMPT = """
You are an expert Python debugger. The following Python code, designed to run in a special environment with predefined tools, failed during execution.
Your task is to analyze the error traceback and the faulty code, and provide a corrected version of **only the solution program part**.
**Context:**
- The program has access to a pre-defined API.
- The program's goal is to answer a question about an image.
**Available API (for context on what functions can be called):**
{api_code}
**Faulty Program Code:**
{program_code}
**Execution Error and Traceback:**
{traceback}
**Instructions:**
1.  Carefully read the traceback to understand the error (e.g., `NameError`, `TypeError`, `IndexError`).
2.  Examine the provided code to locate the source of the error.
3.  Rewrite the program to fix the bug. Do NOT change the overall logic unless it's necessary to fix the error.
4.  Ensure the corrected code still aims to solve the original problem.
5.  Wrap the corrected and complete Python code block inside `<program></program>` tags.
6.  Do not add any explanations, apologies, or text outside the code block.
**Corrected Program:**
"""

def execute_program(context: VADARContext, program, image, depth, detected_objects, api):
    max_retries = 2
    current_program_code = program
    full_html_trace = []
    header_str = "\n".join(["import math", "from typing import Tuple, List, Dict, Optional", "from PIL import Image as PILImage, ImageDraw", "import numpy as np", "import open3d as o3d", "import io, base64, sys, os, re, tempfile, json, time, torch", "from pathlib import Path", "import PIL", ""]) + '\n'
    api_methods = re.findall(r"def (\w+)\s*\(.*\):", api)

    for attempt in range(max_retries + 1):
        full_html_trace.append(f"<h2>Execution Attempt {attempt + 1}</h2>")
        if attempt > 0:
            full_html_trace.append("<h4>Corrected Program Code:</h4>" + f"<pre style='background-color:#f0f0f0; border:1px solid #ccc; padding:10px; border-radius:5px;'><code>{current_program_code}</code></pre>")

        wrapped_program = wrap_solution_code(current_program_code)
        executable_program = header_str + api + wrapped_program
        program_lines = executable_program.split("\n")
        def get_line(line_no): return program_lines[line_no - 1] if 0 <= line_no - 1 < len(program_lines) else ""
        attempt_trace = []
        def trace_lines(frame, event, arg):
            if event == "line":
                method_name, line_no = frame.f_code.co_name, frame.f_lineno
                if method_name == "solution_program" or method_name in api_methods:
                    line = get_line(line_no).strip()
                    if line: attempt_trace.append(f"<p><code>[{method_name}] Line {line_no}: {line}</code></p>")
            return trace_lines

        namespace = {
            "DetectedObject": DetectedObject, "image": image, "detected_objects": detected_objects, "depth": depth,
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
            print(f"--- Program execution successful on attempt {attempt + 1} ---\n" + '-----LUCKY LUCKY-----'*10 + f"\n{final_result}\n" + '-----LUCKY LUCKY-----'*10)
            return final_result, "".join(full_html_trace)
        except Exception:
            sys.settrace(None)
            traceback_str = traceback.format_exc()
            full_html_trace.append("<h3><span style='color:red;'>Failed</span></h3>" + "".join(attempt_trace) + f"<h4>Error Traceback:</h4><pre style='background-color:#ffebeb; color:red; border:1px solid red; padding:10px; border-radius:5px;'>{traceback_str}</pre>")
            print(f"--- Program execution failed on attempt {attempt + 1} ---\n{traceback_str}")
            if attempt >= max_retries:
                full_html_trace.append("<h3>Max retries reached. Aborting.</h3>")
                return f"Execution failed after {max_retries + 1} attempts. See trace for details.", "".join(full_html_trace)

            print("--- Requesting code correction from Gemini... ---")
            full_html_trace.append("<h4>Requesting Correction from LLM:</h4>")
            correction_prompt = PROGRAM_CORRECTION_PROMPT.format(api_code=api, program_code=current_program_code, traceback=traceback_str)
            try:
                corrected_output, _ = context.gemini_generator.generate(prompt=correction_prompt, temperature=0.2)
                program_match = re.search(r"<program>(.*?)</program>", corrected_output, re.DOTALL)
                if program_match:
                    current_program_code = program_match.group(1).strip()
                    print("--- Received corrected code. Retrying... ---")
                else:
                    print("--- LLM failed to provide a valid correction. Retrying with original code... ---")
                    full_html_trace.append("<p><strong>LLM correction failed (no &lt;program&gt; tag). Retrying with previous code.</strong></p>")
            except Exception as llm_e:
                print(f"--- LLM call for correction failed: {llm_e}. Retrying with original code... ---")
                full_html_trace.append(f"<p><strong>LLM correction call failed: {llm_e}. Retrying with previous code.</strong></p>")

    return "Execution failed after all retries.", "".join(full_html_trace)

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

def signature_agent(context: VADARContext, predef_api, query, vqa_functions):
    prompt = SIGNATURE_PROMPT.format(signatures=predef_api, question=query, vqa_functions=vqa_functions)
    print("Signature Agent Prompt (first 200 chars):\n", prompt[:200] + "...")
    console.print(Padding(f"[Signature Agent] Query: {query}", (1, 2), style="on blue"))
    output_text, _ = generate(context, prompt=prompt, enable_thinking=False, temperature=0.3)
    if not isinstance(output_text, str) or output_text.startswith('Error:'):
        print(f"Signature Agent Error or invalid output: {output_text}")
        return [], []
    docstrings = re.findall(r"<docstring>(.*?)</docstring>", output_text, re.DOTALL)
    signatures = re.findall(r"<signature>(.*?)</signature>", output_text, re.DOTALL)
    return signatures, docstrings

def expand_template_to_instruction(json_obj):
    instructions = []
    if explanation := json_obj.get("explanation"): instructions.append(f"\nClarify the request: {explanation}.")
    for i, check in enumerate(json_obj.get("visual_checks", []), 1):
        obj, adj, vqa_call = check.get("object", "object"), check.get("adjective", "property"), check.get("vqa_call", "vqa(...)")
        instructions.append(f"{i}. Check visually if the '{obj}' is '{adj}' by calling:\n   → {vqa_call}\n   You need to implement this function or route it to your VQA module.")
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
**PURPOSE**: Check ONLY intrinsic object properties visible on the object itself.

**ALLOWED** (object's own state):
- Empty/loaded status: "Is this <mask_tag> (transporter) empty?"
- Physical condition: "Is this <mask_tag> (machine) damaged?"
- Operational state: "Is this <mask_tag> (conveyor) running?"
- Accessibility: "Is this <mask_tag> (path) blocked?"

**FORBIDDEN** (requires external context):
- Spatial relationships: "Is this pallet in the buffer zone?"
- Comparisons: "Is this the closest/largest/best?"
- System states: "Is this transporter available/ready?"
- Distance-based: "Is this near the exit?"

**RULE**: VQA must be answerable by looking only at the cropped object without knowing about other objects, zones, or systems.

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
    manager = MilvusManager(port = 19530, host = 'milvus-standalone', database_mode='nvidia_aic')
    # Assuming the search function returns a list of dicts. We take the top 5.
    few_shot_examples = manager.search_fewshot(query, top_k=5)

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
    output_text, _ = context.gemini_generator.generate(prompt=full_prompt + query)
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
    output_text_refined, _ = generate(context, prompt=prompt_json_refine + output_text, enable_thinking=False, temperature=0.2)
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
        output_text, _ = generate(context, prompt=prompt, enable_thinking=True)
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

def program_agent(context: VADARContext, api, query, vqa_functions):
    console.print(Padding(f"[Program Agent] Query: {query}", (1,2), style="on blue"))
    prompt = PROGRAM_PROMPT.format(predef_signatures=MODULES_SIGNATURES, api=api, question=query, vqa_functions=vqa_functions)
    output_text, _ = generate(context, prompt=prompt, enable_thinking=False, temperature=0.3)
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

def wrap_solution_code(solution_program_code):
  indented_code = "\n".join("    " + line if line.strip() != "" else line for line in solution_program_code.splitlines())
  return f"\ndef solution_program(image, detected_objects):\n{indented_code}\n    return final_result\nfinal_result = solution_program(image, detected_objects)"

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

  def serialize(self):
    return {
      'class_name': self.class_name,
      'description': self.description,
      'segmentation_mask_2d': self.segmentation_mask_2d.tolist() if self.segmentation_mask_2d is not None else None,
      'rle_mask_2d': self.rle_mask_2d,
      'bounding_box_2d': self.bounding_box_2d.tolist() if self.bounding_box_2d is not None else None,
      'point_cloud_3d': np.asarray(self.point_cloud_3d.points).tolist() if self.point_cloud_3d and self.point_cloud_3d.has_points() else [],
      'point_cloud_colors': np.asarray(self.point_cloud_3d.colors).tolist() if self.point_cloud_3d and self.point_cloud_3d.has_colors() else [],
      'bounding_box_3d_oriented': {
        'center': self.bounding_box_3d_oriented.center.tolist(),
        'extent': self.bounding_box_3d_oriented.extent.tolist(),
        'rotation': self.bounding_box_3d_oriented.R.tolist(),
      } if self.bounding_box_3d_oriented and hasattr(self.bounding_box_3d_oriented, 'center') else None, # Added hasattr check
      'bounding_box_3d_axis_aligned': {
        'min_bound': self.bounding_box_3d_axis_aligned.get_min_bound().tolist(),
        'max_bound': self.bounding_box_3d_axis_aligned.get_max_bound().tolist(),
      } if self.bounding_box_3d_axis_aligned and hasattr(self.bounding_box_3d_axis_aligned, 'get_min_bound') else None, # Added hasattr check
      'image_crop_pil': DetectedObject._pil_to_bytes(self.image_crop_pil) if self.image_crop_pil else None
    }

  @staticmethod
  def _pil_to_bytes(image: PILImage.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

  @staticmethod
  def _bytes_to_pil(b64_string: str) -> PILImage.Image:
    return PILImage.open(io.BytesIO(base64.b64decode(b64_string)))

  @classmethod
  def deserialize(cls, data):
    pc = o3d.geometry.PointCloud()
    if data.get('point_cloud_3d'):
        pc.points = o3d.utility.Vector3dVector(np.array(data['point_cloud_3d']))
    if data.get('point_cloud_colors'):
        pc.colors = o3d.utility.Vector3dVector(np.array(data['point_cloud_colors']))
    obb = o3d.geometry.OrientedBoundingBox() # Default empty
    obb_data = data.get('bounding_box_3d_oriented')
    if obb_data:
        try:
            obb = o3d.geometry.OrientedBoundingBox(
              center=np.array(obb_data['center']),
              R=np.array(obb_data['rotation']),
              extent=np.array(obb_data['extent']))
        except Exception: pass # Keep default empty obb
    aabb = o3d.geometry.AxisAlignedBoundingBox() # Default empty
    aabb_data = data.get('bounding_box_3d_axis_aligned')
    if aabb_data:
        try:
            aabb = o3d.geometry.AxisAlignedBoundingBox(
              min_bound=np.array(aabb_data['min_bound']),
              max_bound=np.array(aabb_data['max_bound']))
        except Exception: pass # Keep default empty aabb
    image_crop = cls._bytes_to_pil(data['image_crop_pil']) if data.get('image_crop_pil') else None
    bounding_box_2d_data = data.get('bounding_box_2d')
    bounding_box_2d_np = np.array(bounding_box_2d_data) if bounding_box_2d_data is not None else None
    segmentation_mask_2d_data = data.get('segmentation_mask_2d')
    segmentation_mask_2d_np = np.array(segmentation_mask_2d_data, dtype=bool) if segmentation_mask_2d_data is not None else None
    return cls(
      class_name=data['class_name'], description=data['description'],
      segmentation_mask_2d=segmentation_mask_2d_np, rle_mask_2d=data['rle_mask_2d'],
      bounding_box_2d=bounding_box_2d_np, point_cloud_3d=pc,
      bounding_box_3d_oriented=obb, bounding_box_3d_axis_aligned=aabb,
      image_crop_pil=image_crop)

  def __repr__(self):
      num_points = len(self.point_cloud_3d.points) if self.point_cloud_3d and self.point_cloud_3d.has_points() else 0
      mask_sum_repr = self.segmentation_mask_2d.sum() if self.segmentation_mask_2d is not None else 'None'
      mask_shape_repr = self.segmentation_mask_2d.shape if self.segmentation_mask_2d is not None else 'N/A'
      obb_center_repr = self.bounding_box_3d_oriented.center.tolist() if self.bounding_box_3d_oriented and hasattr(self.bounding_box_3d_oriented, 'center') else 'N/A'
      bb2d_repr = self.bounding_box_2d.tolist() if self.bounding_box_2d is not None else 'N/A'
      return (f"<DetectedObject: {self.class_name} "
              f"(Desc: '{self.description[:30]}...'), "
              f"2D_bbox: {bb2d_repr}, "
              f"Mask_Sum: {mask_sum_repr} (Shape: {mask_shape_repr}), "
              f"3D_pts: {num_points}, "
              f"3D_OBB_center: {obb_center_repr}>")
"""


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

def _run_api_test_in_subprocess(
    api_info_under_test: dict,
    predef_signatures_text: str, # Kept for potential future use, though not used in mocks
    successfully_implemented_apis: list,
    serialized_mock_data: dict
):
    """
    This function runs in a separate process to safely test a single generated API function.
    It now correctly sets up a mocked environment with tool functions.
    """
    temp_dir = None
    try:
        # 1. Deserialize the mock data needed for the test
        deserialized_data = _deserialize_in_subprocess(serialized_mock_data)

        # 2. *** THIS IS THE CRITICAL FIX ***
        #    Create a mock global namespace that includes mocked tool functions.
        mock_globals = {
            # Standard libraries
            "np": np, "PILImage": PILImage, "o3d": o3d, "re": re,
            "math": math, "ImageDraw": ImageDraw,
            # The class definition itself
            "DetectedObject": DetectedObject,
            # Deserialized mock data
            **deserialized_data,
            # --- Mocked Tool Functions ---
            # These return hardcoded, predictable values for testing.
            "vqa": lambda image, question, object: "yes" if "is there" in question.lower() else "red",
            "retrieve_objects": lambda detected_objects, object_prompt: [deserialized_data['mock_detected_object_1']] if detected_objects else [],
            "is_similar_text": lambda text1, text2: str(text1).lower() == str(text2).lower(),
            "extract_2d_bounding_box": lambda detected_object: detected_object.bounding_box_2d,
            "extract_3d_bounding_box": lambda detected_object: [tuple(p) for p in np.asarray(detected_object.bounding_box_3d_oriented.get_box_points())],
            "get_3D_object_size": lambda detected_object: tuple(float(x) for x in detected_object.bounding_box_3d_oriented.extent),
            "find_overlapping_regions": lambda parent, countable: [0],
            "calculate_3d_distance": lambda obj1, obj2: 1.5,
        }

        # 3. Build the full script content
        method_name_under_test = api_info_under_test["method_name"]
        script_content = [
            # Include successfully implemented dependencies
            "# --- Successfully Implemented Generated APIs (Dependencies) ---",
        ]
        for api_item in successfully_implemented_apis:
            if api_item["method_name"] != method_name_under_test:
                # The implementation of dependencies needs to be present
                script_content.append(f"{api_item['signature']}\n{api_item['implementation']}\n")

        # Include the function under test
        script_content.append(f"\n# --- Function under test: {method_name_under_test} ---")
        script_content.append(api_info_under_test['signature'])
        script_content.append(normalize_indentation(api_info_under_test['implementation']) or '    pass')

        # Create the test call logic using the original argument mapping logic
        arg_types, _ = _get_docstring_types_for_pipeline(api_info_under_test["docstring"])
        call_args_str_list = []
        for arg_name, arg_type_str in arg_types:
            norm_type = arg_type_str.lower()
            if "pilimage.image" in norm_type: call_args_str_list.append(f"{arg_name}=image")
            elif "list[detectedobject]" in norm_type: call_args_str_list.append(f"{arg_name}=detected_objects")
            elif "detectedobject" == norm_type: call_args_str_list.append(f"{arg_name}=mock_detected_object_1")
            elif "str" in norm_type: call_args_str_list.append(f"{arg_name}='mock question'")
            else: call_args_str_list.append(f"{arg_name}=None") # Safe default
        
        call_string = f"{method_name_under_test}({', '.join(call_args_str_list)})"
        script_content.append(f"\nif __name__ == '__main__':\n    test_result = {call_string}\n    # A simple print indicates success to the parent process\n    print('Subprocess test completed successfully.')")

        # 4. Write and execute the script
        temp_dir = tempfile.mkdtemp()
        script_path = os.path.join(temp_dir, "test_script.py")
        with open(script_path, "w") as f:
            # We don't need to write standard imports because they are in mock_globals
            # but we do need the implementations of other functions.
            f.write("\n".join(script_content))

        # Execute using runpy, injecting our complete mock environment
        stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
        original_stdout, original_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = stdout_capture, stderr_capture
            runpy.run_path(script_path, init_globals=mock_globals, run_name="__main__")
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr

        err_output = stderr_capture.getvalue()
        if err_output:
            # If there's any output on stderr, it's an error.
            return {"error": f"Stderr: {err_output}", "stacktrace": err_output}
        
        # If we reach here, the script ran without Python errors.
        return {"error": None, "stacktrace": None}

    except Exception as e:
        stacktrace = traceback.format_exc()
        return {"error": f"Outer error in subprocess: {str(e)}", "stacktrace": stacktrace}
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# `test_individual_api_function` itself is now much cleaner.
# It only prepares data and orchestrates the subprocess.
def test_individual_api_function(
    api_info_under_test: dict,
    predef_signatures_text: str,
    successfully_implemented_apis: list
):
    """
    Orchestrates the testing of a single API function in a separate process.
    This function is thread-safe and now correctly prepares the mock environment.
    """
    # 1. Create mock data for the test
    MOCK_IMAGE_WIDTH, MOCK_IMAGE_HEIGHT = 200, 150
    mock_image = create_mock_image_with_content(MOCK_IMAGE_WIDTH, MOCK_IMAGE_HEIGHT)
    mock_do1 = create_mock_detected_object(class_name='red_square_mock', description='a mock red square', image_width=MOCK_IMAGE_WIDTH, image_height=MOCK_IMAGE_HEIGHT, bbox_2d_coords=[10, 10, 50, 50])
    mock_do2 = create_mock_detected_object(class_name='blue_item_mock', description='a mock blue item', image_width=MOCK_IMAGE_WIDTH, image_height=MOCK_IMAGE_HEIGHT, bbox_2d_coords=[70, 70, 120, 120])
    
    # This dictionary is what gets passed to the subprocess after serialization.
    # Its keys will become global variables in the subprocess.
    mock_data_to_serialize = {
        "image": mock_image,
        "detected_objects": [mock_do1, mock_do2],
        "mock_detected_object_1": mock_do1,
    }
    serialized_data = _serialize_for_subprocess(mock_data_to_serialize)

    # 2. Execute the test in a subprocess with a timeout
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _run_api_test_in_subprocess,
            api_info_under_test,
            predef_signatures_text,
            successfully_implemented_apis,
            serialized_data
        )
        try:
            result = future.result(timeout=60) # 60-second timeout
            return result
        except concurrent.futures.TimeoutError:
            return {"error": "API test execution timed out after 60 seconds.", "stacktrace": "TimeoutError"}
        except Exception as e:
            return {"error": f"Failed to execute API test subprocess: {e}", "stacktrace": traceback.format_exc()}

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
    temp_test_base_dir = "temp_api_test_run"
    if os.path.exists(temp_test_base_dir): shutil.rmtree(temp_test_base_dir)
    os.makedirs(temp_test_base_dir, exist_ok=True)
    processing_queue = list(method_names_ordered)

    def get_header_info(name): return next((h for h in all_method_headers if h["method_name"] == name), None)
    idx = 0
    while idx < len(processing_queue):
        current_method_name = processing_queue[idx]
        header_info = get_header_info(current_method_name)
        if not header_info or any(api["method_name"] == current_method_name for api in processed_apis):
            idx += 1; continue
        console.print(Padding(f"[API Agent] Implementing: {current_method_name} (Attempt: {error_counts[current_method_name] + 1})", (1,2), style="on blue"))
        if error_counts[current_method_name] >= MAX_RETRIES:
            final_impl = last_attempted_implementations.get(current_method_name) or "    pass # Max retries reached"
            processed_apis.append({"method_name": current_method_name, **header_info, "implementation": normalize_indentation(final_impl), "status": "failed_max_retries"})
            idx += 1; continue

        context_signatures_text = "\n\n".join([h["docstring"]+"\n"+h["signature"] for h in all_method_headers if h["method_name"] != current_method_name])
        current_prompt_text = API_PROMPT.format(predef_signatures=predef_signatures, generated_signatures=context_signatures_text, docstring=header_info["docstring"], signature=header_info["signature"], question=query)
        current_messages = llm_messages_history[current_method_name] or [{"role": "user", "content": current_prompt_text}]
        output_text, updated_messages = generate(context, messages=current_messages, enable_thinking=False, temperature=0.3)
        llm_messages_history[current_method_name] = updated_messages

        if not isinstance(output_text, str) or "Error:" in output_text:
            error_counts[current_method_name] += 1
            llm_messages_history[current_method_name].append({"role": "user", "content": "Generation failed. Please try again."}); continue
        implementation_match = re.search(r"<implementation>(.*?)</implementation>", output_text, re.DOTALL)
        if not implementation_match:
            error_counts[current_method_name] += 1
            llm_messages_history[current_method_name].append({"role": "user", "content": "No <implementation> tag. Provide code in <implementation></implementation>."}); continue

        raw_impl = implementation_match.group(1).strip()
        lines = raw_impl.split("\n")
        if lines and lines[0].strip().startswith("def "): raw_impl = "\n".join(lines[1:])
        last_attempted_implementations[current_method_name] = raw_impl
        api_info_for_test = {**header_info, "implementation": raw_impl, "messages": llm_messages_history[current_method_name]}

        test_result = test_individual_api_function(
            api_info_for_test,
            predef_signatures,
            [api for api in processed_apis if api["status"] == "success"]
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
            processed_apis.append({**header_info, "implementation": normalize_indentation(raw_impl), "status": "success"})
            idx += 1

    final_api_parts = []
    for name in method_names_ordered:
        found = next((api for api in processed_apis if api["method_name"] == name), None)
        if found: final_api_parts.append(f"{found['docstring']}\n{found['signature']}\n{found['implementation']}\n")
        else:
            header = get_header_info(name)
            if header: final_api_parts.append(f"{header['docstring']}\n{header['signature']}\n    pass # Implementation failed\n")
    return "\n".join(final_api_parts).replace("\t", "    ")

def load_annotations_data():
    actual_json_path = "/root/Projects/NVIDIAFinalRun/PhysicalAI_Dataset/val/val.json"
    actual_image_dir = "/root/Projects/NVIDIAFinalRun/PhysicalAI_Dataset/val/images"
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
        return generate(context, messages=messages)[0]

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
    context = initialize_modules(gemini_api_key=None)
    if not context: return None
    all_annotations, actual_image_dir, json_file_path = load_annotations_data()
    if all_annotations is None: return None
    display_available_annotations(all_annotations)
    return {'context': context, 'all_annotations': all_annotations, 'actual_image_dir': actual_image_dir, 'json_file_path': json_file_path}

def extract_image_id(image_path):
    if not image_path or image_path == 'N/A': return 'N/A'
    numeric_chars = re.findall(r'\d', os.path.basename(image_path))
    return ''.join(numeric_chars) or 'N/A'

def extract_answer_from_result(context: VADARContext, query: str, result: str):
    numbers = re.findall(r'\d+\.\d+|\d+', result)
    if numbers: return [float(n) if '.' in n else int(n) for n in numbers][0]
    result_lower = result.lower()
    if 'left' in result_lower: return 'left'
    if 'right' in result_lower: return 'right'
    # if 'yes' in result_lower or 'no' in result_lower:
    prompt = (
        f"Given a question: '{query}'\n"
        f"And an answer: '{result}'\n\n"
        "Your task is to infer whether the correct directional response should be 'left' or 'right'.\n"
        "- If the answer is binary (i.e., clearly a 'yes' or 'no' answer), use both the question and the answer to determine whether the direction should be 'left' or 'right'.\n"
        "- If the answer is not binary (e.g., contains phrases like 'not found', 'no suitable pallets', 'unknown', etc.), then respond with None.\n\n"
        "Only respond with 'left', 'right', or None. Do not explain or include any other text."
    )
    try:
        normalized, _ = context.gemini_generator.generate(prompt=prompt, temperature=0.1)
        # normalized, _ = generate(context, prompt=prompt, enable_thinking=False, temperature=0.1)
        normalized = normalized.strip().lower()
        if 'left' in normalized: return 'left'
        if 'right' in normalized: return 'right'
    except Exception as e: print(f'An exception occurred: {e}')
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
    vqa_functions = generate_description_functions(context, test_query, class_names_list)
    html_trace_output, generated_api_code, solution_program_code = None, None, None

    try:
        print("\n--- Running Signature Agent ---")
        generated_signatures, generated_docstrings = signature_agent(context, predef_api_signatures, test_query, vqa_functions)
        if not generated_signatures: raise Exception("Signature agent failed.")
        display_generated_signatures(generated_signatures, generated_docstrings)

        print("\n--- Running API Agent ---")
        generated_api_code = api_agent(context, predef_api_signatures, generated_signatures, generated_docstrings, test_query)
        generated_api_code = enforce_python_code_output(context, generated_api_code)
        if not generated_api_code.strip(): raise Exception("API agent failed.")
        display_generated_api(generated_api_code)

        print("\n--- Running Program Agent ---")
        solution_program_code = program_agent(context, generated_api_code, test_query, vqa_functions)
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

    trace_file_path = f"execution_trace_qwen_v2_id_{image_id}_index_{index}.html"
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

    def _initialize_shared_resources(self):
        """Loads models and data once before starting the workers."""
        console.print("[bold cyan]Initializing shared resources (models, data)...[/bold cyan]")
        self.context = initialize_modules(gemini_api_key=gemini_api_key)
        if not self.context:
            raise RuntimeError("Failed to initialize VADAR context.")
        
        # Initialize the Scene Graph Generator once
        self.sgg_generator = initialize_and_get_generator(self.context)
        if not self.sgg_generator:
            raise RuntimeError("Failed to initialize GeneralizedSceneGraphGenerator.")
            
        # Load annotation data
        self.all_annotations, self.image_dir, _ = load_annotations_data()
        if not self.all_annotations:
            raise RuntimeError("Failed to load annotation data.")
        
        console.print("[bold green]All resources initialized successfully.[/bold green]")

    def _producer_task(self):
        """
        Stage 1: Processes annotations sequentially and puts them in the queue.
        This runs in a single, dedicated thread.
        """
        console.print(f"[Producer] Starting Stage 1 processing for {self.total_items} items.")
        for index in self.indices:
            if self.stop_event.is_set():
                console.print("[Producer] Stop event received. Shutting down.")
                break
            try:
                processed_data = process_annotation_by_index(
                    self.context, 
                    self.sgg_generator, # Pass the pre-loaded generator
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

        # Signal consumers to stop by putting None in the queue
        console.print("[Producer] Finished processing all items. Signaling consumers to exit.")
        for _ in range(self.num_consumers):
            self.stage1_output_queue.put(None)

    def _consumer_task(self, progress, task_id):
        """
        Stage 2: Pulls data from the queue and runs the API-heavy part of the pipeline.
        This runs in multiple concurrent threads.
        """
        while not self.stop_event.is_set():
            try:
                # Wait for an item from the producer
                data_for_stage2 = self.stage1_output_queue.get(timeout=1)

                if data_for_stage2 is None:
                    # Sentinel value received, exit the loop
                    break

                index = data_for_stage2.get('index', 'N/A')
                console.print(f"[Consumer-{threading.get_ident()}] Picked up item [bold blue]index {index}[/bold blue]. Starting Stage 2.")
                
                # Execute the second part of the pipeline
                result = process_query_stage2(self.context, data_for_stage2)
                
                if result:
                    self.results_list.append(result)
                    console.print(f"[Consumer-{threading.get_ident()}] [bold green]Successfully[/bold green] processed item [bold blue]index {index}[/bold blue].")
                else:
                    console.print(f"[Consumer-{threading.get_ident()}] [bold yellow]Failed[/bold yellow] to get result for item [bold blue]index {index}[/bold blue] in Stage 2.")
                
                progress.update(task_id, advance=1)
                self.stage1_output_queue.task_done()

            except queue.Empty:
                # Queue is empty, just continue waiting
                continue
            except Exception as e:
                console.print(f"[Consumer-{threading.get_ident()}] [bold red]CRITICAL ERROR[/bold red] during processing: {e}")
                traceback.print_exc()

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
            
            p_task = progress.add_task("[green]Processing Batch", total=self.total_items)

            # Start the producer in a separate thread
            self.producer_thread = threading.Thread(target=self._producer_task)
            self.producer_thread.start()

            # Start the consumer threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_consumers, thread_name_prefix='Consumer') as executor:
                # Submit consumer tasks
                future_to_worker = {executor.submit(self._consumer_task, progress, p_task): i for i in range(self.num_consumers)}

                # Wait for all futures to complete
                concurrent.futures.wait(future_to_worker)

            # Ensure producer thread is finished
            self.producer_thread.join()

        console.print(f"\n[bold green]Batch processing complete.[/bold green]")
        console.print(f"Successfully processed {len(self.results_list)} out of {self.total_items} items.")
        return self.results_list

# ==============================================================================
# NEW AND MODIFIED FUNCTIONS FOR SUBPROCESS EXECUTION
# ==============================================================================

def run_program_in_subprocess(
    program_code: str,
    api_code: str,
    serialized_data: Dict,
    gemini_api_key: str, # Pass configs directly
    qwen_base_url_for_subprocess: str
):
    """
    This function runs in a separate process. It initializes its own limited
    context, deserializes data, and executes the generated program.
    """
    try:
        # 1. Initialize a new, lightweight context within this process
        # This is crucial for making tools available to the executed code.
        # It avoids pickling large models.
        context_in_subprocess = initialize_modules(
            gemini_api_key=gemini_api_key,
            # qwen_model_name="Qwen/Qwen1.5-7B-Chat", # A smaller model for API calls
            # qwen_device_preference="cpu",
            # other_models_device_preference="cpu" # Avoid loading heavy models on GPU here
        )

        # 2. Deserialize the data
        deserialized_inputs = _deserialize_in_subprocess(serialized_data)
        image = deserialized_inputs['image']
        depth = deserialized_inputs['depth_path']
        detected_objects = deserialized_inputs['detected_objects']

        # 3. Execute the program using the original, robust `execute_program` logic
        final_result, html_trace = execute_program(
            context_in_subprocess, program_code, image, depth, detected_objects, api_code
        )
        return final_result, html_trace
    
    except Exception as e:
        error_message = f"Error inside subprocess: {str(e)}\n{traceback.format_exc()}"
        return error_message, f"<pre>{error_message}</pre>"


# THE CORRECTED VERSION OF process_query_stage2
def process_query_stage2(context: VADARContext, processed_instance_data: Dict):
    """
    Performs Stage 2. It now correctly serializes data before orchestrating subprocess execution.
    """
    if not processed_instance_data:
        return None

    # Unpack data from Stage 1
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
        # --- API and Program Generation (runs in the consumer thread) ---
        console.print(f"[Consumer-{threading.get_ident()}] Index {index}: Generating APIs and program...")
        class_names_list = list(set(det_obj.class_name for det_obj in detected_objects_list))
        query_expansion_str = query_expansion(context, test_image_pil, remake_query(invert_query(refined_query), '_tag'))
        test_query = refined_query + query_expansion_str + "\nAnswer in an integer, string 'regionX', decimal distance, or 'left'/'right'."
        
        predef_api_signatures = display_predef_api()
        vqa_functions = generate_description_functions(context, test_query, class_names_list)
        
        generated_signatures, generated_docstrings = signature_agent(context, predef_api_signatures, test_query, vqa_functions)
        if not generated_signatures: raise Exception("Signature agent failed.")
        
        generated_api_code = api_agent(context, predef_api_signatures, generated_signatures, generated_docstrings, test_query)
        if not generated_api_code.strip(): raise Exception("API agent failed.")
        
        solution_program_code = program_agent(context, generated_api_code, test_query, vqa_functions)
        if not solution_program_code.strip(): raise Exception("Program agent failed.")

        # --- Subprocess Execution with Timeout ---
        console.print(f"[Consumer-{threading.get_ident()}] Index {index}: Serializing data and starting subprocess for execution...")

        # *** THIS IS THE CRITICAL FIX ***
        # Create a dictionary of data to be serialized and passed to the subprocess.
        data_to_serialize = {
            "image": test_image_pil,
            "depth_path": depth_path,
            "detected_objects": detected_objects_list
        }
        serialized_data = _serialize_for_subprocess(data_to_serialize)

        # Use a ProcessPoolExecutor to run the code in a separate process with a timeout
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                run_program_in_subprocess,
                solution_program_code,
                generated_api_code,
                serialized_data,
                # Pass necessary configuration safely
                gemini_api_key, 
                context.qwen_generator.client.base_url if hasattr(context.qwen_generator.client, 'base_url') else "http://localhost:8000/v1"
            )
            try:
                final_result, html_trace_output = future.result(timeout=120)  # 120-second timeout
            except concurrent.futures.TimeoutError:
                error_msg = f"Execution for index {index} timed out after 120 seconds."
                console.print(f"[Consumer-{threading.get_ident()}] [bold red]{error_msg}[/bold red]")
                final_result = f"Error: {error_msg}"
                html_trace_output = f"<h2>Execution Timed Out</h2><p>{error_msg}</p>"

    except Exception as e:
        console.print(f"[Consumer-{threading.get_ident()}] Program generation or execution failed for index {index}: {e}. Falling back to VQA.")
        html_trace_output += f"<h2>Fallback to VQA</h2><p><strong>Reason:</strong> {e}</p><pre>{traceback.format_exc()}</pre>"
        try:
            # Ensure the fallback uses a robust, thread-safe VQA call
            vqa_fallback_question = VQA_PROMPT.format(question=refined_query)
            final_result, _ = generate_vl(context, prompt=vqa_fallback_question, images=[image_path])
            html_trace_output += f"<p>VQA Fallback Answer: {final_result}</p>"
        except Exception as fallback_e:
            final_result = f"Program failed ({e}), and VQA fallback also failed: {fallback_e}"
            html_trace_output += f"<p><strong>VQA fallback also failed:</strong> {fallback_e}</p>"

    # --- Save results ---
    trace_file_path = f"execution_trace_id_{image_id}_index_{index}.html"
    with open(trace_file_path, "w", encoding="utf-8") as f:
        f.write(html_trace_output)

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
        loaded_data = load_all_data()
        if not loaded_data:
            console.print("[bold red]Failed to load data and models. Exiting.[/bold red]")
            return
        
        sgg_generator = initialize_and_get_generator(loaded_data['context'])
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
                console.print("\n[bold yellow]Exiting interactive mode. Goodbye![/bold yellow]")
                break

            choice_idx = int(choice_str)
            if not (0 <= choice_idx < len(all_annotations)):
                raise IndexError("Index out of range.")

            console.print(f"\n[Phase 3] Processing selected index: [bold magenta]{choice_idx}[/bold magenta]")

            # --- Stage 1: Scene Graph Generation ---
            console.print("\n--- Running Stage 1: Scene Graph Generation ---")
            processed_instance_data = process_annotation_by_index(
                loaded_data['context'], 
                sgg_generator, 
                all_annotations, 
                loaded_data['actual_image_dir'], 
                choice_idx
            )

            if not processed_instance_data:
                console.print(f"[bold red]Failed to complete Stage 1 for index {choice_idx}. Please select another.[/bold red]")
                continue

            # --- Stage 2: Agentic Workflow ---
            console.print("\n--- Running Stage 2: Agentic Program Generation & Execution ---")
            result = process_query(loaded_data['context'], processed_instance_data)
            
            if result:
                console.print("\n[bold green on black] Query processing completed for this item. [/bold green on black]")
                console.print(f"Final normalized result: {json.dumps(result, indent=2)}")
            else:
                console.print("[bold red]Query processing failed for this item.[/bold red]")
            
            console.print("\nReturning to selection menu...")
            time.sleep(2)

        except (ValueError, IndexError):
            console.print("[bold red]Invalid input. Please enter a valid number from the table.[/bold red]\n")
        except Exception as e:
            console.print(f"[bold red]An unexpected error occurred in the main loop: {e}[/bold red]")
            traceback.print_exc()
            console.print("Please try another selection or restart the program.")

# --- Main entry point ---
if __name__ == "__main__":
    # To run in batch processing mode, comment out main_interactive_mode()
    # and uncomment the main_batch_processing() call below.
    
    # --- Option 1: Interactive Single-Question Mode (Default) ---
    main_interactive_mode()
    
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
    #
    # INDICES_TO_PROCESS = None  # Or specify a list of indices, e.g., [0, 1, 5]
    # NUM_WORKERS = 8
    # main_batch_processing(indices_to_process=INDICES_TO_PROCESS, workers=NUM_WORKERS)

# --- Main entry point ---
# if __name__ == "__main__":
#     # The default behavior is to run the interactive, combined workflow.
#     # To run in batch mode, you can comment out main_combined() and uncomment main_batch_processing().
#     while True:
#         result = main_combined()
#         if result:
#             print("\n--- FINAL NORMALIZED RESULT ---")
#             print(json.dumps(result, indent=2))
#         else:
#             print("Process did not complete successfully.")

#         # If you want to stop after one run, add a 'break' statement here.
#         # break

#     # Example of batch processing all annotations:
#     # batch_results = main_batch_processing()
#     # if batch_results:
#     #     with open("batch_results.json", "w") as f:
#     #         json.dump(batch_results, f, indent=2)
#     #     print("Batch results saved to batch_results.json")