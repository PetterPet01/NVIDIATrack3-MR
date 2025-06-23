import os
import sys
import numpy as np
import json
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import time
import torch
from IPython.display import Markdown, display, Code, HTML
from rich.console import Console
from rich.syntax import Syntax      
from rich.padding import Padding
from rich.style import Style
import re
# import groundingdino.datasets.transforms as T_gd
# from groundingdino.util.inference import load_model, predict
from unik3d.models import UniK3D
import torchvision.transforms as TV_T
from transformers import AutoModelForCausalLM, AutoTokenizer

from pathlib import Path

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

console = Console(highlight=False, force_terminal=False)

import base64
import io
import json
import os
from typing import List, Dict, Any, Optional, Union

import requests # For making HTTP requests
from PIL import Image # For image handling

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

from openai import OpenAI

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


import torch
import re
import math
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import numpy as np

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
        if messages:
            self.history = messages
        elif prompt:
            self.history = [{"role": "user", "content": prompt}]
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
            if pixel_values is not None:
                response, new_history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    self.history[-1]["content"],
                    generation_config=gen_config,
                    num_patches_list=num_patches_list,
                    history=self.history[:-1] if len(self.history) > 1 else None,
                    return_history=True
                )
            else:
                response, new_history = self.model.chat(
                    self.tokenizer,
                    None,
                    self.history[-1]["content"],
                    generation_config=gen_config,
                    history=self.history[:-1] if len(self.history) > 1 else None,
                    return_history=True
                )
                
            # Clean response
            response = self.remove_substring(response, "```python")
            response = self.remove_substring(response, "```")
            response = self.remove_think_tags(response)
            
            # Update history
            self.history = new_history
            
            return response, self.history
            
        except Exception as e:
            print(f"InternVL3 Generator: Error during generation: {e}")
            raise

# --- Globals ---
qwen_generator = None
device = None
grounding_dino = None
unik3d_model = None

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

def initialize_modules(
    qwen_model_name="Qwen/Qwen3-8B",
    qwen_max_new_tokens=1536,
    qwen_device_preference=None,
    # SpatialRGPT API parameters
    spatialrgpt_api_base_url="http://localhost:8001", # URL of your SpatialRGPT FastAPI server
    other_models_device_preference="cuda:0",
    unik3d_model_size="Large"
):
    global qwen_generator, spatialrgpt_generator, device, unik3d_model

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
    print(f"Main device for GroundingDINO/UniK3D: {device}")

    # Initialize Qwen3 Generator
    print(f"Initializing Qwen3 Generator with model: {qwen_model_name}")
    try:
            # def __init__(self, model_name="Qwen/Qwen3-8B", temperature=0.7, base_url=None, api_key="dummy", max_new_tokens=1024):

        # qwen_generator = Generator(
        #     model_name=qwen_model_name,
        #     temperature=qwen_temperature,
        #     max_new_tokens=qwen_max_new_tokens,
        #     device_preference=qwen_device_preference
        # )
        qwen_generator = Generator(
            model_name=qwen_model_name,
            max_new_tokens=qwen_max_new_tokens,
            base_url="http://localhost:8000/v1"
        )
        print("Qwen3 Generator initialized successfully.")
    except Exception as e:
        print(f"Error initializing Qwen3 Generator: {e}")
        qwen_generator = None

    # --- Initialize SpatialRGPT API Client Wrapper ---
    print(f"Initializing SpatialRGPT Client for API at '{spatialrgpt_api_base_url}'")
    try:
        spatialrgpt_generator = SpatialRGPTClient(
            base_url=spatialrgpt_api_base_url
        )
        # This client does not load a model locally; it calls an API.
        # Its 'device' and 'dtype' are just placeholders indicating it's API-based.
        print("SpatialRGPT API Client Wrapper initialized successfully.")
    except Exception as e:
        print(f"Error initializing SpatialRGPT API Client Wrapper: {e}")
        import traceback
        traceback.print_exc()
        spatialrgpt_generator = None

    # Initialize GroundingDINO
    # print("Initializing GroundingDINO")
    # try:
    #     config_path = os.path.join(vadar_root, "models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    #     weights_path = os.path.join(vadar_root, "models/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    #     if not (os.path.exists(config_path) and os.path.exists(weights_path)):
    #         print(f"Warning: GroundingDINO paths not found. Trying relative paths...")
    #         config_path = "../models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    #         weights_path = "../models/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    #     grounding_dino = load_model(config_path, weights_path)
    #     print(f"GroundingDINO initialized. It will use device '{device}' during prediction.")
    # except Exception as e:
    #     print(f"Error initializing GroundingDINO: {e}")
    #     grounding_dino = None

    # Initialize UniK3D
    print(f"Initializing UniK3D (Size: {unik3d_model_size})")
    try:
        unik3d_model = _instantiate_unik3d_model(model_size_str=unik3d_model_size, device_str=str(device))
    except Exception as e:
        unik3d_model = None
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

def generate(prompt: str = None, messages: list = None, enable_thinking=False, temperature=0.2):
    global qwen_generator
    if not qwen_generator:
        error_msg = "Error: Qwen3 Generator not initialized. Call initialize_modules first."
        print(error_msg)
        return error_msg, messages or []
    try:
        response_text, updated_history = qwen_generator.generate(prompt=prompt, messages=messages, enable_thinking=enable_thinking, temperature=temperature)
        return response_text, updated_history
    except Exception as e:
        error_msg = f"Error during qwen_generator.generate() call: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, messages or []

def generate_wrapped(
                    messages,
                    max_new_tokens,
                    temperature,
                    do_sample,
                    return_full_text,
                    tokenizer,  # Will be None when using custom function
                    pipeline,    # Will be None when using custom function
                    logger):
    return generate(messages=messages)[0]

def generate_vl(prompt: str = None, messages: list = None, images=None, enable_thinking=False):
    global internvl_generator
    if not internvl_generator:
        error_msg = "Error: InternVL3 Generator not initialized. Call initialize_modules first."
        print(error_msg)
        return error_msg, messages or []
    
    try:
        # Handle multimodal inputs
        pixel_values = None
        num_patches_list = None
        
        # Process images
        if images:
            if isinstance(images, str):
                images = [images]
                
            all_pixel_tensors = [] # Renamed for clarity
            for img_path in images:
                single_image_tensor = internvl_generator.load_image(img_path) 
                all_pixel_tensors.append(single_image_tensor)
                
            pixel_values_tensor = torch.cat(all_pixel_tensors) # Now this should work
            num_patches_list = [p.shape[0] for p in all_pixel_tensors]
            pixel_values_tensor = pixel_values_tensor.to(internvl_generator.dtype).to(internvl_generator.device)

        # Convert messages to InternVL3 format
        if messages:
            # Convert OpenAI-style messages to InternVL3 format
            internvl_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    internvl_messages.append({
                        "role": "user",
                        "content": msg["content"]
                    })
                elif msg["role"] == "assistant":
                    internvl_messages.append({
                        "role": "assistant",
                        "content": msg["content"]
                    })
            
            # Extract last message for generation
            last_message = internvl_messages[-1]["content"]
            history = internvl_messages[:-1] if len(internvl_messages) > 1 else None
            
        elif prompt:
            internvl_messages = [{"role": "user", "content": prompt}]
            last_message = prompt
            history = None
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")
        
        # Generation config
        gen_config = {
            "max_new_tokens": internvl_generator.max_new_tokens,
            "do_sample": True if internvl_generator.temperature > 0 else False
        }
        
        if internvl_generator.temperature > 0:
            gen_config.update({
                "temperature": internvl_generator.temperature,
                "top_p": 0.9,
                "top_k": 20
            })
        
        # Generate response
        if pixel_values_tensor is not None:
            response, new_history = internvl_generator.model.chat(
                internvl_generator.tokenizer,
                pixel_values_tensor,
                last_message,
                generation_config=gen_config,
                num_patches_list=num_patches_list,
                history=history,
                return_history=True
            )
        else:
            response, new_history = internvl_generator.model.chat(
                internvl_generator.tokenizer,
                None,
                last_message,
                generation_config=gen_config,
                history=history,
                return_history=True
            )
            
        # Clean response
        response = internvl_generator.remove_substring(response, "```python")
        response = internvl_generator.remove_substring(response, "```")
        response = internvl_generator.remove_think_tags(response)
        
        return response, new_history
        
    except Exception as e:
        print(f"InternVL3 Generator: Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", messages or []

import torch
import numpy as np
from PIL import Image
import cv2 # For resizing masks if needed (though your code already does it)
from typing import Tuple

# Assume spatialrgpt_generator is initialized globally and has the necessary attributes/methods
# global spatialrgpt_generator
def generate_spatial_vlm_response(
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    rgb_image: Optional[Union[str, Image.Image]] = None,
    depth_image: Optional[Union[str, Image.Image]] = None,
    segmentation_masks: Optional[List[Union[Image.Image, np.ndarray]]] = None, # User-provided masks
    temperature: float = 0.7,
    max_tokens: int = 512,
    client_instance: Optional[SpatialRGPTClient] = None,
    server_base_url: str = "http://localhost:8001", # Default from spatialrgpt_server.py
    model_name: str = "SpatialRGPT-VILA1.5-8B",     # Default from server & client
    conv_mode: str = "llama_3",                     # Default from client & server
    use_bfloat16: bool = True                       # Default from client & server
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generates a response from the SpatialRGPT model via the SpatialRGPTClient,
    passing user-provided segmentation masks directly. This function does NOT
    generate or use bounding boxes from these masks.

    Args:
        prompt (str, optional): A single text prompt. Used if 'messages' is None.
        messages (list, optional): A list of messages in OpenAI-style format.
        rgb_image (Union[str, Image.Image], optional): Path to an RGB image or a PIL Image object.
                                                       Required if segmentation_masks are provided.
        depth_image (Union[str, Image.Image], optional): Path to a depth image or a PIL Image object.
        segmentation_masks (List[Union[Image.Image, np.ndarray]], optional):
                                                       A list of PIL.Image or NumPy arrays representing
                                                       user-provided segmentation masks. These are sent directly
                                                       to the server. The prompt/messages should use <regionX> tags,
                                                       and the server will assume the order of masks corresponds
                                                       to these tags (e.g., first mask for <region0>).
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens to generate.
        client_instance (SpatialRGPTClient, optional): An existing SpatialRGPTClient instance.
                                                       If None, one is created.
        server_base_url (str): Base URL of the SpatialRGPT FastAPI server.
        model_name (str): Name of the model to use.
        conv_mode (str): LLaVA conversation mode template name.
        use_bfloat16 (bool): Whether to request bfloat16 usage on the server.

    Returns:
        Tuple (response_text, updated_history_list).
        On failure, (error_message_string, original_messages_list_or_empty).
    """
    if client_instance:
        client = client_instance
    else:
        try:
            # In a real setup, SpatialRGPTClient would be properly imported or defined
            if 'SpatialRGPTClient' not in globals():
                raise NameError("SpatialRGPTClient class is not defined. Please ensure it's imported or defined.")
            client = SpatialRGPTClient(base_url=server_base_url)
        except Exception as e:
            err_msg = f"Failed to initialize SpatialRGPTClient: {str(e)}"
            return err_msg, (messages if messages is not None else [])

    active_messages: List[Dict[str, Any]]
    if messages:
        active_messages = [dict(m) for m in messages] # Create a mutable copy
    elif prompt:
        active_messages = [{"role": "user", "content": prompt}]
    else:
        err_msg = "Error: Either 'prompt' or 'messages' must be provided."
        return err_msg, (messages if messages is not None else [])

    if segmentation_masks and not rgb_image:
        err_msg = "Error: If segmentation_masks are provided, an rgb_image must also be provided."
        return err_msg, active_messages
        
    process_depth_flag: Optional[bool] = None
    if depth_image is not None:
        process_depth_flag = True

    try:
        messages_before_api_call = [dict(m) for m in active_messages]

        # Call the client, passing segmentation_masks directly.
        # - region_boxes is explicitly None as per user requirement.
        # - use_sam_segmentation is False, as we are not sending boxes that would need SAM.
        #   The server will prioritize provided_seg_masks if they are present in image_options.
        response_data = client.chat(
            messages=active_messages,
            model_name=model_name,
            main_image_path_or_pil=rgb_image,
            depth_image_path_or_pil=depth_image,
            segmentation_masks_data=segmentation_masks, # Pass user-provided masks
            region_boxes=None,                          # Explicitly NO bounding boxes from this function
            use_sam_segmentation=False,                 # Not attempting SAM-based-on-boxes
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

    except ValueError as ve: # From client input validation (e.g. if client.chat raises it)
        err_msg = f"Client input error: {str(ve)}"
        return err_msg, active_messages
    except requests.exceptions.RequestException as req_e: # From client network/HTTP errors
        err_msg = f"API Call Error: {str(req_e)}"
        return err_msg, active_messages
    except Exception as e: # Catch any other unexpected errors
        import traceback
        err_trace = traceback.format_exc()
        err_msg = f"An unexpected error occurred: {str(e)}\nTrace: {err_trace}"
        return err_msg, active_messages

def enforce_python_code_output(raw_response):
    """
    Reinforces the model output to strictly contain only Python code.
    Removes any surrounding natural language explanation or formatting,
    and preserves only the Python code block.

    Args:
        raw_response (str): The initial model output which may include non-code content.

    Returns:
        str: Cleaned Python code string.
    """
    global qwen_generator
    # First try to extract Python code from triple backticks
    # code_block_match = re.search(r'```python\n(.*?)\n```', raw_response, re.DOTALL)
    # if code_block_match:
    #     print("Qwen3 Generator: Extracted Python code from code block.")
    #     return code_block_match.group(1).strip()

    # If no code block found, prompt the model to fix the response
    correction_prompt = (
        "You are a strict Python code formatter. Your task is to take the following text "
        "and extract or rewrite it as a valid Python code block without any extra explanation, "
        "markdown, or text outside of the code. Preserve the original logic and structure."
        "If the code is not valid Python, fix it to ensure it runs correctly. Else, keep it as is / make only minimal changes. "
        "Ensure that the code is properly indented and formatted according to Python standards. "
        # "Remove any comments, print statements, or other non-code elements. "
        "Do not include any imports or function definitions that are not necessary for the code to run. "
        "Do not add redundant code that is not present in the original response. "
        # "The output should be a single valid Python code block without any additional text.\n\n"
        # "The code should not contain any type annotations or type enforcement. "
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
        corrected_output, _ = qwen_generator.generate(messages=correction_input, enable_thinking=False, temperature=0.2)
    except Exception as e:
        print(f"Qwen3 Generator: Error during Python code enforcement: {e}")
        raise

    # Try to extract Python code again from the refined response
    code_block_match = re.search(r'```python\n(.*?)\n```', corrected_output, re.DOTALL)
    if code_block_match:
        print("Qwen3 Generator: Extracted cleaned Python code after refinement.")
        return code_block_match.group(1).strip()
    else:
        # Fallback: return everything after assuming it's code
        print("Qwen3 Generator: No code block found; returning full output as potential code.")
        return corrected_output.strip()

# -- Visual Reasoning Tools --
def save_visualized_image(img, output_path="visualized_scene.png"):
    """
    Saves the given image to disk.
    
    Args:
        img: PIL.Image object (the visualized image).
        output_path: Path where the image should be saved.
    """
    try:
        img.save(output_path)
        print(f"Visualized image saved to {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Error saving visualized image: {e}")
        
def loc(image, object_prompt): 
    BOX_THRESHOLD = 0.25 
    TEXT_THRESHOLD = 0.25 
    trace_html = []

    if not grounding_dino:
        trace_html.append("<p>Error: GroundingDINO not initialized in loc().</p>")
        return [], trace_html

    original_object_prompt = object_prompt
    width, height = image.size
    prompt_gd = object_prompt.replace(' ', '.')
    if not prompt_gd.endswith('.'): 
        prompt_gd += " ."
    else: 
        prompt_gd += " ."

    _, img_gd_tensor = transform_image(image)

    gd_device_str = str(device)
    is_cpu = gd_device_str == "cpu"
    
    # GroundingDINO's predict function handles moving model to device if not already there
    # For autocast, it's safer to move the model explicitly first.
    # grounding_dino.to(gd_device_str) # Ensure model is on the correct device before autocast
    
    with torch.autocast(device_type=gd_device_str.split(":")[0], enabled=not is_cpu, dtype=torch.float16 if not is_cpu else torch.float32):
        boxes_tensor, logits, phrases = predict(
            model=grounding_dino,
            image=img_gd_tensor, # This tensor will be moved to gd_device_str by predict if not already there
            caption=prompt_gd,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=gd_device_str, 
        )
    bboxes = _parse_bounding_boxes(boxes_tensor, width, height)

    if len(bboxes) == 0:
        trace_html.append(f"<p>Locate '{original_object_prompt}': No objects found</p>")
        return [], trace_html

    trace_html.append(f"<p>Locate: {original_object_prompt}</p>")
    boxed_pil_image = box_image(image, bboxes)
    save_visualized_image(boxed_pil_image, "loc_output.png")
    boxed_html = html_embed_image(boxed_pil_image)
    trace_html.append(boxed_html)

    display_prompt = original_object_prompt
    if len(bboxes) > 1 and not original_object_prompt.endswith("s"):
        display_prompt += "s"
    trace_html.append(f"<p>{len(bboxes)} {display_prompt} found</p>")
    trace_html.append(f"<p>Boxes: {bboxes}</p>")

    return bboxes, trace_html

def extract_2d_bounding_box(detected_object):
    return detected_object.bounding_box_2d.astype(int), []
    
def extract_3d_bounding_box(detected_object):
    return detected_object.bounding_box_3d_oriented.get_box_points(), []

from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Assuming DetectedObject class is defined as above

# Load a pre-trained embedding model (only once)
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', trust_remote_code=True)

from typing import List
from sklearn.metrics.pairwise import cosine_similarity

def is_similar_text(text1: str, text2: str) -> bool:
    """
    Determine whether two texts are semantically similar using cosine similarity.

    Args:
        text1 (str): First text input.
        text2 (str): Second text input.

    Returns:
        bool: True if similarity is above threshold, False otherwise.
    """
    threshold = 0.5
    prompt_embedding = embedding_model.encode([text1], convert_to_numpy=True)
    class_embedding = embedding_model.encode([text2], convert_to_numpy=True)

    similarity = cosine_similarity(prompt_embedding, class_embedding)[0][0]
    return similarity > threshold

def retrieve_objects(detected_objects: List[DetectedObject], object_prompt: str) -> List[DetectedObject]:
    """
    Retrieve DetectedObject instances matching the given object_prompt using semantic similarity.

    Args:
        detected_objects (List[DetectedObject]): List of detected objects.
        object_prompt (str): Description or name of the desired object(s).
                             Use "objects" to return all detected objects.

    Returns:
        List[DetectedObject]: Matching detected objects with similarity > 0.9, sorted by similarity score.
    """
    if not detected_objects:
        return [], []

    # Return all objects if prompt is generic
    if object_prompt.strip().lower() in ("object", "objects", "items", "things"):
        return detected_objects, []

    class_names = [obj.class_name for obj in detected_objects]
    
    prompt_embedding = embedding_model.encode([object_prompt], convert_to_numpy=True)
    class_embeddings = embedding_model.encode(class_names, convert_to_numpy=True)

    similarities = cosine_similarity(prompt_embedding, class_embeddings)[0]
    
    # Filter and sort only those with similarity > 0.9
    scored_objects = [(score, obj) for score, obj in zip(similarities, detected_objects) if score > 0.9]
    scored_objects.sort(reverse=True, key=lambda x: x[0])

    return [obj for score, obj in scored_objects], []

def get_3D_object_size(detected_object) -> Tuple[float, float, float]:
    """
    Returns the width, height, and length of the object in 3D real-world (meters) space.

    Args:
        detected_object (DetectedObject): A detected object with a valid oriented bounding box.

    Returns:
        tuple: (width, height, length) in meters.
    """
    # Get the oriented bounding box
    obb = detected_object.bounding_box_3d_oriented

    # The extent of the box gives the size in each local axis (x, y, z)
    extent = obb.extent

    # By convention:
    # - x-axis → width
    # - y-axis → depth or length
    # - z-axis → height
    width = float(extent[0])
    length = float(extent[1])  # often treated as depth/length depending on coordinate system
    height = float(extent[2])

    return (width, height, length), ([])

def depth(image, bbox): 
    trace_html = []
    global unik3d_model, device 

    if not unik3d_model:
        trace_html.append("<p>Error: UniK3D model not initialized in depth().</p>")
        return 0.0, trace_html

    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(c, (int, float)) for c in bbox)):
        trace_html.append(f"<p>Error: Invalid bbox format for depth(): {bbox}. Expected [x0,y0,x1,y1].</p>")
        return 0.0, trace_html

    x_mid = (bbox[0] + bbox[2]) / 2
    y_mid = (bbox[1] + bbox[3]) / 2

    unik3d_device_str = str(device) 
    with torch.no_grad():
        image_np_rgb = np.array(image.convert("RGB"))
        if image_np_rgb.ndim == 2:
            image_np_rgb = np.stack((image_np_rgb,)*3, axis=-1) 

        image_tensor = torch.from_numpy(image_np_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image_tensor = image_tensor.to(unik3d_device_str)
        
        # unik3d_model.to(unik3d_device_str) # Ensure model is on device

        outputs = unik3d_model.infer(image_tensor, camera=None, normalize=True) 
        points_3d_per_pixel = outputs["points"].squeeze(0).permute(1, 2, 0).cpu().numpy() # H, W, 3
        depth_map_preds = points_3d_per_pixel[:, :, 2] # Z-coordinates as depth

        h_preds, w_preds = depth_map_preds.shape
        safe_y_mid = min(max(0, int(y_mid * h_preds / image.height)), h_preds - 1) 
        safe_x_mid = min(max(0, int(x_mid * w_preds / image.width)), w_preds - 1) 

        depth_val = depth_map_preds[safe_y_mid, safe_x_mid]

    trace_html.append(f"<p>Depth query at original image coordinates: ({x_mid:.1f}, {y_mid:.1f})</p>")
    trace_html.append(f"<p>Corresponding point on predicted depth map (scaled): ({safe_x_mid}, {safe_y_mid})</p>")

    dotted_pil_image = dotted_image(image, [[x_mid, y_mid]]) 
    dotted_html = html_embed_image(dotted_pil_image)
    trace_html.append("<p>Point on original image:</p>")
    trace_html.append(dotted_html)

    dotted_depth_map_pil = dotted_image(depth_map_preds, [[safe_x_mid, safe_y_mid]])
    dotted_depth_map_html = html_embed_image(dotted_depth_map_pil)
    trace_html.append("<p>Point on predicted depth map (Z from UniK3D):</p>")
    trace_html.append(dotted_depth_map_html)

    trace_html.append(f"<p>Depth value: {depth_val:.4f}</p>")
    return float(depth_val), trace_html
import os
import tempfile
import re
import base64
from PIL import Image

def _vqa_predict(img, depth, masks, question):
    """VQA prediction using SpatialRGPT model"""
    try:
        # Format the question with VQA prompt template
        prompt = VQA_PROMPT.format(question=question)
        
        # Create full prompt with image token
        full_prompt = f"{prompt}"
        
        # Create temporary directory for image storage
        # with tempfile.TemporaryDirectory() as tmpdir:
        #     temp_path = os.path.join(tmpdir, "temp_image.png")
            
            # Save PIL image to temporary file
            # img.save(temp_path, format=img.format)
            
            # Call InternVL3 generate function with image path
        output, _ = generate_spatial_vlm_response(
            prompt=full_prompt,
            rgb_image=img,  # Pass image path to multimodal handler
            depth_image=depth,
            segmentation_masks=masks,
            temperature=0.2
        )
        
        # Extract answer from response
        answer_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip().lower()
        return output.strip().lower()
        
    except Exception as e:
        print(f"Error in VQA prediction: {e}")
        return f"Error: {str(e)}"

def remake_query(query, tag=''):
    # Generate region IDs based on number of <mask> tokens
    counter = [0]  # Using a list to allow mutation inside replacer

    def replacer(match):
        replacement = f"<region{counter[0]}{tag}>"
        counter[0] += 1
        return replacement

    query = re.sub(r'<mask>', replacer, query)
    return query

def invert_query(query):
    # Replace all <regionN> tags with <mask>
    return re.sub(r'<region\d+>', '<mask>', query)

def vqa(image, depth, question, objects): 
    trace_html = []

    # is_holistic = False
    # if bbox is None or \
    #    (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and
    #     bbox[0] == 0 and bbox[1] == 0 and 
    #     bbox[2] >= image.width -1 and bbox[3] >= image.height -1 ): 
    #     img_for_vqa_display = image 
    #     trace_html.append(f"<p>VQA (holistic query): {question}</p>")
    #     trace_html.append(html_embed_image(image, 300))
    #     is_holistic = True
    # else:
    #     try:
    #         cmin, rmin, cmax, rmax = [int(c) for c in bbox]
    #         if cmax > cmin and rmax > rmin:
    #             img_for_vqa_display = image.crop((cmin, rmin, cmax, rmax))
    #             trace_html.append(f"<p>VQA (region query): {question}</p>")
    #             boxed_original_pil = box_image(image, [bbox])
    #             trace_html.append("<p>Query region on original image:</p>")
    #             trace_html.append(html_embed_image(boxed_original_pil, 300))
    #             trace_html.append("<p>Cropped region (for display, VLM would use this):</p>")
    #             trace_html.append(html_embed_image(img_for_vqa_display, 200))
    #         else:
    #             print(f"Warning: Invalid bbox for VQA crop: {bbox}. Using full image for display.")
    #             img_for_vqa_display = image
    #             trace_html.append(f"<p>VQA (holistic due to invalid crop {bbox}): {question}</p>")
    #             trace_html.append(html_embed_image(image, 300))
    #             is_holistic = True
    #     except (ValueError, TypeError):
    #         print(f"Warning: Invalid bbox format for VQA crop: {bbox}. Using full image for display.")
    #         img_for_vqa_display = image
    #         trace_html.append(f"<p>VQA (holistic due to invalid bbox format {bbox}): {question}</p>")
    #         trace_html.append(html_embed_image(image, 300))
    #         is_holistic = True
    refined_question = remake_query(question)
    trace_html.append(f"<p>VQA Question: {refined_question}</p>")
    trace_html.append(str([obj.description for obj in objects]))
    print(f"VQA Question: {refined_question}")
    answer = _vqa_predict(image, depth, [det_obj.segmentation_mask_2d for det_obj in objects], refined_question)
    # trace_html.extend(vqa_predict_trace) 
    trace_html.append(f"<p>VQA Final Answer (from _vqa_predict): {answer}</p>")

    return answer.lower(), trace_html

def _get_iou(box1, box2): 
    if not (isinstance(box1, (list, tuple, np.ndarray)) and len(box1) == 4 and all(isinstance(c, (int, float, np.integer, np.floating)) for c in box1) and
            isinstance(box2, (list, tuple, np.ndarray)) and len(box2) == 4 and all(isinstance(c, (int, float, np.integer, np.floating)) for c in box2)):
        print(f"Warning: Invalid box format for IoU calculation. Box1: {box1}, Box2: {box2}")
        return 0.0

    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    width_inter = max(0, x2_inter - x1_inter)
    height_inter = max(0, y2_inter - y1_inter)
    area_inter = width_inter * height_inter
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    area_union = area_box1 + area_box2 - area_inter
    iou = area_inter / area_union if area_union > 1e-6 else 0.0
    return iou

def same_object(image, bbox1, bbox2): 
    iou_val = _get_iou(bbox1, bbox2)
    answer = iou_val > 0.92 

    trace_html = []
    boxes_to_draw = []
    if isinstance(bbox1, (list, tuple)) and len(bbox1) == 4: boxes_to_draw.append(bbox1)
    if isinstance(bbox2, (list, tuple)) and len(bbox2) == 4: boxes_to_draw.append(bbox2)
    
    if boxes_to_draw:
        boxed_pil_image = box_image(image, boxes_to_draw)
        trace_html.append(f"<p>Same Object Check (Bbox1: {bbox1}, Bbox2: {bbox2})</p>")
        trace_html.append(html_embed_image(boxed_pil_image, 300))
    else:
        trace_html.append(f"<p>Same Object Check with invalid bboxes (Bbox1: {bbox1}, Bbox2: {bbox2}). Cannot draw.</p>")

    trace_html.append(f"<p>IoU: {iou_val:.3f}, Same object: {answer}</p>")
    return answer, trace_html

def find_overlapping_regions(parent_region: DetectedObject, 
                            countable_regions: List[DetectedObject]) -> List[int]:
    """
    Find regions that overlap with parent region within cropped area
    
    Args:
        parent_region: parent region object
        countable_regions: List of region objects that can be counted
        
    Returns:
        List of region_index (int) that meet threshold
    """
    padding = 20
    overlap_threshold = 0.2
    # Extract bounding box coordinates
    parent_mask = parent_region.segmentation_mask_2d
    x, y, w, h = parent_region.bounding_box_2d
    
    # Add padding
    x_min = max(0, int(x - padding))
    y_min = max(0, int(y - padding))
    x_max = min(parent_mask.shape[1], int(x + w + padding))
    y_max = min(parent_mask.shape[0], int(y + h + padding))
    # x_min, y_min, x_max, y_max = crop_bbox
    
    overlapping_regions = []

    for region in countable_regions:
        if region.description == parent_region.description:
            continue  # Skip parent region itself
        
        region_mask = region.segmentation_mask_2d
        region_bbox = region.bounding_box_2d
        
        # Quick bbox intersection check for optimization
        # rx, ry, rw, rh = region_bbox
        # if (rx + rw < x_min or rx > x_max or 
        #     ry + rh < y_min or ry > y_max):
        #     continue  # No bbox intersection
        
        # Check pixel-level overlap within cropped area
        crop_parent = parent_mask[y_min:y_max, x_min:x_max]
        crop_region = region_mask[y_min:y_max, x_min:x_max]
        
        # Calculate overlap
        overlap = np.logical_and(crop_parent, crop_region)
        overlap_area = np.sum(overlap)
        
        if overlap_area > 0:
            # Calculate overlap percentage relative to the child region
            region_area = np.sum(crop_region)
            overlap_percentage = overlap_area / region_area if region_area > 0 else 0
            
            # Check if overlap meets threshold
            if overlap_percentage >= overlap_threshold:
                match = re.search(r'\d+', region.description)
                if match:
                    number = int(match.group())
                overlapping_regions.append(number)
    
    return overlapping_regions

def calculate_3d_distance(obj1: DetectedObject, obj2: DetectedObject):
    """
    Calculates the center-to-center distance between two 3D objects using their oriented bounding boxes.

    Args:
        obj1 (DetectedObject): The first detected object, containing a 3D oriented bounding box.
        obj2 (DetectedObject): The second detected object, containing a 3D oriented bounding box.

    Returns:
        float: (meters) Euclidean distance between the centers of the two objects.
    """
    # Get the center of each oriented bounding box
    center1 = obj1.bounding_box_3d_oriented.get_center()
    center2 = obj2.bounding_box_3d_oriented.get_center()
    
    # Calculate Euclidean distance between centers
    distance = np.linalg.norm(center1 - center2)
    
    return distance + distance*0.22

def get_2D_object_size(image, bbox): 
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(c, (int, float)) for c in bbox)):
        trace_html_error = [f"<p>Error: Invalid bbox format for get_2D_object_size(): {bbox}.</p>"]
        return (0, 0), trace_html_error 

    width = abs(bbox[0] - bbox[2])
    height = abs(bbox[1] - bbox[3])

    trace_html = []
    trace_html.append(f"<p>2D Object Size for Bbox: {bbox}</p>")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4: # ensure valid bbox before drawing
        boxed_pil_image = box_image(image, [bbox]) 
        trace_html.append(html_embed_image(boxed_pil_image, 300))
    trace_html.append(f"<p>Width: {width}, Height: {height}</p>")
    return (width, height), trace_html

def execute_program(program, image, depth, detected_objects, api):
    wrapped_program = wrap_solution_code(program)
    header_lines = [
    "import math",
    "from typing import Tuple, List, Dict, Optional",
    "from PIL import Image as PILImage, ImageDraw",  # PILImage will be from inject_globals
    "import numpy as np",  # np from inject_globals
    "import open3d as o3d",  # o3d from inject_globals
    "import io, base64, sys, os, re, tempfile, json, time, torch",
    "from pathlib import Path",
    "import PIL",
    ""
  ]
    header_str = "\n".join(line for line in header_lines) + '\n'


    executable_program = header_str + api + wrapped_program

    # Store program lines in a list for reference
    program_lines = executable_program.split("\n")

    # Create a function to get line text from our stored program
    def get_line(line_no):
        # Adjust line number to 0-based index
        idx = line_no - 1
        if 0 <= idx < len(program_lines):
            return program_lines[idx]
        return ""

    # parse API methods
    api_methods = re.findall(r"def (\w+)\s*\(.*\):", api)

    # Create a trace string to record execution
    html_trace = []

    # Create namespace for execution
    def _traced_loc(image, object_prompt):
        result, html = loc(image, object_prompt)
        html_trace.extend(html)
        return result

    def _traced_retrieve_objects(detected_objects, object_prompt):
        result, html = retrieve_objects(detected_objects, object_prompt)
        html_trace.extend(html)
        return result

    def _traced_extract_2d_bounding_box(detected_object):
        result, html = extract_2d_bounding_box(detected_object)
        html_trace.extend(html)
        return result
        
    def _traced_extract_3d_bounding_box(detected_object):
        result, html = extract_3d_bounding_box(detected_object)
        html_trace.extend(html)
        return result

    def _traced_get_3D_object_size(detected_object):
        result, html = get_3D_object_size(detected_object)
        html_trace.extend(html)
        return result

    def _traced_vqa(image, depth, question, objects):
        result, html = vqa(image, depth, question, objects)
        html_trace.extend(html)
        return result

    def _traced_depth(image, bbox):
        result, html = depth(image, bbox)
        html_trace.extend(html)
        return result

    def _traced_get_2D_object_size(image, bbox):
        result, html = get_2D_object_size(image, bbox)
        html_trace.extend(html)
        return result

    # def _traced_same_object(image, bbox1, bbox2):
    #     result, html = same_object(image, bbox1, bbox2)
    #     html_trace.extend(html)
    #     return result

    def _traced_find_overlapping_regions(parent_region, 
                            countable_regions):
        result = find_overlapping_regions(parent_region, countable_regions)
        html_trace.extend([])
        return result

    def _traced_calculate_3d_distance(obj1, obj2):
        result = calculate_3d_distance(obj1, obj2)
        html_trace.extend([])
        return result
    
    def _traced_is_similar_text(text1, text2):
        result = is_similar_text(text1, text2)
        return result

    # Create a custom trace function to track line execution
    def trace_lines(frame, event, arg):
        if event == "line":
            method_name = frame.f_code.co_name
            if method_name == "solution_program" or method_name in api_methods:
                line_no = frame.f_lineno
                line = get_line(line_no).strip()
                if len(line) > 0:
                    html_trace.append(
                        f"<p><code>[{method_name}] Line {line_no}: {line}</code></p>"
                    )
        return trace_lines

    namespace = {
        "DetectedObject": DetectedObject,
        "loc": _traced_loc,
        "retrieve_objects": _traced_retrieve_objects,
        "vqa": _traced_vqa,
        "depth": depth,
        "image": image,
        "detected_objects": detected_objects,
        "extract_2d_bounding_box": _traced_extract_2d_bounding_box,
        "extract_3d_bounding_box": _traced_extract_3d_bounding_box,
        "get_3D_object_size": _traced_get_3D_object_size,
        # "get_2D_object_size": _traced_get_2D_object_size,
        # "same_object": _traced_same_object,
        "find_overlapping_regions": _traced_find_overlapping_regions,
        "calculate_3d_distance": _traced_calculate_3d_distance,
        "is_similar_text": _traced_is_similar_text,
    }

    # Set up the trace function
    sys.settrace(trace_lines)

    try:
        # Execute the program
        exec(executable_program, namespace)

    finally:
        # Disable tracing
        sys.settrace(None)

    final_result = namespace["final_result"]

    print('-----LUCKY LUCKY-----'*10)
    print(final_result)
    print('-----LUCKY LUCKY-----'*10)

    # Return both the text trace and HTML trace
    return final_result, "\n".join(html_trace)


def display_result(final_result, image_pil, question, ground_truth): 
    result_html_parts = []
    result_html_parts.append(f"<h3>Question:</h3><p>{question}</p>")
    result_html_parts.append("<h3>Input Image:</h3>")
    result_html_parts.append(html_embed_image(image_pil, 300))
    result_html_parts.append(f"<h3>Program Result:</h3><p>{final_result}</p>")
    if ground_truth is not None: 
        result_html_parts.append(f"<h3>Ground Truth:</h3><p>{ground_truth}</p>")
    return "\n".join(result_html_parts)

import spacy
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from wordfreq import word_frequency

nlp = spacy.load("en_core_web_sm")

def extract_descriptions(query: str, objects: list[str], similarity_threshold=0.8) -> dict:
  doc = nlp(query)
  result = defaultdict(list)

  for token in doc:
    if token.text.lower() in objects:
      for child in token.children:
        if child.dep_ in ("amod", "compound"):
          result[token.text.lower()].append(child.text)
      for left in token.lefts:
        if left.pos_ == "ADJ":
          result[token.text.lower()].append(left.text)

  deduped = {}
  for obj, adjs in result.items():
    if not adjs:
      deduped[obj] = []
      continue

    embeddings = embedding_model.encode(adjs)
    used = set()
    clusters = []

    for i in range(len(adjs)):
      if i in used:
        continue
      cluster = [adjs[i]]
      used.add(i)
      for j in range(i + 1, len(adjs)):
        if j in used:
          continue
        sim = util.cos_sim(embeddings[i], embeddings[j]).item()
        if sim >= similarity_threshold:
          cluster.append(adjs[j])
          used.add(j)

      # Pick the simplest (most frequent) word
      representative = min(cluster, key=lambda w: -word_frequency(w, 'en'))
      clusters.append(representative)

    deduped[obj] = clusters

  return deduped


def generate_description_functions(query: str, objects: list[str]):
  descriptions = extract_descriptions(query, objects)

  function_lines = []

  for obj, descs in descriptions.items():
    for desc in descs:
      func_name = f"is_{obj}_{desc}".replace(" ", "_")
      function_code = (
        f"def {func_name}(image, detected_object):\n"
        f"  # Check if {obj} is {desc}\n"
        f"  return vqa(image, depth, 'Is {obj} <region0> {desc}?', [detected_object])\n"
      )
      function_lines.append(function_code)

  return "\n".join(function_lines)

def signature_agent(predef_api, query, vqa_functions):
    template_prompt = SIGNATURE_PROMPT
    prompt = template_prompt.format(signatures=predef_api, question=query, vqa_functions=vqa_functions)
    print("Signature Agent Prompt (first 200 chars):\n", prompt[:200] + "...") 
    console.print(Padding(f"[Signature Agent] Query: {query}", (1, 2), style="on blue"))
    output_text, _ = generate(prompt=prompt, enable_thinking=False, temperature=0.3)
    
    if not isinstance(output_text, str) or output_text.startswith('Error:'):
        print(f"Signature Agent Error or invalid output: {output_text}")
        return [], []
    
    docstrings = re.findall(r"<docstring>(.*?)</docstring>", output_text, re.DOTALL)
    signatures = re.findall(r"<signature>(.*?)</signature>", output_text, re.DOTALL)
    return signatures, docstrings

def expand_template_to_instruction(json_obj):
  instructions = []

  # Step 1: Explain the core task
  explanation = json_obj.get("explanation")
  if explanation:
    instructions.append(f"\nClarify the request: {explanation}.")

  # Step 2: Generate visual check reminders
  for i, check in enumerate(json_obj.get("visual_checks", []), 1):
    obj = check.get("object", "object")
    adj = check.get("adjective", "property")
    vqa_call = check.get("vqa_call", "vqa(...)")
    instructions.append(
      f"{i}. Check visually if the '{obj}' is '{adj}' by calling:\n   → {vqa_call}\n"
      f"   You need to implement this function or route it to your VQA module."
    )

  # Step 3: Describe spatial instructions
  spatial_steps = json_obj.get("spatial_instructions", [])
  if spatial_steps:
    instructions.append("\nThen follow these spatial steps:")
    for i, step in enumerate(spatial_steps, 1):
      instructions.append(f"  {i}. {step}")

  return "\n".join(instructions)

def query_expansion(img, query):
    prompt = """
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

## Examples

**Query**: "Which transporter is best for pickup?"
```json
{
  "explanation": "Best means closest available (empty) transporter",
  "visual_checks": [
    {
      "object": "transporter",
      "adjective": "empty",
      "vqa_call": "vqa(image, depth, 'Is this <mask_tag> (transporter) empty?', detected_objects[0])"
    }
  ],
  "spatial_instructions": [
    "Filter transporters by empty status",
    "Calculate distances to pickup location", 
    "Select closest empty transporter"
  ]
}
```

**Query**: "Are pallets in the buffer zone?"
```json
{
  "explanation": "Pallets located within buffer zone boundaries",
  "visual_checks": [],
  "spatial_instructions": [
    "Get buffer zone coordinates",
    "Check each pallet position against boundaries",
    "Count pallets within zone"
  ]
}
```

## Output Format
```json
{
  "explanation": "Concrete definition of vague terms (1 sentence max)",
  "visual_checks": [
    {
      "object": "object_name",
      "adjective": "property",
      "vqa_call": "vqa(image, depth, 'Is this <mask_tag> (object) [property]?', [reference])"
    }
  ],
  "spatial_instructions": [
    "Clear action steps for spatial logic (3-5 steps max)"
  ]
}
```
Input:

"""
    print(f"Calling SpatialRGPT with query: {query}")
    output_text, _ = generate_spatial_vlm_response(
    prompt=prompt + query,
    # rgb_image=img,
    temperature=0.1)

    # output_text = generate(prompt=prompt + query, enable_thinking=False, temperature=0.2)[0]

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
      "vqa_call": "string (function call of form: vqa(image, depth, 'Is this <mask> (object) adjective?', [detected_objects[i]]))"
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
   vqa(image, depth, 'Is this <region0> (object) adjective?', [detected_objects[i]])
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
    output_text_refined = generate(prompt=prompt_json_refine + output_text, enable_thinking=False, temperature=0.2)[0]

    print('-Debug-'*20)
    print(output_text_refined)
    print('-Debug-'*20)
    # More robust regex
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', output_text_refined, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)
        result = json.loads(json_str)
        print(result)
    else:
        print("No JSON found")
    return expand_template_to_instruction(result)


def api_agent_2(predef_signatures, gen_signatures, gen_docstrings):
    if not gen_signatures: 
        print("API Agent: No generated signatures to process.")
        return ""

    method_names = []
    for sig in gen_signatures:
        match = re.compile(r"def (\w+)\s*\(.*\):").search(sig)
        if match:
            method_names.append(match.group(1))
        else:
            print(f"Warning: Could not parse method name from signature: {sig}")
            method_names.append(f"unknown_method_{len(method_names)}")

    gen_signatures_text = "".join(
        [doc + "\n" + sig + "\n\n" for doc, sig in zip(gen_docstrings, gen_signatures)]
    )

    implementations = {}
    error_count = {} 
    max_retries_per_signature = 3 

    sig_idx = 0
    while sig_idx < len(gen_signatures):
        signature = gen_signatures[sig_idx]
        docstring = gen_docstrings[sig_idx]
        current_method_name = method_names[sig_idx]

        console.print(Padding(f"[API Agent] Generating implementation for: {current_method_name}", (1,2), style="on blue"))

        if error_count.get(sig_idx, 0) >= max_retries_per_signature:
            print(f"Skipping implementation generation for '{current_method_name}' after {max_retries_per_signature} retries.")
            implementations[sig_idx] = f"    # Error: Max retries reached for {current_method_name}\n    pass\n"
            sig_idx += 1
            continue

        template_prompt = API_PROMPT
        prompt = template_prompt.format(
            predef_signatures=predef_signatures,
            generated_signatures=gen_signatures_text, 
            docstring=docstring, 
            signature=signature, 
        )
        
        output_text, _ = generate(prompt=prompt, enable_thinking=True)
        
        if not isinstance(output_text, str) or "Error:" in output_text :
            print(f"API Agent Error for {current_method_name}: {output_text}. Retrying...")
            error_count[sig_idx] = error_count.get(sig_idx, 0) + 1
            continue 

        implementation_match = re.search(
            r"<implementation>(.*?)</implementation>", output_text, re.DOTALL
        )

        if not implementation_match:
            print(f"Warning: No <implementation> tag found for {current_method_name}. Retrying. Output was: {output_text[:200]}...")
            error_count[sig_idx] = error_count.get(sig_idx, 0) + 1
            continue

        implementation = implementation_match.group(1).strip()
        
        lines = implementation.split("\n")
        if lines and lines[0].strip().startswith("def "):
            implementation = "\n".join(lines[1:])
        
        implementation = "\n".join(["    " + line.strip() for line in implementation.split("\n") if line.strip()])

        # Basic recursion check (can be refined or removed if too restrictive)
        error = False
        # Allowing calls to other generated methods. Self-recursion is also allowed if LLM generates it.
        # More complex checks can be added here if needed.

        if not error:
            implementations[sig_idx] = implementation
            sig_idx += 1
        
    api_parts = []
    for i in range(len(gen_signatures)):
        doc = gen_docstrings[i]
        sig = gen_signatures[i]
        impl = implementations.get(i, "    pass # Implementation generation failed")
        api_parts.append(f"{doc}\n{sig}\n{impl}\n")

    merged_api = "\n".join(api_parts)
    return merged_api.replace("\t", "    ") 

def program_agent(api, query, vqa_functions):
    console.print(Padding(f"[Program Agent] Query: {query}", (1,2), style="on blue"))
    prompt = PROGRAM_PROMPT.format(
        predef_signatures=MODULES_SIGNATURES, api=api, question=query, vqa_functions=vqa_functions
    )
    
    output_text, _ = generate(prompt=prompt, enable_thinking=False, temperature=0.3)

    if not isinstance(output_text, str) or "Error:" in output_text:
        print(f"Program Agent Error: {output_text}")
        return "final_result = 'Error: Program generation failed'"

    program_match = re.search(r"<program>(.*?)</program>", output_text, re.DOTALL)
    if not program_match:
        print(f"Warning: No <program> tag found in program_agent output. Raw output: {output_text[:200]}...")
        program_code = output_text.strip()
    else:
        program_code = program_match.group(1).strip()

    program_lines = program_code.split('\n')
    indented_program_lines = ["    " + line for line in program_lines] 
    
    return "\n".join(indented_program_lines).replace("\t", "    ")

def transform_image(og_image): 
    transform = T_gd.Compose(
        [
            T_gd.RandomResize([800], max_size=1333),
            T_gd.ToTensor(),
            T_gd.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    og_image = og_image.convert("RGB")
    im_t, _ = transform(og_image, None)
    return og_image, im_t

def box_image(img, boxes): 
    img1 = img.copy()
    draw = ImageDraw.Draw(img1)
    for box in boxes:
        try:
            x_0, y_0, x_1, y_1 = [int(c) for c in box]
            if x_1 > x_0 and y_1 > y_0:
                 draw.rectangle([x_0, y_0, x_1, y_1], outline="red", width=3)
            else:
                print(f"Warning: Invalid box coordinates for drawing: {box}")
        except (ValueError, TypeError):
            print(f"Warning: Non-numeric or invalid box coordinates for drawing: {box}")
            continue
    return img1

def html_embed_image(img, size=300): 
    if isinstance(img, np.ndarray):
        try:
            if img.dtype == np.float32 or img.dtype == np.float64:
                img_pil = Image.fromarray(depth_to_grayscale(img)) 
            else:
                img_pil = Image.fromarray(img)
        except Exception as e:
            print(f"Could not convert NumPy array to PIL for HTML embed: {e}. Using blank image.")
            img_pil = Image.new('RGB', (size,size), color='grey')
    elif isinstance(img, Image.Image):
        img_pil = img.copy()
    else:
        print(f"Unsupported image type for HTML embed: {type(img)}. Using blank image.")
        img_pil = Image.new('RGB', (size,size), color='grey')

    img_pil.thumbnail((size, size)) 
    
    if img_pil.mode == "F": 
        img_to_save = Image.fromarray(depth_to_grayscale(np.array(img_pil))).convert("RGB")
    elif img_pil.mode in ["RGBA", "P", "L"]: 
        img_to_save = img_pil.convert("RGB")
    elif img_pil.mode == "RGB":
        img_to_save = img_pil
    else: 
        print(f"Warning: Unusual image mode '{img_pil.mode}' for HTML embed. Attempting RGB conversion.")
        try:
            img_to_save = img_pil.convert("RGB")
        except Exception as e:
            print(f"Conversion to RGB failed for mode {img_pil.mode}: {e}. Using placeholder.")
            img_to_save = Image.new('RGB', (img_pil.width, img_pil.height), color='lightgrey')

    with BytesIO() as buffer:
        try:
            img_to_save.save(buffer, "jpeg")
        except Exception as e:
            print(f"Error saving image to buffer for HTML embed: {e}. Using blank image JPEG.")
            blank_pil = Image.new('RGB', (size//2, size//2), color='lightgrey')
            blank_pil.save(buffer, "jpeg")

        base64_img = base64.b64encode(buffer.getvalue()).decode()
    return (
        f'<img style="vertical-align:middle" src="data:image/jpeg;base64,{base64_img}">'
    )

def depth_to_grayscale(depth_map): 
    depth_map_np = np.array(depth_map, dtype=np.float32)
    d_min = np.min(depth_map_np)
    d_max = np.max(depth_map_np)
    if d_max - d_min < 1e-6:
        normalized = np.zeros_like(depth_map_np)
    else:
        normalized = (depth_map_np - d_min) / (d_max - d_min)
    grayscale = (normalized * 255).astype(np.uint8)
    return grayscale

def dotted_image(img, points): 
    if isinstance(img, np.ndarray):
        if img.ndim == 2 or img.dtype in [np.float32, np.float64]:
            img_pil = Image.fromarray(depth_to_grayscale(img)).convert("RGB")
        else: 
            img_pil = Image.fromarray(img).convert("RGB")
    elif isinstance(img, Image.Image):
        img_pil = img.copy().convert("RGB")
    else:
        print(f"Unsupported image type for dotted_image: {type(img)}. Returning blank.")
        return Image.new('RGB', (100,100), color='grey')

    img_width = img_pil.size[0]
    dot_size = max(1, int(img_width * 0.01)) 

    draw = ImageDraw.Draw(img_pil)
    for pt in points:
        try:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img_pil.width and 0 <= y < img_pil.height:
                bbox = (x - dot_size, y - dot_size, x + dot_size, y + dot_size)
                draw.ellipse(bbox, fill="red", outline="black")
            else:
                print(f"Warning: Point ({x},{y}) out of bounds for dotting on image size {img_pil.size}")
        except (ValueError, TypeError, IndexError):
            print(f"Warning: Invalid point coordinates for dotting: {pt}")
            continue
    return img_pil

def _parse_bounding_boxes(boxes, width, height): 
    if len(boxes) == 0:
        return []
    bboxes = []
    for box_tensor in boxes: 
        cx, cy, w, h = box_tensor.tolist()
        x1 = (cx - 0.5 * w) * width
        y1 = (cy - 0.5 * h) * height
        x2 = (cx + 0.5 * w) * width
        y2 = (cy + 0.5 * h) * height
        bboxes.append(
            [
                max(0, int(x1)),
                max(0, int(y1)),
                min(width -1, int(x2)), 
                min(height -1, int(y2)),
            ]
        )
    return bboxes

def load_image(im_pth):
    try:
        image = Image.open(im_pth).convert("RGB")
        return image
    except FileNotFoundError:
        print(f"Error: Image file not found at {im_pth}")
        return None
    except Exception as e:
        print(f"Error loading image {im_pth}: {e}")
        return None

def display_predef_api():
    console.print(
        Syntax(
            MODULES_SIGNATURES,
            "python",
            theme="dracula", 
            line_numbers=False,
            word_wrap=True,
        )
    )
    return MODULES_SIGNATURES

def display_generated_program(program, api):
    api_lines = len(api.split("\n"))
    console.print(
        Syntax(
            program,
            "python",
            theme="dracula",
            line_numbers=True,
            start_line=api_lines + 2, 
            word_wrap=True,
        )
    )

def display_generated_signatures(generated_signatures, generated_docstrings):
    code_to_display = ""
    for signature, docstring in zip(generated_signatures, generated_docstrings):
        code_to_display += docstring + "\n" 
        code_to_display += signature + "\n\n" 
    console.print(
        Syntax(
            code_to_display,
            "python",
            theme="dracula",
            line_numbers=False,
            word_wrap=True,
        )
    )

def display_generated_api(api):
    console.print(
        Syntax(
            api,
            "python",
            theme="dracula",
            line_numbers=True,
            word_wrap=True,
        )
    )
# -- Main Demo Section --

def wrap_solution_code(solution_program_code):
  indented_code = "\n".join(
    "    " + line if line.strip() != "" else line
    for line in solution_program_code.splitlines()
  )
  return f"\ndef solution_program(image, detected_objects):\n{indented_code}\n    return final_result\nfinal_result = solution_program(image, detected_objects)"

import signal
import os
import sys
import shutil # For rmtree
import traceback # For stacktrace formatting
import runpy # For executing files
import linecache # For tracing (optional here)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def _get_docstring_types_for_pipeline(docstring: str):
    args_pattern = re.compile(r"Args:\s*((?:\s+\w+ \(\w+\): .+\n)+)")
    args_match = args_pattern.search(docstring)
    args_section = args_match.group(1) if args_match else ""

    returns_pattern = re.compile(r"Returns:\s+(\w+): .+")
    returns_match = returns_pattern.search(docstring)
    returns_section = returns_match.group(1) if returns_match else ""

    arg_types = re.findall(r"\s+(\w+) \((\w+)\):", args_section)
    return arg_types, returns_section

import re
import numpy as np # Ensure np is available for this helper

# Assume _get_docstring_types_for_pipeline is defined elsewhere and works correctly
# def _get_docstring_types_for_pipeline(docstring: str):
#     args_pattern = re.compile(r"Args:\s*((?:\s+\w+ \([\w.:\s\[\]|]+\): .+\n)+)") # Updated regex for complex types
#     args_match = args_pattern.search(docstring)
#     args_section = args_match.group(1) if args_match else ""

#     returns_pattern = re.compile(r"Returns:\s+([\w.:\s\[\]|]+): .+") # Updated regex
#     returns_match = returns_pattern.search(docstring)
#     returns_section = returns_match.group(1).strip() if returns_match else "None" # Default to "None" type string

#     arg_types = re.findall(r"\s+([\w_]+) \(([\w.:\s\[\]|]+)\):", args_section) # Updated regex
#     return arg_types, returns_section

def _get_robust_return_code_for_pipeline(docstring, signature, img_width, img_height):
    """
    Generates plausible Python code for the body of a stub function.
    """
    method_name_match = re.search(r"def\s+([\w_]+)\s*\(", signature)
    if not method_name_match:
        return "    return None # Fallback: Could not parse method name from signature"
    method_name = method_name_match.group(1)

    # Simplified argument parsing for stubs: find the names of arguments from the signature
    args_in_signature_match = re.search(r"def\s+[\w_]+\s*\((.*?)\)", signature)
    arg_names = []
    if args_in_signature_match:
        arg_string = args_in_signature_match.group(1)
        # Extract just the names, ignore types for stub generation simplicity
        arg_names = [a.split(':')[0].strip() for a in arg_string.split(',') if a.strip()]


    # --- Specific stubs for your predefined functions ---
    if method_name == "is_similar_text":
        # Args: text1 (str), text2 (str) -> Returns: bool
        return "    return str(text1).lower() == str(text2).lower() # Simple stub"

    elif method_name == "extract_2d_bounding_box":
        # Args: detected_object (DetectedObject) -> Returns: array (np.ndarray)
        # Assumes detected_object has a bounding_box_2d attribute
        # The mock DetectedObject created in the test script will have this.
        return f"""
    if hasattr(detected_object, 'bounding_box_2d') and detected_object.bounding_box_2d is not None:
        return detected_object.bounding_box_2d
    return np.array([0, 0, min({img_width-1}, 10), min({img_height-1}, 10)]) # Fallback bbox
"""

    elif method_name == "extract_3d_bounding_box":
        # Args: detected_object (DetectedObject) -> Returns: List[Tuple[float, float, float]]
        # Assumes detected_object.bounding_box_3d_oriented.get_box_points()
        return """
    if hasattr(detected_object, 'bounding_box_3d_oriented') and detected_object.bounding_box_3d_oriented:
        try:
            # o3d.geometry.OrientedBoundingBox.get_box_points() returns Vector3dVector
            # which needs to be converted to list of tuples of floats.
            points_vector = detected_object.bounding_box_3d_oriented.get_box_points()
            return [tuple(p) for p in np.asarray(points_vector)]
        except Exception:
            pass # Fall through to default if attribute access fails
    # Fallback default 8 corners for a small box
    return [(0.0,0.0,0.0), (0.1,0.0,0.0), (0.0,0.1,0.0), (0.1,0.1,0.0),
            (0.0,0.0,0.1), (0.1,0.0,0.1), (0.0,0.1,0.1), (0.1,0.1,0.1)]
"""

    elif method_name == "retrieve_objects":
        # Args: detected_objects (List[DetectedObject]), object_prompt (str) -> Returns: List[DetectedObject]
        # Relies on the first argument in the signature being the list of detected_objects.
        detected_objects_arg_name = arg_names[0] if arg_names else "detected_objects" # Fallback name
        return f"""
    if str(object_prompt).lower() == "objects":
        return list({detected_objects_arg_name}) # Return all
    # Simple mock: return first object if its class_name loosely matches, else empty
    if {detected_objects_arg_name} and hasattr({detected_objects_arg_name}[0], 'class_name'):
        if str(object_prompt).lower() in {detected_objects_arg_name}[0].class_name.lower():
            return [{detected_objects_arg_name}[0]]
    return []
"""

    elif method_name == "vqa":
        # Args: image (PIL.Image.Image), question (str), bbox (list) -> Returns: str
        return """
    if "color" in str(question).lower():
        return "red"
    elif "what" in str(question).lower():
        return "a mock object"
    elif "is there" in str(question).lower() or "are there" in str(question).lower() :
        return "yes"
    return "mock vqa answer"
"""

    elif method_name == "same_object":
        # Args: image (PIL.Image.Image), bbox1 (list), bbox2 (list) -> Returns: bool
        # Simple IoU-like check for stub (assuming bbox format [x1,y1,x2,y2])
        return """
    try:
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])
        if x2_inter < x1_inter or y2_inter < y1_inter: # No overlap
            return False
        # Simplified overlap check: if they overlap by more than 50% of min area
        area1 = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
        area2 = (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])
        area_inter = (x2_inter-x1_inter) * (y2_inter-y1_inter)
        if area1 > 0 and area_inter / area1 > 0.5: return True
        if area2 > 0 and area_inter / area2 > 0.5: return True
        return False # Default if small or no significant overlap
    except:
        return False # Fallback if bbox access fails
"""

    elif method_name == "find_overlapping_regions":
        # Args: parent_region (DetectedObject), countable_regions (List[DetectedObject]) -> Returns: List[int]
        # Mock: Use simplified bbox IoU approximation to simulate overlap
        return """
        try:
            def get_iou(bbox1, bbox2):
                x1_inter = max(bbox1[0], bbox2[0])
                y1_inter = max(bbox1[1], bbox2[1])
                x2_inter = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
                y2_inter = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
                if x2_inter < x1_inter or y2_inter < y1_inter:
                    return 0.0
                inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                area1 = bbox1[2] * bbox1[3]
                area2 = bbox2[2] * bbox2[3]
                return inter_area / area2 if area2 > 0 else 0.0

            overlapping = []
            for idx, region in enumerate(countable_regions):
                if region.description == parent_region.description:
                    continue
                iou = get_iou(parent_region.bounding_box_2d, region.bounding_box_2d)
                if iou >= 0.6:
                    overlapping.append(idx)
            return overlapping
        except:
            return []
        """


    elif method_name == "get_3D_object_size":
        # Args: detected_object (DetectedObject) -> Returns: tuple (width, height, length)
        # Assumes detected_object.bounding_box_3d_oriented.extent
        return """
    if hasattr(detected_object, 'bounding_box_3d_oriented') and detected_object.bounding_box_3d_oriented:
        try:
            extent = detected_object.bounding_box_3d_oriented.extent # extent is [width, depth, height] typically
            return (float(extent[0]), float(extent[2]), float(extent[1])) # W, H, L mapping
        except:
            pass
    return (0.1, 0.05, 0.2) # Fallback W, H, L
"""
    else:
        # Generic fallback based on return type if specific stub not found
        _, return_type_str = _get_docstring_types_for_pipeline(docstring)
        if "bool" in return_type_str.lower(): return "    return False"
        if "str" in return_type_str.lower(): return "    return 'stub_string'"
        if "int" in return_type_str.lower(): return "    return 0"
        if "float" in return_type_str.lower(): return "    return 0.0"
        if "list" in return_type_str.lower() or "array" in return_type_str.lower(): return "    return []"
        if "tuple" in return_type_str.lower(): return "    return ()"
        return "    return None # Fallback: Unknown return type for stub"

# You also need the _get_docstring_types_for_pipeline function to be robust:
def _get_docstring_types_for_pipeline(docstring: str):
    # Improved regex to handle various type hints including unions, optionals, and qualified names
    args_pattern = re.compile(r"Args:\s*((?:\s*[\w_]+\s*\([\w\s.:|\[\],_]+\)\s*:.+\n)+)", re.IGNORECASE)
    args_match = args_pattern.search(docstring)
    args_section = args_match.group(1) if args_match else ""

    returns_pattern = re.compile(r"Returns:\s*([\w\s.:|\[\],_]+)\s*:.+", re.IGNORECASE)
    returns_match = returns_pattern.search(docstring)
    returns_section = returns_match.group(1).strip() if returns_match else "None"

    # Regex to capture arg_name and arg_type, accommodating complex types
    arg_entries = re.findall(r"\s*([\w_]+)\s*\(([\w\s.:|\[\],_]+)\)\s*:", args_section)
    
    # Normalize types: remove spaces, handle common aliases
    def normalize_type(type_str):
        type_str = type_str.strip()
        # Add more normalizations as needed, e.g., 'array' -> 'np.ndarray' if consistent
        if type_str.lower() == "array": return "np.ndarray" # Example
        if type_str.lower() == "pil.image.image": return "PILImage.Image"
        return type_str

    arg_types = [(name, normalize_type(type_str)) for name, type_str in arg_entries]
    
    return arg_types, normalize_type(returns_section)

import open3d as o3d
from PIL import Image as PILImage # Assuming this is how PIL.Image is imported for DetectedObject
import numpy as np
import io
import base64

# --- (Your DetectedObject class definition provided above) ---
# class DetectedObject:
#   ... (rest of your class definition)

def create_mock_detected_object(
    class_name="mock_item",
    description="a generic mock detected item",
    image_width=200, image_height=150, # For mask and bbox consistency
    bbox_2d_coords=None, # e.g., [10, 10, 50, 50]
    has_point_cloud=True,
    num_3d_points=100,
    has_crop=True
):
    # 2D Mask (simple rectangle)
    mask_2d = np.zeros((image_height, image_width), dtype=bool)
    if bbox_2d_coords:
        x1, y1, x2, y2 = bbox_2d_coords
        mask_2d[y1:y2, x1:x2] = True
    else: # Default small mask if no bbox given
        mask_2d[5:15, 5:15] = True
        bbox_2d_coords = np.array([5, 5, 15, 15])


    # RLE Mask (simplified for mock - real RLE is more complex)
    # For a simple mock, we can just represent the mask shape or a placeholder
    rle_mask_2d_mock = f"rle_placeholder_for_shape_{image_height}x{image_width}"

    # 3D Point Cloud
    point_cloud_3d = o3d.geometry.PointCloud()
    if has_point_cloud:
        points = np.random.rand(num_3d_points, 3) * np.array([0.5, 0.5, 0.2]) # Small extent
        colors = np.random.rand(num_3d_points, 3) # Random colors
        point_cloud_3d.points = o3d.utility.Vector3dVector(points)
        point_cloud_3d.colors = o3d.utility.Vector3dVector(colors)

    # 3D Bounding Boxes (derived simply from point cloud for mock)
    if has_point_cloud and point_cloud_3d.has_points():
        bounding_box_3d_axis_aligned = point_cloud_3d.get_axis_aligned_bounding_box()
        bounding_box_3d_oriented = point_cloud_3d.get_oriented_bounding_box()
    else: # Default small boxes if no point cloud
        center = np.array([0.0, 0.0, 0.1])
        extent = np.array([0.05, 0.05, 0.05])
        R = np.identity(3)
        bounding_box_3d_oriented = o3d.geometry.OrientedBoundingBox(center, R, extent)
        bounding_box_3d_axis_aligned = o3d.geometry.AxisAlignedBoundingBox(center - extent/2, center + extent/2)


    # Image Crop
    image_crop_pil = None
    if has_crop:
        if bbox_2d_coords:
            crop_w = bbox_2d_coords[2] - bbox_2d_coords[0]
            crop_h = bbox_2d_coords[3] - bbox_2d_coords[1]
            image_crop_pil = PILImage.new("RGB", (max(1, crop_w), max(1, crop_h)), "green") # Simple green crop
        else:
            image_crop_pil = PILImage.new("RGB", (10, 10), "green")


    return DetectedObject(
        class_name=class_name,
        description=description,
        segmentation_mask_2d=mask_2d,
        rle_mask_2d=rle_mask_2d_mock,
        bounding_box_2d=np.array(bbox_2d_coords) if bbox_2d_coords else None,
        point_cloud_3d=point_cloud_3d,
        bounding_box_3d_oriented=bounding_box_3d_oriented,
        bounding_box_3d_axis_aligned=bounding_box_3d_axis_aligned,
        image_crop_pil=image_crop_pil
    )

def create_mock_image_with_content(width=200, height=150):
    img = PILImage.new("RGB", (width, height), "lightgrey")
    draw = PILImage.Draw(img)
    # Define some regions for potential mock objects
    # Region 1: Red square
    draw.rectangle([10, 10, 50, 50], fill="red", outline="black")
    # Region 2: Blue circle
    draw.ellipse([width - 60, height - 60, width - 10, height - 10], fill="blue", outline="black")
    return img

def _execute_file_for_pipeline(program_executable_path: str, 
                              execution_namespace: dict, # Keep for compatibility, or rename
                              timeout_seconds: int = 15,
                              inject_globals_dict: dict = None): # New parameter
    signal.signal(signal.SIGALRM, timeout_handler)
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        signal.alarm(timeout_seconds)
        script_dir = os.path.dirname(program_executable_path)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        # Use inject_globals_dict for runpy if provided
        effective_globals = inject_globals_dict if inject_globals_dict is not None else None # runpy expects None or a dict
        
        runpy.run_path(program_executable_path, init_globals=effective_globals, run_name="__main__")
        
        signal.alarm(0)
        if script_dir in sys.path and sys.path[0] == script_dir:
            sys.path.pop(0)
        
        err_output = stderr_capture.getvalue()
        if err_output:
            return Exception(f"Error during script execution (stderr):\n{err_output}"), err_output

        return None, None 
    except TimeoutException as e:
        stacktrace = traceback.format_exc() + f"\nStderr: {stderr_capture.getvalue()}"
        return e, stacktrace
    except Exception as e:
        stacktrace = traceback.format_exc() + f"\nStderr: {stderr_capture.getvalue()}"
        return e, stacktrace
    finally:
        signal.alarm(0)
        sys.stdout = original_stdout
        sys.stderr = original_stderr

def test_individual_api_function(
    api_info_under_test: dict,
    predef_signatures_text: str, # Still useful for API agent context, but not for stubs here
    successfully_implemented_apis: list,
    temp_dir_for_testing: str = "temp_api_test"
):
    method_name_under_test = api_info_under_test["method_name"]
    current_implementation_code = api_info_under_test["implementation"]
    current_signature_code = api_info_under_test["signature"]
    current_docstring = api_info_under_test["docstring"]

    test_exec_dir = os.path.join(temp_dir_for_testing, method_name_under_test)
    if os.path.exists(test_exec_dir):
        shutil.rmtree(test_exec_dir)
    os.makedirs(test_exec_dir, exist_ok=True)

    program_executable_path = os.path.join(test_exec_dir, f"test_{method_name_under_test}.py")

    MOCK_IMAGE_WIDTH = 200
    MOCK_IMAGE_HEIGHT = 150

    # --- Start building globals_to_inject ---
    # This assumes that models like grounding_dino, unik3d_model, internvl_generator,
    # embedding_model, device, and functions like loc, vqa, etc., are accessible
    # in the scope where test_individual_api_function is called (e.g., global in main script).

    # Create actual mock Python objects for the function under test's direct inputs
    from PIL import ImageDraw # <--- ADD THIS IMPORT

    _mock_image_for_test = PILImage.new('RGB', (MOCK_IMAGE_WIDTH, MOCK_IMAGE_HEIGHT), 'lightgray')
    # Draw on it if needed by specific tests, e.g., for visual properties.
    _draw = ImageDraw.Draw(_mock_image_for_test)
    _draw.rectangle([10,10,50,50], fill="red")
    _draw.ellipse([70,70,120,120], fill="blue")

    _mock_depth_image_for_test = PILImage.new('L', (MOCK_IMAGE_WIDTH, MOCK_IMAGE_HEIGHT), 128)
    # Draw depth variations - darker = closer, lighter = farther
    _draw = ImageDraw.Draw(_mock_depth_image_for_test)
    _draw.rectangle([10,10,50,50], fill=80)   # Closer object (darker)
    _draw.ellipse([70,70,120,120], fill=200)  # Farther object (lighter)
    _draw.rectangle([30,60,80,90], fill=50)   # Closest object (darkest)


    _obj1_bbox_coords = [10, 10, 50, 50] # Matches red square
    _mock_detected_object_1 = create_mock_detected_object(
        class_name='red_square_mock', description='a mock red square',
        image_width=MOCK_IMAGE_WIDTH, image_height=MOCK_IMAGE_HEIGHT,
        bbox_2d_coords=_obj1_bbox_coords,
        has_point_cloud=True, num_3d_points=20
    )
    _obj2_bbox_coords = [70, 70, 120, 120] # Matches blue circle
    _mock_detected_object_2 = create_mock_detected_object(
        class_name='blue_item_mock', description='a mock blue item',
        image_width=MOCK_IMAGE_WIDTH, image_height=MOCK_IMAGE_HEIGHT,
        bbox_2d_coords=_obj2_bbox_coords,
        has_point_cloud=True, num_3d_points=15
    )
    _mock_detected_objects_list = [_mock_detected_object_1, _mock_detected_object_2]

    def _traced_retrieve_objects(detected_objects, object_prompt):
        result, html = retrieve_objects(detected_objects, object_prompt)
        # html_trace.extend(html)
        return result

    def _traced_extract_2d_bounding_box(detected_object):
        result, html = extract_2d_bounding_box(detected_object)
        # html_trace.extend(html)
        return result
        
    def _traced_extract_3d_bounding_box(detected_object):
        result, html = extract_3d_bounding_box(detected_object)
        # html_trace.extend(html)
        return result

    def _traced_get_3D_object_size(detected_object):
        result, html = get_3D_object_size(detected_object)
        # html_trace.extend(html)
        return result

    def _traced_vqa(image, depth, question, objects):
        return 'result'

    def _traced_get_2D_object_size(image, bbox):
        result, html = get_2D_object_size(image, bbox)
        # html_trace.extend(html)
        return result

    # def _traced_same_object(image, bbox1, bbox2):
    #     result, html = same_object(image, bbox1, bbox2)
    #     html_trace.extend(html)
    #     return result

    def _traced_find_overlapping_regions(parent_region, 
                            countable_regions):
        result = find_overlapping_regions(parent_region, countable_regions)
        # html_trace.extend([])
        return result

    def _traced_calculate_3d_distance(obj1, obj2):
        result = calculate_3d_distance(obj1, obj2)
        # html_trace.extend([])
        return result
    
    def _traced_is_similar_text(text1, text2):
        result = is_similar_text(text1, text2)
        return result

    globals_to_inject = {
        # Essential modules (will be available globally in the executed script)
        "np": np, "PILImage": PILImage, "o3d": o3d, "torch": torch,
        "re": re, "os": os, "tempfile": tempfile, "math": math,
        "ImageDraw": ImageDraw, "BytesIO": BytesIO, "base64": base64, "json": json,
        "time": time, # If any tool uses it
        "Path": Path, # from pathlib

        # Models and device (must be initialized and accessible in current scope)
        # "grounding_dino": grounding_dino, # Global from initialize_modules
        "unik3d_model": unik3d_model,     # Global from initialize_modules
        "spatialrgpt_generator": spatialrgpt_generator, # Global from initialize_modules
        "qwen_generator": qwen_generator, # Global, if needed by any tool directly
        "embedding_model": embedding_model, # Global from retrieve_objects context
        "device": device,                 # Global from initialize_modules

        # Actual predefined utility functions (ensure these are the correct, working functions)
        "vqa": _traced_vqa,
        "retrieve_objects": _traced_retrieve_objects, "is_similar_text": _traced_is_similar_text,
        "extract_2d_bounding_box": _traced_extract_2d_bounding_box,
        "extract_3d_bounding_box": _traced_extract_3d_bounding_box,
        "get_3D_object_size": _traced_get_3D_object_size,
        "get_2D_object_size": _traced_get_2D_object_size,
        "find_overlapping_regions": _traced_find_overlapping_regions,
        "calculate_3d_distance": _traced_calculate_3d_distance,
        
        # Generator function for VQA if it uses it directly
        "generate_vl": generate_vl, 
        "generate": generate, # Qwen generator, if any utility needs it

        # Helper functions for tools (if they are not methods of a class)
        "transform_image": transform_image, # Needs T_gd
        "_parse_bounding_boxes": _parse_bounding_boxes,
        # "predict": predict, # from groundingdino.util.inference
        # "T_gd": T_gd, # groundingdino.datasets.transforms as T_gd - injecting the module
        # "TV_T": TV_T, # torchvision.transforms as TV_T
        "VQA_PROMPT": VQA_PROMPT, # Constant for _vqa_predict
        "DetectedObject": DetectedObject, # The class itself

        # Mock data to be globally available in the script's context
        # Use generic names like 'image' and 'detected_objects' for common tool parameters
        "image": _mock_image_for_test,
        "depth":_mock_depth_image_for_test,
        "detected_objects": _mock_detected_objects_list,

        # Also provide specific mock variable names if the test call construction uses them
        "mock_image_for_test": _mock_image_for_test,
        "mock_detected_object_1": _mock_detected_object_1,
        "mock_detected_object_2": _mock_detected_object_2,
        "mock_detected_objects_list": _mock_detected_objects_list,
        "MOCK_IMAGE_WIDTH": MOCK_IMAGE_WIDTH,
        "MOCK_IMAGE_HEIGHT": MOCK_IMAGE_HEIGHT,
    }
    # --- End building globals_to_inject ---

    # --- Start building the test script content ---
    script_content = []
    # Add imports for readability of the generated code, though objects come from inject_globals
    script_content.append("import math")
    script_content.append("from typing import Tuple, List, Dict, Optional")
    script_content.append("from PIL import Image as PILImage, ImageDraw") # PILImage will be from inject_globals
    script_content.append("import numpy as np") # np from inject_globals
    script_content.append("import open3d as o3d") # o3d from inject_globals
    script_content.append("import io, base64, sys, os, re, tempfile, json, time, torch, PIL")
    script_content.append("from pathlib import Path\n")


    # Write DetectedObject Class Definition (still good to have it defined in the script)
    script_content.append("\n# --- DetectedObject Class Definition (copied for test execution context) ---")
    # (DetectedObject class string as before)
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
    script_content.append(detected_object_class_string)
    script_content.append("# --- End DetectedObject Class Definition ---\n")

    # PREDEFINED API STUBS ARE NO LONGER GENERATED HERE

    # Write Successfully Implemented Generated APIs (Dependencies)
    script_content.append("# --- Successfully Implemented Generated APIs (Dependencies) ---")
    for api_item in successfully_implemented_apis:
        if api_item["method_name"] != method_name_under_test: # Should already be indented
            script_content.append(f"{api_item['signature']}\n{api_item['implementation']}\n")
    script_content.append("# --- End Dependencies ---\n")

    # Write the Function Under Test
    script_content.append(f"\n# --- Function under test: {method_name_under_test} ---")
    script_content.append(f"{current_signature_code}")
    indented_impl_body = "\n".join(["    " + line for line in current_implementation_code.strip().split("\n")])
    script_content.append(f"{indented_impl_body}\n")
    script_content.append("# --- End Function under test ---\n")

    # Prepare Mock Inputs and Call
    # The actual mock objects are in `globals_to_inject`.
    # The script will use their names directly.
    script_content.append("\n# --- Test Call ---")
    # This mapping helps construct the call string using the global variable names
    # that we've put into `globals_to_inject`.
    arg_name_to_global_var_name = {}
    arg_types, _ = _get_docstring_types_for_pipeline(current_docstring)

    for arg_name, arg_type_str in arg_types:
        normalized_arg_type = arg_type_str.lower()
        if "pilimage.image" in normalized_arg_type: # e.g. "image: PILImage.Image"
            arg_name_to_global_var_name[arg_name] = "image" # We put _mock_image_for_test as "image" in globals
        elif normalized_arg_type == "int":
            arg_name_to_global_var_name[arg_name] = "5" # Literal value, or a named global int if complex
        elif "str" in normalized_arg_type:
             # Create a specific mock string in globals if needed, else literal
            globals_to_inject[f"mock_str_arg_for_{arg_name}"] = f"mock_prompt_for_{arg_name}"
            arg_name_to_global_var_name[arg_name] = f"mock_str_arg_for_{arg_name}"
        elif "list[detectedobject]" in normalized_arg_type or "list[ detectedobject]" in normalized_arg_type :
            arg_name_to_global_var_name[arg_name] = "detected_objects" # We put _mock_detected_objects_list as "detected_objects"
        elif "list" in normalized_arg_type and ("int" in normalized_arg_type or "float" in normalized_arg_type): # bbox list
            globals_to_inject[f"mock_bbox_arg_for_{arg_name}"] = list(_mock_detected_object_1.bounding_box_2d) # Example bbox
            arg_name_to_global_var_name[arg_name] = f"mock_bbox_arg_for_{arg_name}"
        elif "detectedobject" == normalized_arg_type:
             arg_name_to_global_var_name[arg_name] = "mock_detected_object_1" # Example default
        # Add more specific mappings as needed based on your argument names and types
        else:
            print(f"Warning: Unhandled mock type '{arg_type_str}' for arg '{arg_name}' in test call. Defaulting to None for script.")
            globals_to_inject[f"mock_none_arg_for_{arg_name}"] = None
            arg_name_to_global_var_name[arg_name] = f"mock_none_arg_for_{arg_name}"


    call_args_str_list = []
    for arg_name, _ in arg_types:
        global_var_name_for_arg = arg_name_to_global_var_name.get(arg_name, "None")
        call_args_str_list.append(f"{arg_name}={global_var_name_for_arg}")
    
    call_string = f"{method_name_under_test}({', '.join(call_args_str_list)})"
    
    script_content.append("\nif __name__ == '__main__':")
    script_content.append(f"    print('--- Starting test execution for {method_name_under_test} with real tools ---')")
    script_content.append(f"    test_result = {call_string}")
    script_content.append(f"    print(f'Raw test result for {method_name_under_test}: {{test_result}}')")
    script_content.append(f"    print('--- Test execution finished ---')")

    with open(program_executable_path, "w") as f:
        f.write("\n".join(script_content))

    # Execute the program, passing the globals_to_inject
    error_obj, stacktrace_str = _execute_file_for_pipeline(
        program_executable_path, 
        execution_namespace={}, # Old parameter, not used by runpy like this
        timeout_seconds=60, # Increase timeout as real models are slower
        inject_globals_dict=globals_to_inject
    )

    if error_obj:
        return {"error": str(error_obj), "stacktrace": stacktrace_str}
    return {"error": None, "stacktrace": None}

def api_agent(predef_signatures, gen_signatures, gen_docstrings, query):
    if not gen_signatures:
        print("API Agent: No generated signatures to process.")
        return ""

    all_method_headers = []
    method_names_ordered = []
    for i in range(len(gen_signatures)):
        doc = gen_docstrings[i]
        sig = gen_signatures[i]
        match = re.compile(r"def\s+([\w_][\w\d_]*)\s*\((.*?)\)\s*(?:->\s*.*?)?:").search(sig)
        if match:
            method_name = match.group(1)
            all_method_headers.append({"method_name": method_name, "docstring": doc, "signature": sig})
            if method_name not in method_names_ordered:
                method_names_ordered.append(method_name)
        else:
            print(f"Warning: Could not parse method name from signature: {sig}")
            unknown_name = f"unknown_method_{i}"
            all_method_headers.append({"method_name": unknown_name, "docstring": doc, "signature": sig})
            if unknown_name not in method_names_ordered:
                method_names_ordered.append(unknown_name)

    processed_apis = [] # Stores dicts: {method_name, docstring, signature, implementation, messages, status}
    error_counts = {name: 0 for name in method_names_ordered}
    llm_messages_history = {name: [] for name in method_names_ordered}
    last_attempted_implementations = {name: None for name in method_names_ordered} # Store last valid code before test
    MAX_RETRIES_PER_FUNCTION = 3
    MAX_DEPENDENCY_DEPTH = 5

    temp_test_base_dir = "temp_api_test_run"
    if os.path.exists(temp_test_base_dir):
        shutil.rmtree(temp_test_base_dir)
    os.makedirs(temp_test_base_dir, exist_ok=True)

    processing_queue = list(method_names_ordered)
    
    def get_header_info(name):
        for header in all_method_headers:
            if header["method_name"] == name:
                return header
        return None

    idx = 0
    while idx < len(processing_queue):
        current_method_name = processing_queue[idx]
        header_info = get_header_info(current_method_name)

        if not header_info:
            print(f"Error: No header info found for {current_method_name}. Skipping.")
            idx += 1
            continue

        if any(api["method_name"] == current_method_name for api in processed_apis):
            idx += 1
            continue

        console.print(Padding(f"[API Agent] Attempting to implement: {current_method_name} (Attempt: {error_counts[current_method_name] + 1})", (1,2), style="on blue"))

        if error_counts[current_method_name] >= MAX_RETRIES_PER_FUNCTION:
            print(f"Skipping implementation for '{current_method_name}' after {MAX_RETRIES_PER_FUNCTION} retries.")
            
            final_impl_to_use = "    pass # Max retries reached, no prior valid implementation found"
            current_status = "failed_max_retries_placeholder"

            if last_attempted_implementations[current_method_name] is not None:
                print(f"Keeping last attempted (but failed) implementation for {current_method_name}.")
                last_failed_impl_body = last_attempted_implementations[current_method_name]
                # Ensure it's just the body and correctly indented
                final_impl_to_use = "\n".join(["    " + line for line in last_failed_impl_body.strip().split("\n")])
                current_status = "failed_max_retries_kept_last"
            
            processed_apis.append({
                "method_name": current_method_name,
                "docstring": header_info["docstring"],
                "signature": header_info["signature"],
                "implementation": final_impl_to_use,
                "messages": llm_messages_history[current_method_name],
                "status": current_status
            })
            idx += 1
            continue

        context_generated_signatures = []
        for h in all_method_headers:
            if h["method_name"] == current_method_name:
                continue
            if error_counts.get(h["method_name"], 0) < MAX_RETRIES_PER_FUNCTION:
                 context_generated_signatures.append(h["docstring"] + "\n" + h["signature"])

        context_signatures_text = "\n\n".join(context_generated_signatures)

        current_prompt_text = API_PROMPT.format(
            predef_signatures=predef_signatures,
            generated_signatures=context_signatures_text,
            docstring=header_info["docstring"],
            signature=header_info["signature"],
            question=query,
        )

        current_messages = llm_messages_history[current_method_name]
        if not current_messages:
            current_messages.append({"role": "user", "content": current_prompt_text})

        output_text, updated_messages = generate(messages=current_messages, enable_thinking=False, temperature=0.3)
        llm_messages_history[current_method_name] = updated_messages

        if not isinstance(output_text, str) or "Error:" in output_text:
            print(f"API Agent LLM Error for {current_method_name}: {output_text}. Retrying...")
            error_counts[current_method_name] += 1
            llm_messages_history[current_method_name].append({"role": "user", "content": f"The generation failed or returned an error. Please try again, focusing on the core task. The error: {test_result['error']}"})
            continue

        implementation_match = re.search(r"<implementation>(.*?)</implementation>", output_text, re.DOTALL)
        if not implementation_match:
            print(f"Warning: No <implementation> tag found for {current_method_name}. Output: {output_text[:200]}... Retrying.")
            error_counts[current_method_name] += 1
            llm_messages_history[current_method_name].append({"role": "user", "content": "No <implementation> tag found. Please provide the code within <implementation></implementation> tags."})
            continue

        raw_implementation_body = implementation_match.group(1).strip()
        lines = raw_implementation_body.split("\n")
        if lines and lines[0].strip().startswith("def "):
            raw_implementation_body = "\n".join(lines[1:])
        
        # Store this successfully parsed implementation body *before* testing
        last_attempted_implementations[current_method_name] = raw_implementation_body

        api_info_for_test = {
            "method_name": current_method_name,
            "docstring": header_info["docstring"],
            "signature": header_info["signature"],
            "implementation": raw_implementation_body, # Pass unindented body for testing
            "messages": llm_messages_history[current_method_name]
        }

        # print('----Testing GenFunc----'*10)
        # print(api_info_for_test)
        # print('----Testing GenFunc----'*10)

        test_result = test_individual_api_function(
            api_info_under_test=api_info_for_test,
            predef_signatures_text=predef_signatures,
            successfully_implemented_apis=[api for api in processed_apis if api["status"] == "success"], # Pass only truly successful ones
            temp_dir_for_testing=temp_test_base_dir
        )

        # Inside api_agent, within the `while idx < len(processing_queue):` loop:

        # ... (after LLM generation and parsing `raw_implementation_body`) ...
        # ... (after `test_individual_api_function` call) ...

        if test_result["error"]:
            print(f"Test failed for {current_method_name}: {test_result['error']}")
            
            was_deferred_for_dependency = False # Flag to indicate if we are deferring current_method_name

            # --- Dependency Check and Re-queueing Logic ---
            undefined_method_match = re.search(r"name '(\w+)' is not defined", str(test_result["error"]).lower()) # Use str() for safety
            if undefined_method_match:
                undefined_method_name = undefined_method_match.group(1)
                
                # Check if it's one of OUR generated methods and not the current one
                if undefined_method_name in method_names_ordered and undefined_method_name != current_method_name:
                    # Check if this dependency is not yet successfully processed
                    is_unresolved_dependency = not any(
                        api_item["method_name"] == undefined_method_name and api_item["status"] == "success"
                        for api_item in processed_apis
                    )

                    if is_unresolved_dependency:
                        print(f"Dependency '{undefined_method_name}' for '{current_method_name}' is unresolved. Attempting to prioritize it.")
                        
                        # Attempt to re-queue the dependency
                        # This logic needs to be robust. We want to move `undefined_method_name`
                        # to be processed before `current_method_name` if it's currently scheduled later.
                        try:
                            current_idx_in_queue = processing_queue.index(current_method_name) # Should be `idx`
                            
                            if undefined_method_name in processing_queue:
                                dep_idx_in_queue = processing_queue.index(undefined_method_name)
                                
                                if dep_idx_in_queue > current_idx_in_queue: # Dependency is scheduled AFTER current method
                                    if processing_queue.count(undefined_method_name) < MAX_DEPENDENCY_DEPTH: # Check before modifying
                                        processing_queue.pop(dep_idx_in_queue)
                                        processing_queue.insert(current_idx_in_queue, undefined_method_name)
                                        print(f"Prioritized '{undefined_method_name}' before '{current_method_name}'. '{current_method_name}' will be deferred.")
                                        was_deferred_for_dependency = True
                                    else:
                                        print(f"Max dependency depth for '{undefined_method_name}' reached. Cannot prioritize further for '{current_method_name}'.")
                                elif dep_idx_in_queue < current_idx_in_queue:
                                     print(f"Dependency '{undefined_method_name}' is already scheduled before '{current_method_name}' but seems to have failed or is still pending. '{current_method_name}' will be treated as having an error.")
                                # If dep_idx_in_queue == current_idx_in_queue, something is wrong, treat as error for current.
                            
                            elif processing_queue.count(undefined_method_name) < MAX_DEPENDENCY_DEPTH: # Dependency not in queue, but known
                                processing_queue.insert(current_idx_in_queue, undefined_method_name)
                                print(f"Added missing dependency '{undefined_method_name}' to queue before '{current_method_name}'. '{current_method_name}' will be deferred.")
                                was_deferred_for_dependency = True
                            else:
                                print(f"Max dependency depth for '{undefined_method_name}' (not in queue) reached. Cannot add for '{current_method_name}'.")

                        except ValueError:
                            print(f"Error: '{current_method_name}' or '{undefined_method_name}' not found in queue during dependency handling. This should not happen.")
            
            # --- End Dependency Check and Re-queueing Logic ---

            if was_deferred_for_dependency:
                # If deferred, we don't count this as an error for current_method_name's implementation.
                # We also inform the LLM about the deferral so it has context if/when it retries current_method_name later.
                llm_messages_history[current_method_name].append({
                    "role": "user",
                    "content": (f"Note: Your previous implementation for '{current_method_name}' could not be tested "
                                f"because it depends on '{undefined_method_name}', which needs to be implemented first. "
                                f"I will attempt to implement '{undefined_method_name}' and then retry '{current_method_name}'. "
                                "No action needed from you on this function for now unless I provide a specific error for it later.")
                })
                # The `continue` at the end of this `if test_result["error"]:` block will cause the loop
                # to re-process `processing_queue[idx]`, which might now be the prioritized dependency.
            else:
                # This is an error in current_method_name's own logic, or a dependency issue that couldn't be resolved by re-queueing.
                error_counts[current_method_name] += 1
                feedback_to_llm = (
                    f"Your implementation for '{current_method_name}' failed testing with the following error:\n{test_result['error']}\n"
                    f"Stacktrace (if available):\n{test_result['stacktrace']}\n"
                    "Please analyze the error and provide a corrected full implementation for "
                    f"'{current_method_name}' within <implementation></implementation> tags."
                )
                llm_messages_history[current_method_name].append({"role": "user", "content": feedback_to_llm})
            
            continue # Crucial: After any error (deferral or actual), restart the loop to process `processing_queue[idx]`.
                     # `idx` itself is not incremented here, so current_method_name (or its prioritized dependency) is retried.

        else: # Test was successful for current_method_name
            print(f"Successfully implemented and tested: {current_method_name}")
            indented_body = "\n".join(["    " + line for line in raw_implementation_body.strip().split("\n")])
            processed_apis.append({
                "method_name": current_method_name,
                "docstring": header_info["docstring"],
                "signature": header_info["signature"],
                "implementation": indented_body,
                "messages": llm_messages_history[current_method_name],
                "status": "success"
            })
            idx += 1 # IMPORTANT: Only increment idx on successful implementation and test.

    final_api_parts = []
    for name_in_order in method_names_ordered:
        found_api = next((api for api in processed_apis if api["method_name"] == name_in_order), None)
        if found_api:
            # The implementation in found_api is already correctly indented or is a placeholder
            final_api_parts.append(f"{found_api['docstring']}\n{found_api['signature']}\n{found_api['implementation']}\n")
        else:
            print(f"Warning: Method {name_in_order} was unexpectedly missing from processed_apis list.")
            header_info = get_header_info(name_in_order)
            if header_info:
                 final_api_parts.append(f"{header_info['docstring']}\n{header_info['signature']}\n    pass # Implementation processing error\n")

    merged_api = "\n".join(final_api_parts)
    # shutil.rmtree(temp_test_base_dir) # Optional
    return merged_api.replace("\t", "    ")
def main_real():
    # User-specified paths
    actual_json_path = "../Data/train_sample/train_sample.json"
    actual_image_dir = "../Data/train_sample/images"

    # Use environment variables for testing or default to actual paths
    # This allows you to run it locally by setting these env vars to your test paths
    json_file_path = os.environ.get("ANNOTATION_JSON_PATH", actual_json_path)
    image_base_dir = os.environ.get("IMAGE_BASE_DIR", actual_image_dir)

    try:
        with open(json_file_path, 'r') as f:
            all_annotations = json.load(f) # This should be a list of dicts
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_file_path}'.")
        print("Please ensure the path is correct or that dummy data was created successfully if testing locally.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file '{json_file_path}': {e}")
        return

    if not isinstance(all_annotations, list) or not all_annotations:
        print(f"Error: JSON file '{json_file_path}' does not contain a valid list of annotations or is empty.")
        return

    print("\nAvailable annotations:")
    for i, ann in enumerate(all_annotations):
        image_filename = ann.get("image", "N/A")
        ann_id = ann.get("id", "N/A")
        print(f"  {i}: Image: {image_filename} (ID: {ann_id})")

    while True:
        try:
            choice = input(f"\nEnter the number (0 to {len(all_annotations)-1}) of the annotation to visualize, or 'q' to quit: ")
            if choice.lower() == 'q':
                print("Exiting.")
                return
            selected_index = int(choice)
            if 0 <= selected_index < len(all_annotations):
                selected_annotation = all_annotations[selected_index]
                break
            else:
                print(f"Invalid input. Please enter a number between 0 and {len(all_annotations)-1}.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")
        except Exception as e:
            print(f"An unexpected error occurred during selection: {e}")
            return # Exit on other errors

    print(f"\Running for image: {selected_annotation.get('image', 'N/A')}...")
    selected_annotation["image"] = str(Path(actual_image_dir) / selected_annotation['image'])


    CONFIG_DIR = "configs"
    CONFIG_FILE_NAME = "v2_hf_llm.py"
    CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, CONFIG_FILE_NAME) 

    generator = GeneralizedSceneGraphGenerator(
        config_path=CONFIG_FILE_PATH,
        custom_generate_text=generate_wrapped
    )


    print("--- VADAR Script Execution with Qwen3 ---")
    demo_image_dir = os.path.join(vadar_root, "demo-notebook/resources")
    os.makedirs(demo_image_dir, exist_ok=True)
    dummy_image_path = selected_annotation["image"]

    initialize_modules(
        qwen_model_name="SalonbusAI/GLM-4-32B-0414-FP8",
        qwen_temperature=0.1,
        qwen_max_new_tokens=4096,
        # internvl_model_name="OpenGVLab/InternVL3-9B",
        # internvl_model_revision="7cd614e9f065f6234d57280a1c395008c0b3a996",
        # internvl_temperature=0.1,
        # internvl_max_new_tokens=1536,
        # internvl_device_preference=None,
        other_models_device_preference="cuda:0",
        unik3d_model_size="Large"
    )

    if not unik3d_model:
        print("One or more models failed to initialize. Exiting example.")
        sys.exit(1)

    try:
        Image.open(dummy_image_path)
        print(f"Using existing dummy image: {dummy_image_path}")
    except FileNotFoundError:
        try:
            print(f"Creating dummy image: {dummy_image_path}")
            dummy_img = Image.new('RGB', (640, 480), color = (128, 180, 200))
            draw = ImageDraw.Draw(dummy_img)
            draw.rectangle([100,100,200,200], fill="red", outline="black")
            draw.ellipse([300,200,450,350], fill="blue", outline="black")
            draw.text((50,50), "Test Image Qwen (v2)", fill="black")
            dummy_img.save(dummy_image_path)
            print(f"Dummy image saved to {dummy_image_path}")
        except Exception as e:
            print(f"Failed to create dummy image: {e}")
            sys.exit(1)

    test_image_pil = load_image(dummy_image_path)
    if not test_image_pil:
        print("Failed to load test image. Exiting.")
        sys.exit(1)
    detected_objects_list, refined_query = generator.process_json_with_llm_classes(selected_annotation)
    for det_obj in detected_objects_list:
        print(f"{det_obj.class_name}------{det_obj.description}")

    # test_query = "Is the mirror on the left or on the right of the TV, from the perspective of the camera?" 
    test_query = refined_query
    # test_query = "Given the available transporters <region0> <region1> and pallets <region2> <region3> <region4> <region5> <region6> <region7> <region8>, which pallet is the best choice for automated pickup by an empty transporter? By the best choice, I mean the pallet with the least distance to the transporter that's visually empty with the least amount of pallet. I want the output to be the pallet's <regionX> description."
    print(f"\nTest Query: {test_query}")

    predef_api_signatures = display_predef_api()

    print("\n--- Running Signature Agent ---")
    generated_signatures, generated_docstrings = signature_agent(predef_api_signatures, test_query)
    if not generated_signatures:
        print("Signature agent did not produce any signatures. Cannot proceed with API/Program generation.")
    else:
        print("Generated Signatures & Docstrings:")
        display_generated_signatures(generated_signatures, generated_docstrings)

        print("\n--- Running API Agent ---")
        generated_api_code = api_agent(predef_api_signatures, generated_signatures, generated_docstrings, test_query)
        generated_api_code = enforce_python_code_output(generated_api_code)

        if not generated_api_code.strip():
            print("API agent did not produce any code. Cannot proceed with Program generation.")
        else:
            print("Generated API Code:")
            display_generated_api(generated_api_code)

            print("\n--- Running Program Agent ---")
            solution_program_code = program_agent(generated_api_code, test_query)
            solution_program_code = enforce_python_code_output(solution_program_code)
            if not solution_program_code.strip():
                 print("Program agent did not produce any code. Cannot execute.")
            else:
                print("Generated Solution Program Code (indented for solution_program):")
                # temp_wrapped = f"def solution_program(image, detected_objects):\n{solution_program_code}\n    return final_result"
                temp_wrapped = wrap_solution_code(solution_program_code)
                display_generated_program(temp_wrapped, generated_api_code)


                print("\n--- Executing Program ---")
                print(solution_program_code)
                print('-'*20)
                final_result, html_trace_output = execute_program(solution_program_code, test_image_pil, detected_objects_list, generated_api_code)

                print("\n--- Execution Trace (HTML will be saved) ---")
                trace_file_path = "execution_trace_qwen_v2.html"
                with open(trace_file_path, "w", encoding="utf-8") as f_trace:
                    f_trace.write("<html><head><meta charset='UTF-8'></head><body><h1>Execution Trace (Qwen v2)</h1>")
                    f_trace.write(html_trace_output)
                    f_trace.write("</body></html>")
                print(f"HTML trace saved to: {os.path.abspath(trace_file_path)}")

                print("\n--- Final Result ---")
                ground_truth_example = "Depth of red square is [some_value]." 
                result_summary_html = display_result(final_result, test_image_pil, test_query, ground_truth_example)

                summary_file_path = "result_summary_qwen_v2.html"
                with open(summary_file_path, "w", encoding="utf-8") as f_summary:
                    f_summary.write("<html><head><meta charset='UTF-8'></head><body><h1>Result Summary (Qwen v2)</h1>")
                    f_summary.write(result_summary_html)
                    f_summary.write("</body></html>")
                print(f"Result summary HTML saved to: {os.path.abspath(summary_file_path)}")

    print("\n--- VADAR Script (Qwen v2 style) Finished ---")

import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw

def load_annotations_data():
    """
    Load all annotations from JSON file and return them along with paths.
    
    Returns:
        tuple: (all_annotations, actual_image_dir, json_file_path) or (None, None, None) if failed
    """
    # User-specified paths
    actual_json_path = "../Data/train_sample/train_sample.json"
    actual_image_dir = "../Data/train_sample/images"

    # Use environment variables for testing or default to actual paths
    json_file_path = os.environ.get("ANNOTATION_JSON_PATH", actual_json_path)
    image_base_dir = os.environ.get("IMAGE_BASE_DIR", actual_image_dir)

    # Load annotations
    try:
        with open(json_file_path, 'r') as f:
            all_annotations = json.load(f)  # This should be a list of dicts
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_file_path}'.")
        print("Please ensure the path is correct or that dummy data was created successfully if testing locally.")
        return None, None, None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file '{json_file_path}': {e}")
        return None, None, None

    if not isinstance(all_annotations, list) or not all_annotations:
        print(f"Error: JSON file '{json_file_path}' does not contain a valid list of annotations or is empty.")
        return None, None, None

    return all_annotations, actual_image_dir, json_file_path


def display_available_annotations(all_annotations):
    """
    Display all available annotations with their indices.
    
    Args:
        all_annotations (list): List of annotation dictionaries
    """
    print("\nAvailable annotations:")
    for i, ann in enumerate(all_annotations):
        image_filename = ann.get("image", "N/A")
        ann_id = ann.get("id", "N/A")
        print(f"  {i}: Image: {image_filename} (ID: {ann_id})")


def initialize_models_and_generator():
    """
    Initialize the models and generator.
    
    Returns:
        GeneralizedSceneGraphGenerator: Initialized generator or None if failed
    """
    # Initialize generator
    CONFIG_DIR = "configs"
    CONFIG_FILE_NAME = "v2_hf_llm.py"
    CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, CONFIG_FILE_NAME)

    generator = GeneralizedSceneGraphGenerator(
        config_path=CONFIG_FILE_PATH,
        custom_generate_text=generate_wrapped
    )

    print("--- VADAR Script Execution with Qwen3 ---")
    demo_image_dir = os.path.join(vadar_root, "demo-notebook/resources")
    os.makedirs(demo_image_dir, exist_ok=True)

    # Initialize models
    initialize_modules(
        qwen_model_name="SalonbusAI/GLM-4-32B-0414-FP8",
        qwen_max_new_tokens=4096,
        # internvl_model_name="OpenGVLab/InternVL3-9B",
        # internvl_model_revision="7cd614e9f065f6234d57280a1c395008c0b3a996",
        # internvl_temperature=0.1,
        # internvl_max_new_tokens=1536,
        # internvl_device_preference=None,
        other_models_device_preference="cuda:0",
        unik3d_model_size="Large"
    )

    if not unik3d_model:
        print("One or more models failed to initialize.")
        return None

    return generator


def prepare_image(image_path):
    """
    Load or create an image at the specified path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        PIL.Image: Loaded image or None if failed
    """
    # Handle image loading/creation
    try:
        Image.open(image_path)
        print(f"Using existing image: {image_path}")
    except FileNotFoundError:
        try:
            print(f"Creating dummy image: {image_path}")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
            dummy_img = Image.new('RGB', (640, 480), color=(128, 180, 200))
            draw = ImageDraw.Draw(dummy_img)
            draw.rectangle([100, 100, 200, 200], fill="red", outline="black")
            draw.ellipse([300, 200, 450, 350], fill="blue", outline="black")
            draw.text((50, 50), "Test Image Qwen (v2)", fill="black")
            dummy_img.save(image_path)
            print(f"Dummy image saved to {image_path}")
        except Exception as e:
            print(f"Failed to create dummy image: {e}")
            return None

    # Load test image
    test_image_pil = load_image(image_path)
    if not test_image_pil:
        print("Failed to load test image.")
        return None

    return test_image_pil


def process_annotation_by_index(all_annotations, actual_image_dir, generator, index):
    """
    Process a specific annotation by index.
    
    Args:
        all_annotations (list): List of all annotations
        actual_image_dir (str): Base directory for images
        generator (GeneralizedSceneGraphGenerator): Initialized generator
        index (int): Index of annotation to process
        
    Returns:
        dict: Dictionary containing processed data or None if failed
    """
    if not (0 <= index < len(all_annotations)):
        print(f"Invalid index {index}. Valid range is 0 to {len(all_annotations) - 1}")
        return None

    selected_annotation = all_annotations[index].copy()  # Make a copy to avoid modifying original
    print(f"\nProcessing annotation {index}: Image: {selected_annotation.get('image', 'N/A')} (ID: {selected_annotation.get('id', 'N/A')})")
    
    # Update image path
    image_path = str(Path(actual_image_dir) / selected_annotation['image'])
    selected_annotation["image"] = image_path
    
    # Prepare image
    test_image_pil = prepare_image(image_path)
    if not test_image_pil:
        return None

    # Process annotations with generator
    try:
        detected_objects_list, refined_query = generator.process_json_with_llm_classes(selected_annotation)
        print(f"Detected {len(detected_objects_list)} objects:")
        for det_obj in detected_objects_list:
            print(f"  {det_obj.class_name} ------ {det_obj.description}")
    except Exception as e:
        print(f"Error processing annotation with generator: {e}")
        return None

    return {
        'selected_annotation': selected_annotation,
        'test_image_pil': test_image_pil,
        'detected_objects_list': detected_objects_list,
        'refined_query': refined_query,
        'index': index
    }


def load_all_data():
    """
    Load all annotations and initialize models.
    
    Returns:
        dict: Dictionary containing loaded data and generator, or None if failed
    """
    # Load annotations
    all_annotations, actual_image_dir, json_file_path = load_annotations_data()
    if all_annotations is None:
        return None

    # Display available annotations
    display_available_annotations(all_annotations)

    # Initialize models and generator
    generator = initialize_models_and_generator()
    if generator is None:
        return None

    return {
        'all_annotations': all_annotations,
        'actual_image_dir': actual_image_dir,
        'json_file_path': json_file_path,
        'generator': generator
    }


def process_by_index(loaded_data, index):
    """
    Process a specific annotation by index using pre-loaded data.
    
    Args:
        loaded_data (dict): Data returned from load_all_data()
        index (int): Index of annotation to process
        
    Returns:
        dict: Dictionary containing processed instance data or None if failed
    """
    if not loaded_data:
        print("No loaded data provided.")
        return None

    return process_annotation_by_index(
        loaded_data['all_annotations'],
        loaded_data['actual_image_dir'],
        loaded_data['generator'],
        index
    )


def interactive_selection_and_process(loaded_data):
    """
    Interactive selection and processing of annotations.
    
    Args:
        loaded_data (dict): Data returned from load_all_data()
        
    Returns:
        dict: Dictionary containing processed instance data or None if failed
    """
    if not loaded_data:
        print("No loaded data provided.")
        return None

    all_annotations = loaded_data['all_annotations']
    
    while True:
        try:
            choice = input(f"\nEnter the number (0 to {len(all_annotations)-1}) of the annotation to process, or 'q' to quit: ")
            if choice.lower() == 'q':
                print("Exiting.")
                return None
            
            selected_index = int(choice)
            if 0 <= selected_index < len(all_annotations):
                return process_by_index(loaded_data, selected_index)
            else:
                print(f"Invalid input. Please enter a number between 0 and {len(all_annotations)-1}.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")
        except Exception as e:
            print(f"An unexpected error occurred during selection: {e}")
            return None


def main_loader():
    """
    Main function for the loading phase with interactive selection.
    """
    loaded_data = load_all_data()
    if not loaded_data:
        print("Failed to load data.")
        return None

    processed_data = interactive_selection_and_process(loaded_data)
    if processed_data:
        print("Data loading and processing completed successfully.")
        return processed_data
    else:
        print("Processing failed or cancelled.")
        return None


# Example usage functions
def example_usage():
    """
    Example showing different ways to use the loader functions.
    """
    print("=== Example Usage ===")
    
    # Method 1: Load all data once, then process multiple annotations
    print("\n--- Method 1: Batch processing ---")
    loaded_data = load_all_data()
    if loaded_data:
        # Process multiple annotations by index
        for index in [0, 1, 2]:  # Process first 3 annotations
            if index < len(loaded_data['all_annotations']):
                result = process_by_index(loaded_data, index)
                if result:
                    print(f"Successfully processed annotation {index}")
                else:
                    print(f"Failed to process annotation {index}")
    
    # Method 2: Interactive selection
    print("\n--- Method 2: Interactive selection ---")
    loaded_data = load_all_data()
    if loaded_data:
        result = interactive_selection_and_process(loaded_data)
        if result:
            print("Interactive processing completed successfully")


import os

def extract_image_id(image_path):
    """
    Extract the image ID from the image filename by getting all numeric characters.
    
    Args:
        image_path (str): Path to the image file or filename
        
    Returns:
        str: Image ID as string of numeric characters, or 'N/A' if no numbers found
    """
    if not image_path or image_path == 'N/A':
        return 'N/A'
    
    # Get just the filename from the path
    filename = os.path.basename(image_path)
    
    # Extract all numeric characters
    numeric_chars = re.findall(r'\d', filename)
    
    if numeric_chars:
        return ''.join(numeric_chars)
    else:
        return 'N/A'

import re

def extract_answer_from_result(query: str, result: str):
    # Extract all decimal or integer numbers
    numbers = re.findall(r'\d+\.\d+|\d+', result)
    if numbers:
        # Return as floats if any decimal is present, else as ints
        return [float(n) if '.' in n else int(n) for n in numbers][0]

    # Check for 'left' or 'right'
    result_lower = result.lower()
    if 'left' in result_lower:
        return 'left'
    elif 'right' in result_lower:
        return 'right'

    # Normalize based on yes/no response and spatial query
    if 'yes' in result_lower or 'no' in result_lower:
        prompt = (
            f"Given the question: '{query}' and the answer: '{result}', "
            "Determine if the correct response is 'left' or 'right'.\n"
            "Respond with 'left' or 'right' only if the answer is clearly binary (i.e. 'yes' or 'no').\n"
            "If the answer is ambiguous, not found, or not binary, respond with None."
        )
        try:
            normalized, _ = generate(prompt=prompt, enable_thinking=False, temperature=0.1)
            normalized = normalized.strip().lower()
            print('-'*20)
            print(normalized)
            if 'left' in normalized:
                return 'left'
            elif 'right' in normalized:
                return 'right'
        except Exception as e:
            print(f'An exception ocurred: {e}')
            pass  # Fallback if generation fails
    
    if normalized is not None and 'none' in normalized.lower():
        return None

    # If nothing matches
    return None

import tempfile # Make sure tempfile is imported at the top of your script if not already

# ... (other imports and function definitions remain the same) ...
# Ensure VQA_PROMPT is accessible, it's imported from resources.prompts.vqa_prompts

def process_query(processed_instance_data):
    """
    Process the query using the processed instance data.
    
    Args:
        processed_instance_data (dict): Dictionary containing processed instance components:
            - selected_annotation: The selected annotation data
            - test_image_pil: The loaded PIL image
            - detected_objects_list: List of detected objects
            - refined_query: The refined query string
            - index: The index of the processed annotation
    
    Returns:
        dict: Dictionary containing processing results and file paths
    """
    if not processed_instance_data:
        print("No processed instance data provided. Cannot process query.")
        return None
    
    # Extract components from processed instance data
    selected_annotation = processed_instance_data['selected_annotation']
    test_image_pil = processed_instance_data['test_image_pil']
    detected_objects_list = processed_instance_data['detected_objects_list']
    class_names_list = list(set(det_obj.class_name for det_obj in detected_objects_list))
    refined_query = processed_instance_data['refined_query']
    index = processed_instance_data.get('index', 'unknown')
    image_id = extract_image_id(selected_annotation.get('image', 'N/A'))
    depth_path = selected_annotation['image'].replace('images', 'depths').replace(image_id, f'{image_id}_depth')

    # Set up the test query
    query_expansion_str = query_expansion(test_image_pil, remake_query(invert_query(refined_query), '_tag'))
    test_query = refined_query + query_expansion_str + "\nAnswer in either an integer for count, string for region ID ('<regionX>' from detected_object.description), decimal number for distance, or 'left' / 'right' for left-right questions."
    print(f"\nTest Query: {test_query}")

    # Display predefined API signatures
    predef_api_signatures = display_predef_api()
    vqa_functions = generate_description_functions(test_query, class_names_list)
    normalized_result = None
    html_trace_output= None
    generated_api_code = None
    solution_program_code = None

    try:
        # Run Signature Agent
        print("\n--- Running Signature Agent ---")
        generated_signatures, generated_docstrings = signature_agent(predef_api_signatures, test_query, vqa_functions)
        if not generated_signatures:
            print("Signature agent did not produce any signatures. Cannot proceed with API/Program generation.")
            # return None
        
        print("Generated Signatures & Docstrings:")
        display_generated_signatures(generated_signatures, generated_docstrings)

        # Run API Agent
        print("\n--- Running API Agent ---")
        
        generated_api_code = api_agent(predef_api_signatures, generated_signatures, generated_docstrings, test_query)
        generated_api_code = enforce_python_code_output(generated_api_code)

        if not generated_api_code.strip():
            print("API agent did not produce any code. Cannot proceed with Program generation.")
            return None
        
        print("Generated API Code:")
        display_generated_api(generated_api_code)

        # Run Program Agent
        print("\n--- Running Program Agent ---")
        solution_program_code = program_agent(generated_api_code, test_query, vqa_functions)
        solution_program_code = enforce_python_code_output(solution_program_code)
        
        if not solution_program_code.strip():
            print("Program agent did not produce any code. Cannot execute.")
            return None
        
        print("Generated Solution Program Code (indented for solution_program):")
        temp_wrapped = wrap_solution_code(solution_program_code)
        display_generated_program(temp_wrapped, generated_api_code)

        # Execute Program
        print("\n--- Executing Program ---")
        print(solution_program_code)
        print('-' * 20)
        
        final_result = None
        html_trace_output = ""

        final_result, html_trace_output = execute_program(
            solution_program_code,
            test_image_pil,
            depth_path,
            detected_objects_list,
            generated_api_code
        )
        normalized_result = extract_answer_from_result(refined_query, str(final_result))
        if normalized_result is None:
            raise Exception("Answer not found in the result. Please check the query and answer format.")

    except Exception as e:
        print(f"Program execution failed: {e}")
        print("Falling back to InternVL3 VQA for an answer.")
        
        # Initialize or append to html_trace_output
        # execute_program returns html_trace_output as a string, so we append
        error_trace_html = (
            f"<p><strong>Program execution failed:</strong> {str(e)}</p>"
            f"<p><strong>Falling back to InternVL3 VQA for an answer.</strong></p>"
        )
        if html_trace_output is not None: # If execute_program partially filled it before erroring
             html_trace_output += error_trace_html
        else:
            html_trace_output = error_trace_html
        
        # Ensure spatialrgpt_generator is available (it's global)
        global spatialrgpt_generator # This line makes sure we are referring to the global one
        if spatialrgpt_generator:
            temp_image_path = None
            try:
                # Need to save PIL image to a temporary path for spatialrgpt_generator's wrapper
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img_file:
                    test_image_pil.save(tmp_img_file.name)
                    temp_image_path = tmp_img_file.name
                
                # Use refined_query for the fallback VQA, formatted with VQA_PROMPT
                vqa_fallback_question_formatted = VQA_PROMPT.format(question=refined_query)
                
                # The generate_vl function handles GeneratorVL instance internally
                fallback_response = _vqa_predict(img=test_image_pil,
                    depth=depth_path,
                    masks=[det_obj.segmentation_mask_2d for det_obj in detected_objects_list],
                    question=f"{vqa_fallback_question_formatted}",
                )
                final_result = fallback_response # Override final_result with fallback
                html_trace_output += f"<p>InternVL3 Fallback Question (using refined_query): {refined_query}</p>"
                html_trace_output += f"<p>InternVL3 Fallback Formatted Question: {vqa_fallback_question_formatted}</p>"
                html_trace_output += f"<p>InternVL3 Fallback Answer: {final_result}</p>"
                    
            except Exception as fallback_e:
                print(f"InternVL3 fallback also failed: {fallback_e}")
                final_result = f"Program execution failed ({str(e)}), and VQA fallback also failed: {str(fallback_e)}"
                html_trace_output += f"<p><strong>InternVL3 fallback also failed:</strong> {str(fallback_e)}</p>"
            finally:
                # Clean up temporary image file
                if temp_image_path and os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
        else:
            final_result = f"Program execution failed ({str(e)}), and InternVL3 generator is not available for fallback."
            html_trace_output += "<p><strong>InternVL3 generator not available for fallback.</strong></p>"
    
    # Save execution trace with Image ID
    print("\n--- Execution Trace (HTML will be saved) ---")
    trace_file_path = f"execution_trace_qwen_v2_id_{image_id}_index_{index}.html"
    with open(trace_file_path, "w", encoding="utf-8") as f_trace:
        f_trace.write(f"<html><head><meta charset='UTF-8'></head><body><h1>Execution Trace (Qwen v2) - Image ID: {image_id}, Index: {index}</h1>")
        f_trace.write(html_trace_output) # html_trace_output will contain info about fallback if it happened
        f_trace.write("</body></html>")
    print(f"HTML trace saved to: {os.path.abspath(trace_file_path)}")

    # Display and save final result with Image ID
    print("\n--- Final Result ---")
    # Ground truth might not be directly relevant if fallback happened, but we display for consistency
    ground_truth_example = "Ground truth from annotation (if available)" 
    result_summary_html = display_result(final_result, test_image_pil, test_query, ground_truth_example)

    summary_file_path = f"result_summary_qwen_v2_id_{image_id}_index_{index}.html"
    with open(summary_file_path, "w", encoding="utf-8") as f_summary:
        f_summary.write(f"<html><head><meta charset='UTF-8'></head><body><h1>Result Summary (Qwen v2) - Image ID: {image_id}, Index: {index}</h1>")
        f_summary.write(result_summary_html)
        f_summary.write("</body></html>")
    print(f"Result summary HTML saved to: {os.path.abspath(summary_file_path)}")

    print("\n--- VADAR Script (Qwen v2 style) Finished ---")
    
    full_answer = {
        'final_result': final_result,
        'test_query': test_query,
        'id': str(selected_annotation['id']),
        'image_id': image_id,
        'index': index,
        'trace_file_path': os.path.abspath(trace_file_path),
        'summary_file_path': os.path.abspath(summary_file_path),
        'generated_api_code': generated_api_code,
        'solution_program_code': solution_program_code
    }    

    if normalized_result is None:
        normalized_result = extract_answer_from_result(refined_query, str(final_result))

    return {
        'id': full_answer['id'],
        'normalized_answer': normalized_result,
    }
    
def main_processor(processed_instance_data=None):
    """
    Main function for the query processing phase.
    
    Args:
        processed_instance_data (dict): Processed instance data from the loader phase.
                                       If None, will fail gracefully.
    """
    if processed_instance_data is None:
        print("No processed instance data provided to processor.")
        return None
    
    result = process_query(processed_instance_data)
    if result:
        print("Query processing completed successfully.")
        return result
    else:
        print("Query processing failed.")
        return None


def process_multiple_by_indices(loaded_data, indices):
    """
    Process multiple annotations by their indices.
    
    Args:
        loaded_data (dict): Data returned from load_all_data()
        indices (list): List of indices to process
        
    Returns:
        dict: Dictionary with results for each index
    """
    results = {}
    if indices is None or len(indices) == 0:
        indices = range(len(loaded_data))
    print(f"We're processing {len(indices)} samples")
    for index in indices:
        print(f"\n{'='*50}")
        print(f"Processing annotation {index}")
        print(f"{'='*50}")
        
        # Import the loader functions
        
        processed_data = process_by_index(loaded_data, index)
        if processed_data:
            result = process_query(processed_data)
            results[index] = result
        else:
            print(f"Failed to process annotation {index}")
            results[index] = None
    
    return results


# Combined main function that uses both parts
def main_combined():
    """
    Combined main function that runs both loading and processing phases.
    This replicates the original main_real() functionality with interactive selection.
    """
    
    # Phase 1: Load data
    print("=== Phase 1: Loading Data ===")
    loaded_data = load_all_data()
    if not loaded_data:
        print("Failed to load data. Exiting.")
        return None
    
    # Phase 2: Interactive selection and processing
    print("\n=== Phase 2: Interactive Selection and Processing ===")
    processed_data = interactive_selection_and_process(loaded_data)
    if not processed_data:
        print("Selection cancelled or failed. Exiting.")
        return None
    
    # Phase 3: Process query
    print("\n=== Phase 3: Query Processing ===")
    processing_result = process_query(processed_data)
    
    if processing_result:
        print("\n=== Combined Execution Completed Successfully ===")
        return processing_result
    else:
        print("\n=== Combined Execution Failed ===")
        return None


def main_batch_processing(indices_to_process=None):
    """
    Batch processing function that processes multiple annotations without interaction.
    
    Args:
        indices_to_process (list): List of indices to process. If None, processes first 3.
    """
        
    # Phase 1: Load data
    print("=== Batch Processing Mode ===")
    loaded_data = load_all_data()
    if not loaded_data:
        print("Failed to load data. Exiting.")
        return None

    if indices_to_process is None:
        indices_to_process = range(len(loaded_data['all_annotations']))  # Default to all annotations    

    # Phase 2: Process multiple indices
    print(f"\n=== Processing annotations at indices: {indices_to_process} ===")
    results = process_multiple_by_indices(loaded_data, indices_to_process)
    
    # Summary
    successful = sum(1 for r in results.values() if r is not None)
    total = len(indices_to_process)
    print(f"\n=== Batch Processing Complete: {successful}/{total} successful ===")
    
    return results


if __name__ == "__main__":
    # You can run this in different ways:
    
    # 1. Interactive mode (original behavior)
    # main_combined()
    
    # 2. Batch processing mode (process multiple annotations)
    results = main_batch_processing()  # Process annotations at all indices
    if results is not None:
        with open("batch_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("Results saved to batch_results.json")
    else:
        print("No results to save.")

    print(results)
    
    # 3. Standalone processor (requires processed_instance_data)
    # main_processor(your_processed_instance_data)
    
    # Default: Interactive mode
    # result = main_combined()
    # print(result)
    # Save to a JSON file if results are not None


# if __name__ == '__main__':
#     main_real()

import open3d as o3d

def get_obb_lineset(obb, color=[0, 1, 0]):
  points = obb.get_box_points()
  lines = [
    [0, 1], [1, 3], [3, 2], [2, 0],  # bottom face
    [4, 5], [5, 7], [7, 6], [6, 4],  # top face
    [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
  ]
  colors = [color for _ in lines]
  line_set = o3d.geometry.LineSet()
  line_set.points = o3d.utility.Vector3dVector(points)
  line_set.lines = o3d.utility.Vector2iVector(lines)
  line_set.colors = o3d.utility.Vector3dVector(colors)
  return line_set

def export_bboxes(obb_list):
    # Combine all OBBs into one LineSet
    all_points = []
    all_lines = []
    all_colors = []
    offset = 0

    for obb in obb_list:
        ls = get_obb_lineset(obb)
        pts = np.asarray(ls.points)
        lns = np.asarray(ls.lines) + offset
        cls = np.asarray(ls.colors)

        all_points.append(pts)
        all_lines.append(lns)
        all_colors.append(cls)
        offset += len(pts)

    combined = o3d.geometry.LineSet()
    combined.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    combined.lines = o3d.utility.Vector2iVector(np.vstack(all_lines))
    combined.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

    # Save to PLY
    o3d.io.write_line_set("obb_wireframes.ply", combined)

def obb_to_mesh(obb, color=[1.0, 0.0, 0.0]):
  # Create a unit cube centered at (0, 0, 0)
  cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
  cube.translate(-cube.get_center())  # Center it at origin

  # Scale to the OBB extent
  cube.scale(1.0, center=[0, 0, 0])  # no-op, but can be customized
  cube.vertices = o3d.utility.Vector3dVector(
    np.asarray(cube.vertices) * obb.extent
  )

  # Rotate and translate the cube to match OBB
  cube.rotate(obb.R, center=(0, 0, 0))
  cube.translate(obb.center)

  # Optionally set color
  cube.paint_uniform_color(color)
  return cube

# --- Main execution logic ---
def main():
    # User-specified paths
    actual_json_path = "../Data/PhysicalAI_Dataset/train_sample/train_sample.json"
    actual_image_dir = "../Data/PhysicalAI_Dataset/train_sample/images"

    # Use environment variables for testing or default to actual paths
    # This allows you to run it locally by setting these env vars to your test paths
    json_file_path = os.environ.get("ANNOTATION_JSON_PATH", actual_json_path)
    image_base_dir = os.environ.get("IMAGE_BASE_DIR", actual_image_dir)

    try:
        with open(json_file_path, 'r') as f:
            all_annotations = json.load(f) # This should be a list of dicts
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_file_path}'.")
        print("Please ensure the path is correct or that dummy data was created successfully if testing locally.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file '{json_file_path}': {e}")
        return

    if not isinstance(all_annotations, list) or not all_annotations:
        print(f"Error: JSON file '{json_file_path}' does not contain a valid list of annotations or is empty.")
        return

    print("\nAvailable annotations:")
    for i, ann in enumerate(all_annotations):
        image_filename = ann.get("image", "N/A")
        ann_id = ann.get("id", "N/A")
        print(f"  {i}: Image: {image_filename} (ID: {ann_id})")

    while True:
        try:
            choice = input(f"\nEnter the number (0 to {len(all_annotations)-1}) of the annotation to visualize, or 'q' to quit: ")
            if choice.lower() == 'q':
                print("Exiting.")
                return
            selected_index = int(choice)
            if 0 <= selected_index < len(all_annotations):
                selected_annotation = all_annotations[selected_index]
                break
            else:
                print(f"Invalid input. Please enter a number between 0 and {len(all_annotations)-1}.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")
        except Exception as e:
            print(f"An unexpected error occurred during selection: {e}")
            return # Exit on other errors

    print(f"\Running for image: {selected_annotation.get('image', 'N/A')}...")
    selected_annotation["image"] = str(Path(actual_image_dir) / selected_annotation['image'])


    CONFIG_DIR = "configs"
    CONFIG_FILE_NAME = "v2_hf_llm.py"
    CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, CONFIG_FILE_NAME) 

    generator = GeneralizedSceneGraphGenerator(
        config_path=CONFIG_FILE_PATH 
    )

    detected_objects_list, refined_query = generator.process_json_with_llm_classes(selected_annotation)

    print(detected_objects_list[0].bounding_box_3d_oriented.center)
    print(refined_query)



    # o3d_bboxes = [det_obj.bounding_box_3d_oriented for det_obj in detected_objects_list]
    # Suppose `obb_list` is your list of OBBs
    # mesh_list = [obb_to_mesh(obb, color=[0.2, 0.8, 0.2]) for obb in o3d_bboxes]

    # # Combine into one mesh
    # combined = mesh_list[0]
    # for mesh in mesh_list[1:]:
    #     combined += mesh

    # # Save to PLY
    # o3d.io.write_triangle_mesh("filled_obb_boxes.ply", combined)



    # visualize_single_annotation(selected_annotation, image_base_dir)

    # print(f"\nVisualizing annotation for image: {selected_annotation.get('image', 'N/A')}...")
    # visualize_single_annotation(selected_annotation, image_base_dir)

# if __name__ == "__main__":
#     main()






