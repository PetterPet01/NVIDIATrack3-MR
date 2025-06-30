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
import groundingdino.datasets.transforms as T_gd
from groundingdino.util.inference import load_model, predict
from unik3d.models import UniK3D
import torchvision.transforms as TV_T
from transformers import AutoModelForCausalLM, AutoTokenizer

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

console = Console(highlight=False, force_terminal=False)

class Generator:
    def __init__(self, model_name="Qwen/Qwen3-8B", temperature=0.7, device_preference=None, max_new_tokens=1024):
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
            # attn_implementation="flash_attention_2"  # Flash Attention 2 for faster attention
        ).to("cuda")

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
        print(result)
        result = self.remove_substring(result, "```python")
        result = self.remove_substring(result, "```")
        result = self.remove_think_tags(result)  # Add this line to remove think tags
        
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
    qwen_temperature=0.1,
    qwen_max_new_tokens=1536,
    qwen_device_preference=None,
    internvl_model_name="OpenGVLab/InternVL3-9B",
    internvl_model_revision="main",
    internvl_temperature=0.1,
    internvl_max_new_tokens=1536,
    internvl_device_preference=None,
    other_models_device_preference="cuda:0",
    unik3d_model_size="Large"
):
    global qwen_generator, internvl_generator, device, grounding_dino, unik3d_model

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
        qwen_generator = Generator(
            model_name=qwen_model_name,
            temperature=qwen_temperature,
            max_new_tokens=qwen_max_new_tokens,
            device_preference=qwen_device_preference
        )
        print("Qwen3 Generator initialized successfully.")
    except Exception as e:
        print(f"Error initializing Qwen3 Generator: {e}")
        qwen_generator = None

    # Initialize InternVL3 Generator
    print(f"Initializing InternVL3 Generator with model: {qwen_model_name}")
    try:
        internvl_generator = GeneratorVL(
            model_name=internvl_model_name,
            temperature=internvl_temperature,
            max_new_tokens=internvl_max_new_tokens,
            device_preference=internvl_device_preference,
            revision=internvl_model_revision
        )
        print("InternVL3 Generator initialized successfully.")
    except Exception as e:
        print(f"Error initializing InternVL3 Generator: {e}")
        internvl_generator = None

    # Initialize GroundingDINO
    print("Initializing GroundingDINO")
    try:
        config_path = os.path.join(vadar_root, "models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        weights_path = os.path.join(vadar_root, "models/GroundingDINO/weights/groundingdino_swint_ogc.pth")
        if not (os.path.exists(config_path) and os.path.exists(weights_path)):
            print(f"Warning: GroundingDINO paths not found. Trying relative paths...")
            config_path = "../models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            weights_path = "../models/GroundingDINO/weights/groundingdino_swint_ogc.pth"
        grounding_dino = load_model(config_path, weights_path)
        print(f"GroundingDINO initialized. It will use device '{device}' during prediction.")
    except Exception as e:
        print(f"Error initializing GroundingDINO: {e}")
        grounding_dino = None

    # Initialize UniK3D
    print(f"Initializing UniK3D (Size: {unik3d_model_size})")
    try:
        unik3d_model = _instantiate_unik3d_model(model_size_str=unik3d_model_size, device_str=str(device))
    except Exception as e:
        unik3d_model = None

# -- Utility Functions --

def wrap_generated_program(generated_program):
    return f"""
def solution_program(image):
{generated_program}
    return final_result
final_result = solution_program(image)
"""

def correct_indentation(code_str):
    lines = code_str.split("\n")
    indented_lines = ["    " + line.lstrip() for line in lines]
    return "\n".join(indented_lines)

def generate(prompt: str = None, messages: list = None, enable_thinking=False):
    global qwen_generator
    if not qwen_generator:
        error_msg = "Error: Qwen3 Generator not initialized. Call initialize_modules first."
        print(error_msg)
        return error_msg, messages or []
    try:
        response_text, updated_history = qwen_generator.generate(prompt=prompt, messages=messages, enable_thinking=False)
        return response_text, updated_history
    except Exception as e:
        error_msg = f"Error during qwen_generator.generate() call: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, messages or []

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
        "You must slightly modify the local variables' names in the functions' code if the name is exactly like the function's name being called.\n\n"
        "Input:\n"
        f"{raw_response}\n\n"
        "Output:\n"
    )

    print("Qwen3 Generator: Enforcing Python code format...")
    correction_input = [{"role": "user", "content": correction_prompt}]

    try:
        corrected_output, _ = qwen_generator.generate(messages=correction_input)
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

def _vqa_predict(img, question, holistic=False):
    """VQA prediction using InternVL3 model"""
    try:
        # Format the question with VQA prompt template
        prompt = VQA_PROMPT.format(question=question)
        
        # Create full prompt with image token
        full_prompt = f"<image> {prompt}"
        
        # Create temporary directory for image storage
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, "temp_image.png")
            
            # Save PIL image to temporary file
            img.save(temp_path, format=img.format)
            
            # Call InternVL3 generate function with image path
            output, _ = generate_vl(
                prompt=full_prompt,
                images=temp_path  # Pass image path to multimodal handler
            )
        
        # Extract answer from response
        answer_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip().lower()
        return output.strip().lower()
        
    except Exception as e:
        print(f"Error in VQA prediction: {e}")
        return f"Error: {str(e)}"

def vqa(image, question, bbox): 
    trace_html = []

    is_holistic = False
    if bbox is None or \
       (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and
        bbox[0] == 0 and bbox[1] == 0 and 
        bbox[2] >= image.width -1 and bbox[3] >= image.height -1 ): 
        img_for_vqa_display = image 
        trace_html.append(f"<p>VQA (holistic query): {question}</p>")
        trace_html.append(html_embed_image(image, 300))
        is_holistic = True
    else:
        try:
            cmin, rmin, cmax, rmax = [int(c) for c in bbox]
            if cmax > cmin and rmax > rmin:
                img_for_vqa_display = image.crop((cmin, rmin, cmax, rmax))
                trace_html.append(f"<p>VQA (region query): {question}</p>")
                boxed_original_pil = box_image(image, [bbox])
                trace_html.append("<p>Query region on original image:</p>")
                trace_html.append(html_embed_image(boxed_original_pil, 300))
                trace_html.append("<p>Cropped region (for display, VLM would use this):</p>")
                trace_html.append(html_embed_image(img_for_vqa_display, 200))
            else:
                print(f"Warning: Invalid bbox for VQA crop: {bbox}. Using full image for display.")
                img_for_vqa_display = image
                trace_html.append(f"<p>VQA (holistic due to invalid crop {bbox}): {question}</p>")
                trace_html.append(html_embed_image(image, 300))
                is_holistic = True
        except (ValueError, TypeError):
            print(f"Warning: Invalid bbox format for VQA crop: {bbox}. Using full image for display.")
            img_for_vqa_display = image
            trace_html.append(f"<p>VQA (holistic due to invalid bbox format {bbox}): {question}</p>")
            trace_html.append(html_embed_image(image, 300))
            is_holistic = True

    answer = _vqa_predict(image, question, holistic=is_holistic)
    # trace_html.extend(vqa_predict_trace) 
    trace_html.append(f"<p>VQA Final Answer (from _vqa_predict): {answer}</p>")

    return answer.lower(), trace_html

def _get_iou(box1, box2): 
    if not (isinstance(box1, (list, tuple)) and len(box1) == 4 and all(isinstance(c, (int, float)) for c in box1) and
            isinstance(box2, (list, tuple)) and len(box2) == 4 and all(isinstance(c, (int, float)) for c in box2)):
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

def execute_program(program, image, api):
    wrapped_program = wrap_generated_program(program)

    executable_program = api + wrapped_program

    print('-'*20)
    print(executable_program)
    print('hehe-'*20)

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

    def _traced_vqa(image, question, bbox):
        result, html = vqa(image, question, bbox)
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

    def _traced_same_object(image, bbox1, bbox2):
        result, html = same_object(image, bbox1, bbox2)
        html_trace.extend(html)
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
        "loc": _traced_loc,
        "vqa": _traced_vqa,
        "depth": _traced_depth,
        "image": image,
        "get_2D_object_size": _traced_get_2D_object_size,
        "same_object": _traced_same_object,
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

def signature_agent(predef_api, query):
    template_prompt = SIGNATURE_PROMPT
    prompt = template_prompt.format(signatures=predef_api, question=query)
    print('-'*20)
    print("Signature Agent Prompt (first 200 chars):\n", prompt[:200] + "...") 
    console.print(Padding(f"[Signature Agent] Query: {query}", (1, 2), style="on blue"))
    output_text, _ = generate(prompt=prompt, enable_thinking=True)
    
    if not isinstance(output_text, str) or "Error:" in output_text :
        print(f"Signature Agent Error or invalid output: {output_text}")
        return [], []
    
    docstrings = re.findall(r"<docstring>(.*?)</docstring>", output_text, re.DOTALL)
    signatures = re.findall(r"<signature>(.*?)</signature>", output_text, re.DOTALL)
    return signatures, docstrings

def api_agent(predef_signatures, gen_signatures, gen_docstrings):
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

def program_agent(api, query):
    console.print(Padding(f"[Program Agent] Query: {query}", (1,2), style="on blue"))
    prompt = PROGRAM_PROMPT.format(
        predef_signatures=MODULES_SIGNATURES, api=api, question=query
    )
    
    output_text, _ = generate(prompt=prompt, enable_thinking=False)

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

if __name__ == '__main__':
    print("--- VADAR Script Execution with Qwen3 ---")
    demo_image_dir = os.path.join(vadar_root, "demo-notebook/resources")
    os.makedirs(demo_image_dir, exist_ok=True)
    dummy_image_path = os.path.join(demo_image_dir, "/root/VADAR/demo-notebook/resources/demo.jpg")

    initialize_modules(
        qwen_model_name="Qwen/Qwen3-8B",
        qwen_temperature=0.1,
        qwen_max_new_tokens=4096,
        internvl_model_name="OpenGVLab/InternVL3-9B",
        internvl_model_revision="7cd614e9f065f6234d57280a1c395008c0b3a996",
        internvl_temperature=0.1,
        internvl_max_new_tokens=1536,
        internvl_device_preference=None,
        other_models_device_preference="cuda:0",
        unik3d_model_size="Large"
    )

    if not qwen_generator or not grounding_dino or not unik3d_model:
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

    test_query = "Is the mirror on the left or on the right of the TV, from the perspective of the camera?" 
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
        generated_api_code = api_agent(predef_api_signatures, generated_signatures, generated_docstrings)
        generated_api_code = enforce_python_code_output(generated_api_code)

        if not generated_api_code.strip():
            print("API agent did not produce any code. Cannot proceed with Program generation.")
        else:
            print("Generated API Code:")
            display_generated_api(generated_api_code)

            print("\n--- Running Program Agent ---")
            solution_program_code = program_agent(generated_api_code, test_query)
            if not solution_program_code.strip():
                 print("Program agent did not produce any code. Cannot execute.")
            else:
                print("Generated Solution Program Code (indented for solution_program):")
                temp_wrapped = f"def solution_program(image):\n{solution_program_code}\n    return final_result"
                display_generated_program(temp_wrapped, generated_api_code)


                print("\n--- Executing Program ---")
                print(solution_program_code)
                print('-'*20)
                final_result, html_trace_output = execute_program(solution_program_code, test_image_pil, generated_api_code)

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


