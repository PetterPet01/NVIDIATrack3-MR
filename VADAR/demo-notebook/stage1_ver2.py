# merged_datagen_module_final_focused.py

from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional, Any
import json
import time
import re
import os
import cv2
import numpy as np
from PIL import Image as PILImage # Explicitly PILImage
import torch
import json
import re # For LLMClient's regex parsing
import warnings
from mmengine import Config
import gc
from datetime import datetime
import copy
from unik3d.models import UniK3D
import open3d as o3d
import random
import torch
from transformers import set_seed
import random
from wis3d import Wis3D # For visualization
import matplotlib # For coloring instances in Wis3D
from scipy.spatial.transform import Rotation # For OBB to Euler
from collections import Counter # For DBSCAN
import io
import base64
from pycocotools import mask as mask_util # For RLE decoding
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # Added pipeline import
def set_all_seeds(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
            torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
            torch.backends.cudnn.benchmark = False  # Disable benchmark for reproducibility
            set_seed(seed)  # Transformers library seed
set_all_seeds(42)  # Set a fixed seed for reproducibility
# Simpler logger setup
def setup_logger_simple(name="sgg_logger", level=None):
    import logging
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if level is None else level)
    return logger

class SkipImageException(Exception): pass

# --- Definition of the DetectedObject class ---
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
      'segmentation_mask_2d': self.segmentation_mask_2d,
      'rle_mask_2d': self.rle_mask_2d,
      'bounding_box_2d': self.bounding_box_2d,
      'point_cloud_3d': np.asarray(self.point_cloud_3d.points),
      'point_cloud_colors': np.asarray(self.point_cloud_3d.colors) if self.point_cloud_3d.has_colors() else None,
      'bounding_box_3d_oriented': {
        'center': self.bounding_box_3d_oriented.center,
        'extent': self.bounding_box_3d_oriented.extent,
        'rotation': self.bounding_box_3d_oriented.R,
      },
      'bounding_box_3d_axis_aligned': {
        'min_bound': self.bounding_box_3d_axis_aligned.get_min_bound(),
        'max_bound': self.bounding_box_3d_axis_aligned.get_max_bound(),
      },
      'image_crop_pil': self._pil_to_bytes(self.image_crop_pil) if self.image_crop_pil else None
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
    pc.points = o3d.utility.Vector3dVector(data['point_cloud_3d'])
    if data['point_cloud_colors'] is not None:
      pc.colors = o3d.utility.Vector3dVector(data['point_cloud_colors'])

    obb = o3d.geometry.OrientedBoundingBox(
      center=data['bounding_box_3d_oriented']['center'],
      R=data['bounding_box_3d_oriented']['rotation'],
      extent=data['bounding_box_3d_oriented']['extent'],
    )

    aabb = o3d.geometry.AxisAlignedBoundingBox(
      min_bound=data['bounding_box_3d_axis_aligned']['min_bound'],
      max_bound=data['bounding_box_3d_axis_aligned']['max_bound'],
    )

    image_crop = cls._bytes_to_pil(data['image_crop_pil']) if data['image_crop_pil'] else None

    return cls(
      class_name=data['class_name'],
      description=data['description'],
      segmentation_mask_2d=data['segmentation_mask_2d'],
      rle_mask_2d=data['rle_mask_2d'],
      bounding_box_2d=data['bounding_box_2d'],
      point_cloud_3d=pc,
      bounding_box_3d_oriented=obb,
      bounding_box_3d_axis_aligned=aabb,
      image_crop_pil=image_crop
    )


    def __repr__(self):
        num_points = len(self.point_cloud_3d.points) if self.point_cloud_3d and self.point_cloud_3d.has_points() else 0
        mask_sum_repr = self.segmentation_mask_2d.sum() if self.segmentation_mask_2d is not None else 'None'
        mask_shape_repr = self.segmentation_mask_2d.shape if self.segmentation_mask_2d is not None else 'N/A'
        return (f"<DetectedObject: {self.class_name} "
                f"(Desc: '{self.description[:30]}...'), "
                f"2D_bbox_orig: {self.bounding_box_2d.tolist() if self.bounding_box_2d is not None else 'N/A'}, "
                f"Mask_Sum_orig: {mask_sum_repr} (Shape: {mask_shape_repr}), "
                f"3D_pts: {num_points}, "
                f"3D_OBB_center: {self.bounding_box_3d_oriented.center.tolist() if self.bounding_box_3d_oriented and not self.bounding_box_3d_oriented.is_empty() else 'N/A'}>")

# --- Helper functions for mask and RLE ---
def get_bounding_box_from_mask(mask_2d_np: np.ndarray) -> np.ndarray | None:
    if not np.any(mask_2d_np): return None
    rows, cols = np.any(mask_2d_np, axis=1), np.any(mask_2d_np, axis=0)
    if not np.any(rows) or not np.any(cols): return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return np.array([xmin, ymin, xmax, ymax])

def rle_to_mask(rle_obj: dict) -> np.ndarray:
    return mask_util.decode(rle_obj).astype(bool)

# --- LLMClient class for query parsing ---
class LLMClient:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", logger=None, custom_generate_text=None):
        self.logger = logger or setup_logger_simple("LLMClientQueryParser")
        self.model_name = model_name
        self.tokenizer = None
        self.pipeline = None
        self.custom_generate_text = custom_generate_text
        
        # Only load model if no custom generate function is provided
        if custom_generate_text is None:
            try:
                self.logger.info(f"Loading tokenizer for: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    trust_remote_code=True
                )

                # Improved pad_token handling - matches reference style
                if self.tokenizer.pad_token is None:
                    if self.tokenizer.eos_token:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        self.logger.info(f"Set tokenizer pad_token to eos_token ({self.tokenizer.eos_token})")
                    elif self.tokenizer.unk_token:
                        self.tokenizer.pad_token = self.tokenizer.unk_token
                        self.logger.warning(
                            f"Set tokenizer pad_token to unk_token ({self.tokenizer.unk_token}). This might not be ideal.")
                    else:
                        self.logger.warning("Tokenizer has no pad_token, eos_token, or unk_token defined.")

                self.logger.info(f"Loading model: {self.model_name}")
                
                # Use pipeline initialization like reference (more efficient)
                self.pipeline = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=self.tokenizer,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,  # Changed from float16 to bfloat16 like reference
                    trust_remote_code=True
                )
                
                self.logger.info(f"Hugging Face LLM pipeline for {model_name} initialized.")

            except Exception as e:
                self.logger.error(f"Failed to initialize Hugging Face LLM pipeline: {e}", exc_info=True)
                self.pipeline = None
                self.tokenizer = None
                raise  # Keep the raise to maintain your original error handling
        else:
            self.logger.info("Using custom generate_text function - skipping model loading.")

    def generate_text(self, messages, max_new_tokens=1024, temperature=0.1, do_sample=False, return_full_text=False):
        """
        Generalized text generation function using the LLM pipeline or custom function.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            return_full_text: Whether to return the full text including prompt
            
        Returns:
            Generated text string or None if generation fails
        """
        # Use custom generate_text function if provided
        if self.custom_generate_text is not None:
            try:
                return self.custom_generate_text(
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    return_full_text=return_full_text,
                    tokenizer=self.tokenizer,  # Will be None when using custom function
                    pipeline=self.pipeline,    # Will be None when using custom function
                    logger=self.logger
                )
            except Exception as e:
                self.logger.error(f"Error in custom generate_text function: {e}", exc_info=True)
                return None
        
        # Use default implementation (requires pipeline and tokenizer)
        if not self.pipeline or not self.tokenizer:
            self.logger.error("LLM pipeline or tokenizer not initialized and no custom generate_text function provided.")
            return None

        try:
            # Format input prompt using tokenizer's chat template
            text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Set stop token (eos) if available
            stop_token_ids = []
            if self.tokenizer.eos_token_id is not None:
                stop_token_ids.append(self.tokenizer.eos_token_id)

            # Generation args — deterministic & clean
            generation_args = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "temperature": temperature,
                "do_sample": do_sample,
                "return_full_text": return_full_text,
            }
            if stop_token_ids:
                generation_args["eos_token_id"] = stop_token_ids[0]

            # Run LLM pipeline
            outputs = self.pipeline(text_prompt, **generation_args)
            raw_output = outputs[0]['generated_text'].strip()

            # Remove the prompt from the generated output if present and return_full_text=False
            if not return_full_text and raw_output.startswith(text_prompt):
                content = raw_output[len(text_prompt):].strip()
            else:
                content = raw_output

            return content

        except Exception as e:
            self.logger.error(f"Error during text generation: {e}", exc_info=True)
            return None

    def get_class_names_and_regions_from_query(self, user_query: str):
        if self.custom_generate_text is None and (not self.pipeline or not self.tokenizer):
            self.logger.error("Query Parser LLM pipeline or tokenizer not initialized and no custom generate_text function provided.")
            return None

        refined_query = self.remake_query(user_query)

        # System prompt for structured output
        system_prompt = """
    You are an information extraction system.  
    Given a natural language query that mentions one or more region tokens in the format <region_id>, identify the class_name associated with each region_id.

    **Instructions:**
    - Each region token (<region_id>) refers to an object in the image.
    - Extract the semantic class name (e.g., "pallet", "buffer", "transporter", "box", "forklift", etc.) associated with each region token based on the context of the query.
    - Return ONLY a JSON list with no additional text or explanations. JSON list format should be:
    [
        {"class_name": "<name>", "region_id": "<region_id>"},
    ...
    ]
    Example query: "What given the pallet in <region0>, where is the box <region1> and how many pallets are there in transporter <region2> inside buffer <region3>?"
    Example output:
    [
        {"class_name": "pallet", "region_id": "<region0>"},
        {"class_name": "box", "region_id": "<region1>"},
        {"class_name": "transporter", "region_id": "<region2>"},
        {"class_name": "buffer", "region_id": "<region3>"}
    ]
        """

        # Messages for chat-style formatting
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "User query:" + refined_query}
        ]

        try:
            # Use the new generate_text function
            content = self.generate_text(messages, max_new_tokens=1024, temperature=0.1, do_sample=False, return_full_text=False)
            
            if content is None:
                self.logger.error("Failed to generate text from LLM")
                return None

            self.logger.info(f'LLM raw output (stripped): {content}')
            print('LLM output:', content)
            
            self.logger.info(f'LLM raw output: {content}')
            print('LLM output:', content)
            
            # Extract JSON from the generated response only
            json_content = self._extract_json_from_response(content)
            
            # Parse JSON
            try:
                parsed_output = json.loads(json_content) if json_content else None
            except Exception as e:
                self.logger.warning(f"JSON parsing failed. Content:\n{json_content}\nError: {e}")
                parsed_output = None

            # Return both parsed output and refined query
            if parsed_output and self._validate_parsed_json(parsed_output):
                return parsed_output, refined_query
            else:
                self.logger.error(f"Failed to extract valid JSON from LLM response:\n{content}")
                return None

        except Exception as e:
            self.logger.error(f"Error during Query Parser LLM interaction: {e}", exc_info=True)
            return None
        
    def llm_for_query_refinement(self, prompt: str) -> str:
        """
        Call the LLM to refine the query based on the merge prompt.
        """
        try:
            if self.custom_generate_text is None and self.pipeline is None:
                self.logger.error("LLM pipeline not initialized and no custom generate_text function provided")
                return self._extract_original_query(prompt)
            
            original_query = self._extract_original_query(prompt)
            merge_instructions = self._extract_merge_instructions(prompt)
            
            # Format prompt for the model (for logging purposes when using custom function)
            messages = [
                {"role": "system", "content": "You are a helpful assistant that refines queries by merging regions with identical masks base on user instruction"},
                {"role": "user", "content": prompt}
            ]
            
            # Log formatted prompt if using default pipeline
            if self.custom_generate_text is None and self.tokenizer:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                print(f"Formatted prompt for LLM:\n{formatted_prompt}\n")
            
            # Use the new generate_text function
            refined_query = self.generate_text(messages, max_new_tokens=1024, temperature=0.1, do_sample=False, return_full_text=False)
            
            if refined_query is None:
                self.logger.error("Failed to generate refined query from LLM")
                return self._extract_original_query(prompt)
            
            print(f"Refined query from LLM:\n{refined_query}\n")
            
            # Clean up the response (remove any leftover system prompts or formatting)
            refined_query = self._clean_llm_response(refined_query)
            
            self.logger.info("Successfully refined query using LLM")
            return refined_query
            
        except Exception as e:
            self.logger.error(f"Error calling LLM for query refinement: {e}")
            # Fallback: return original query
            return self._extract_original_query(prompt)

    def _extract_original_query(self, prompt: str) -> str:
        """Helper method to extract original query from prompt."""
        try:
            if "Original query:\n" in prompt:
                parts = prompt.split("Original query:\n")[1]
                if "\nMerge instructions:" in parts:
                    return parts.split("\nMerge instructions:")[0].strip()
                elif "\n\nMerge instructions:" in parts:
                    return parts.split("\n\nMerge instructions:")[0].strip()
                elif "Merge instructions:" in parts:
                    return parts.split("Merge instructions:")[0].strip()
                else:
                    return parts.strip()
            return prompt
        except Exception:
            return prompt

    def _extract_merge_instructions(self, prompt: str) -> str:
        """Helper method to extract merge instructions from prompt."""
        try:
            if "Merge instructions:\n" in prompt:
                return prompt.split("Merge instructions:\n")[1].strip()
            elif "\nMerge instructions:" in prompt:
                return prompt.split("\nMerge instructions:")[1].strip()
            elif "Merge instructions:" in prompt:
                return prompt.split("Merge instructions:")[1].strip()
            return ""
        except Exception:
            return ""

    def _clean_llm_response(self, response: str) -> str:
        """Clean up LLM response to extract only the refined query."""
        # Remove common prefixes that models might add
        prefixes_to_remove = [
            "Here is the refined query:",
            "Refined query:",
            "The refined query is:",
            "Query:",
            "Answer:",
        ]
        
        cleaned = response.strip()
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove any trailing explanations (look for common patterns)
        lines = cleaned.split('\n')
        if len(lines) > 1:
            # If there are multiple lines, try to identify which is the actual query
            for i, line in enumerate(lines):
                if line.strip() and not line.lower().startswith(('note:', 'explanation:', 'this', 'the query')):
                    return line.strip()
        
        return cleaned
        
    def _extract_json_from_response(self, content: str):
        """
        Extract JSON array from LLM response, being more careful to avoid 
        matching examples from the prompt.
        """
        import json
        import re
        
        # First, try to find a complete JSON array
        json_start = content.find('[')
        json_end = content.rfind(']')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            potential_json = content[json_start:json_end + 1].strip()
            try:
                parsed = json.loads(potential_json)
                if self._validate_parsed_json(parsed):
                    return potential_json
            except json.JSONDecodeError:
                pass
        
        # If that fails, try to find JSON objects and construct an array
        # But be more specific about the pattern to avoid matching examples
        json_objects = re.findall(
            r'\{\s*"class_name"\s*:\s*"[^"]*"\s*,\s*"region_id"\s*:\s*"<region\d+>"\s*\}', 
            content
        )
        
        if json_objects:
            try:
                json_array_str = '[' + ','.join(json_objects) + ']'
                parsed = json.loads(json_array_str)
                if self._validate_parsed_json(parsed):
                    return json_array_str
            except json.JSONDecodeError:
                pass
        
        # Last resort: try to extract individual values and construct JSON
        # Look for actual region tokens in the user's query format
        class_names = re.findall(r'"class_name"\s*:\s*"([^"]*)"', content)
        region_ids = re.findall(r'"region_id"\s*:\s*"(<region\d+>)"', content)
        
        if len(class_names) == len(region_ids) and len(class_names) > 0:
            manual_json = [
                {"class_name": class_name, "region_id": region_id}
                for class_name, region_id in zip(class_names, region_ids)
            ]
            if self._validate_parsed_json(manual_json):
                self.logger.warning("Used manual JSON construction as fallback")
                return json.dumps(manual_json)
        
        return None

    def _validate_parsed_json(self, parsed_output):
        if not isinstance(parsed_output, list): 
            return False
        if not parsed_output: 
            return False # Check for empty list
        
        for item in parsed_output:
            if not isinstance(item, dict): 
                return False
            if "class_name" not in item or "region_id" not in item: 
                return False
            if not isinstance(item["class_name"], str) or not isinstance(item["region_id"], str): 
                return False
            if not item["class_name"].strip() or not item["region_id"].strip(): 
                return False
            # Validate that region_id follows expected format
            if not re.match(r'<region\d+>', item["region_id"]):
                return False
                
        return True

    def remake_query(self, query):
        # Generate region IDs based on number of <mask> tokens
        counter = [0]  # Using a list to allow mutation inside replacer

        def replacer(match):
            replacement = f"<region{counter[0]}>"
            counter[0] += 1
            return replacement

        query = re.sub(r'<mask>', replacer, query)
        return query
# --- Helper functions for Point Cloud Processing ---
def process_pcd_for_unik3d(cfg, pcd, run_dbscan=True):
    if not pcd.has_points() or len(pcd.points) == 0: return pcd
    try:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=cfg.get("pcd_sor_neighbors", 20), std_ratio=cfg.get("pcd_sor_std_ratio", 1.5))
    except RuntimeError: pass
    if not pcd.has_points() or len(pcd.points) == 0: return pcd
    voxel_size = cfg.get("pcd_voxel_size", 0.01)
    if voxel_size > 0: pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if not pcd.has_points() or len(pcd.points) == 0: return pcd
    if cfg.get("dbscan_remove_noise", True) and run_dbscan:
        pcd = pcd_denoise_dbscan_for_unik3d(pcd, eps=cfg.get("dbscan_eps", 0.05), min_points=cfg.get("dbscan_min_points", 10))
    return pcd

def pcd_denoise_dbscan_for_unik3d(pcd: o3d.geometry.PointCloud, eps=0.05, min_points=10) -> o3d.geometry.PointCloud:
    if not pcd.has_points() or len(pcd.points) < min_points: return pcd
    try: labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    except RuntimeError: return pcd
    counts = Counter(labels);
    if -1 in counts: del counts[-1]
    if not counts: return o3d.geometry.PointCloud()
    largest_cluster_label = counts.most_common(1)[0][0]
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
    if len(largest_cluster_indices) < min_points: return o3d.geometry.PointCloud()
    return pcd.select_by_index(largest_cluster_indices)

def get_bounding_box_for_unik3d(cfg, pcd):
    if not pcd.has_points() or len(pcd.points) < 3:
        return o3d.geometry.AxisAlignedBoundingBox(), o3d.geometry.OrientedBoundingBox()
    axis_aligned_bbox = pcd.get_axis_aligned_bounding_box()
    try: oriented_bbox = pcd.get_oriented_bounding_box(robust=cfg.get("obb_robust", True))
    except RuntimeError: oriented_bbox = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(axis_aligned_bbox)
    return axis_aligned_bbox, oriented_bbox

def axis_aligned_bbox_to_center_euler_extent_for_unik3d(min_coords, max_coords):
    center = tuple((min_val + max_val) / 2.0 for min_val, max_val in zip(min_coords, max_coords))
    extent = tuple(abs(max_val - min_val) for min_val, max_val in zip(min_coords, max_coords))
    return center, (0.0, 0.0, 0.0), extent

def oriented_bbox_to_center_euler_extent_for_unik3d(bbox_center, box_R, bbox_extent):
    return np.asarray(bbox_center), Rotation.from_matrix(box_R.copy()).as_euler("XYZ"), np.asarray(bbox_extent)

# --- UniK3D Model Instantiation ---
def instantiate_model(model_name):
    type_ = model_name[0].lower()
    name = f"unik3d-vit{type_}"
    model = UniK3D.from_pretrained(f"lpiccinelli/{name}")
    model.resolution_level = 9
    model.interpolation_mode = "bilinear"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model

warnings.filterwarnings("ignore")

class GeneralizedSceneGraphGenerator:
    def __init__(self, config_path="config/focused_config.py", device="cuda",
                 llm_query_parser_model_name_hf="Qwen/Qwen2.5-7B", custom_generate_text = None, test_path=None):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        self.cfg = Config.fromfile(config_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger_simple(name=self.cfg.get("logger_name", "FocusedSGG"))
        self.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        self.query_parser_llm = None
        _llm_name_to_use = llm_query_parser_model_name_hf or self.cfg.get("llm_model_name_hf")
        if _llm_name_to_use:
            try:
                self.query_parser_llm = LLMClient(model_name=_llm_name_to_use, custom_generate_text=custom_generate_text, logger=self.logger)
            except Exception as e:
                self.logger.error(f"Failed to initialize Query Parsing LLMClient with {_llm_name_to_use}: {e}", exc_info=True)
        else:
            self.logger.error("llm_query_parser_model_name_hf not provided. Critical failure.")
            raise ValueError("LLM for query parsing is not specified in arguments or config.")


        try:
            self.unik3d_model = instantiate_model(self.cfg.get("unik3d_model_size", "Large")) 
            self.logger.info(f"Successfully initialized UniK3D model ({self.cfg.get('unik3d_model_size', 'Large')})")
        except Exception as e:
            self.logger.error(f"Failed to initialize UniK3D model: {e}", exc_info=True)
            raise RuntimeError("Could not initialize UniK3D model") from e
        
        default_wis3d_folder = os.path.join(self.cfg.get("log_dir", "./temp_outputs/log_focused"), f"Wis3D_FocusedSGG_{self.timestamp}")
        self.cfg.wis3d_folder = self.cfg.get("wis3d_folder", default_wis3d_folder)
        os.makedirs(self.cfg.wis3d_folder, exist_ok=True)
        self.cfg.vis = self.cfg.get("vis", False)
    
    
    def _validate_dimensions(self, image_bgr_processed, points_3d_global, rle_masks_data, llm_parsed_regions):
        """
        Comprehensive dimension validation for all input arrays (using processed image dimensions for points_3d_global)
        """
        h_proc, w_proc = image_bgr_processed.shape[:2]
        
        if points_3d_global.shape[:2] != (h_proc, w_proc):
            self.logger.error(f"UniK3D output shape {points_3d_global.shape[:2]} doesn't match processed image shape ({h_proc}, {w_proc})")
            return False
        
        for i, rle_obj in enumerate(rle_masks_data):
            if "size" not in rle_obj:
                self.logger.error(f"RLE mask {i} missing 'size' field")
                return False
            rle_h, rle_w = rle_obj['size']
            if rle_h <= 0 or rle_w <= 0:
                self.logger.error(f"RLE mask {i} has invalid dimensions: ({rle_w}, {rle_h})")
                return False
        
        if len(llm_parsed_regions) != len(rle_masks_data):
            self.logger.error(f"LLM regions count ({len(llm_parsed_regions)}) != RLE masks count ({len(rle_masks_data)})")
            return False
        
        return True

    def _validate_mask_dimensions(self, mask, image_shape_processed, mask_index):
        """
        Validate individual mask dimensions (against processed image dimensions) and content
        """
        h_proc, w_proc = image_shape_processed[:2]
        
        if mask.shape != (h_proc, w_proc):
            self.logger.error(f"Mask {mask_index} shape {mask.shape} != processed image shape ({h_proc}, {w_proc})")
            return False
        
        if not np.any(mask):
            self.logger.warning(f"Mask {mask_index} is completely empty (in processed dimensions)")
            # Do not return False here, an empty mask might be valid, though unusual.
            # It will be handled by bbox checks or point count checks later.
        
        mask_area = np.sum(mask)
        total_area = h_proc * w_proc
        if total_area == 0: # Should not happen if image_shape_processed is valid
            self.logger.error(f"Mask {mask_index} validation: total_area is zero.")
            return False
        mask_ratio = mask_area / total_area
        
        if mask_ratio < 0.0001:
            self.logger.warning(f"Mask {mask_index} very small: {mask_ratio:.4%} of processed image")
        elif mask_ratio > 0.9:
            self.logger.warning(f"Mask {mask_index} very large: {mask_ratio:.4%} of processed image")
        
        return True


    def _load_image(self, image_input: str | np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """
        Loads an image, resizes it if specified, and returns original BGR, processed BGR, and original dimensions (w_orig, h_orig).
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            image_bgr_orig = cv2.imread(image_input)
            if image_bgr_orig is None:
                raise ValueError(f"Could not read image: {image_input}")
        elif isinstance(image_input, np.ndarray):
            image_bgr_orig = image_input.copy()
        else:
            raise TypeError("image_input must be a file path (str) or a NumPy array (BGR).")

        h_orig, w_orig = image_bgr_orig.shape[:2]
        if h_orig == 0 or w_orig == 0:
            raise ValueError("Image has zero height or width.")

        # CHANGE: Default to no resize unless explicitly set and different from original
        image_bgr_processed = image_bgr_orig.copy()  # Always start with copy
        target_h = self.cfg.get("image_resize_height", 0)  # Changed default to 0 (no resize)

        # Only resize if target_h is explicitly set and different from original
        if target_h > 0 and target_h != h_orig:
            scale = target_h / h_orig
            target_w = int(w_orig * scale)
            if target_w > 0: 
                image_bgr_processed = cv2.resize(image_bgr_orig, (target_w, target_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
                self.logger.info(f"Resized image from ({w_orig},{h_orig}) to ({target_w},{target_h})")
            else:
                self.logger.warning(f"Calculated target_w ({target_w}) for resize is not positive. Using original image dimensions for processing.")
        else:
            self.logger.info(f"No resize applied. Using original dimensions ({w_orig},{h_orig}) for processing.")
        
        return image_bgr_orig, image_bgr_processed, (w_orig, h_orig)

    def _process_rle_mask_with_coordinate_validation(self, rle_obj, target_h_proc, target_w_proc, mask_index):
        """
        Process RLE mask, resizing it from its native dimensions (from rle_obj['size'])
        to target processed dimensions (target_h_proc, target_w_proc).
        """
        try:
            if 'size' not in rle_obj:
                self.logger.error(f"RLE mask {mask_index} missing 'size' field")
                return None
            
            rle_h_native, rle_w_native = rle_obj['size']
            
            if rle_h_native <= 0 or rle_w_native <= 0:
                self.logger.error(f"RLE mask {mask_index} has invalid native dimensions in RLE data: ({rle_w_native}, {rle_h_native})")
                return None
            
            mask_native_dims = rle_to_mask(rle_obj)
            
            if mask_native_dims.shape != (rle_h_native, rle_w_native):
                self.logger.error(f"Decoded mask shape {mask_native_dims.shape} != RLE size ({rle_h_native}, {rle_w_native}) for mask {mask_index}")
                return None
            
            self.logger.debug(f"Mask {mask_index}: RLE native ({rle_w_native}x{rle_h_native}) -> Target processed ({target_w_proc}x{target_h_proc})")
            
            if (rle_h_native, rle_w_native) != (target_h_proc, target_w_proc):
                scale_h = target_h_proc / rle_h_native
                scale_w = target_w_proc / rle_w_native
                if abs(scale_h - 1.0) > 0.1 or abs(scale_w - 1.0) > 0.1:
                     self.logger.debug(f"Mask {mask_index} resize from native RLE to processed: scale_h={scale_h:.3f}, scale_w={scale_w:.3f}")
                
                mask_processed_dims = cv2.resize(
                    mask_native_dims.astype(np.uint8), 
                    (target_w_proc, target_h_proc), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                
                if mask_processed_dims.shape != (target_h_proc, target_w_proc):
                    self.logger.error(f"Resized mask shape {mask_processed_dims.shape} != target processed ({target_h_proc}, {target_w_proc}) for mask {mask_index}")
                    return None
                return mask_processed_dims
            else:
                return mask_native_dims.astype(bool)
        
        except Exception as e:
            self.logger.error(f"Error processing RLE mask {mask_index}: {e}", exc_info=True)
            return None

    def _validate_point_cloud_mask_alignment(self, points_3d_global, mask_2d_processed, obj_index):
        """
        Validate that point cloud (global, based on processed image) and mask (processed dimensions) are properly aligned.
        """
        if points_3d_global.shape[:2] != mask_2d_processed.shape:
            self.logger.error(f"Object {obj_index}: Point cloud global shape {points_3d_global.shape[:2]} != processed mask shape {mask_2d_processed.shape}")
            return False
        
        masked_points = points_3d_global[mask_2d_processed]
        
        if len(masked_points) == 0:
            # This warning is now deferred to min_points_threshold check to avoid prematurity
            # self.logger.warning(f"Object {obj_index}: No points extracted from processed mask")
            return True # Allow to proceed, might be filtered by point count later
        
        if np.all(masked_points == 0):
            self.logger.warning(f"Object {obj_index}: All extracted points are zero from processed mask")
        
        if not np.all(np.isfinite(masked_points)):
            self.logger.warning(f"Object {obj_index}: Extracted points from processed mask contain non-finite values")
        
        return True

    def process_and_refine_query(self, data_dict: dict) -> tuple[str, list[DetectedObject]]:
        """
        Complete method that processes JSON data to detect objects and then refines the query
        by merging regions with identical RLE masks.
        
        Args:
            data_dict: Dictionary containing image path, conversations, and RLE masks
            
        Returns:
            tuple: (refined_query, list_of_detected_objects)
        """
        self.logger.info("Starting complete process and query refinement...")
        
        # Step 1: Process JSON data to detect objects
        result = self.process_json_with_llm_classes(data_dict)
        
        # Handle the return value (could be tuple or just list depending on implementation)
        if isinstance(result, tuple) and len(result) == 2:
            detected_objects, original_query = result
            print(f"Detected {len(detected_objects)} objects from LLM parsing.")
            print(f"Original query: {original_query}")
        else:
            detected_objects = result
            # Fallback to extract query from conversations if not returned
            conversations = data_dict.get("conversations", [])
            if conversations and isinstance(conversations, list) and conversations[0].get("value"):
                human_query_full = conversations[0]["value"]
                original_query = human_query_full.split("<image>\n", 1)[-1] if "<image>\n" in human_query_full else human_query_full
            else:
                self.logger.warning("Could not extract original query, using empty string")
                original_query = ""
        
        self.logger.info(f"Initial processing complete. Detected {len(detected_objects)} objects.")
        
        # Step 2: Refine query by merging regions with identical masks
        if not detected_objects:
            self.logger.info("No detected objects to refine")
            return original_query, detected_objects
        
        refined_query, refined_detected_objects = self.refine_query(detected_objects, original_query)
        image_path = data_dict.get("image")
        vis_save_path = os.path.join(self.cfg.get("log_dir", "./temp_outputs/log_focused"), 
                            f"mask_visualization_{data_dict.get('id', 'scene')}_{self.timestamp}.jpg")
        image_bgr_orig, image_bgr_processed, (w_orig, h_orig) = self._load_image(image_path)
        
        self.visualize_all_masks_on_original(image_bgr_orig, refined_detected_objects, vis_save_path)
        self.logger.info(f"Process and refinement complete. Final objects: {len(refined_detected_objects)}")
        self.logger.info(f"Original query:\n{original_query}")
        self.logger.info(f"Refined query:\n{refined_query}")
        
        return  refined_detected_objects, refined_query

    def process_json_with_llm_classes(self, data_dict: dict) -> tuple[list[DetectedObject], str]:
        """
        Process JSON data with LLM classes to detect objects.
        Modified to return both detected objects and the remade query.
        """
        self.logger.info("Starting process_json_with_llm_classes...")
        if not self.query_parser_llm: 
            self.logger.error("Query Parser LLM not initialized."); return [], ""
        if not self.unik3d_model: 
            self.logger.error("UniK3D model not initialized."); return [], ""
        
        image_path = data_dict.get("image")
        if not image_path: 
            self.logger.error("'image' field (path) missing."); return [], ""
        
        try:
            image_bgr_orig, image_bgr_processed, (w_orig, h_orig) = self._load_image(image_path)
            h_proc, w_proc = image_bgr_processed.shape[:2]

            image_rgb_orig = cv2.cvtColor(image_bgr_orig, cv2.COLOR_BGR2RGB)
            image_pil_orig = PILImage.fromarray(image_rgb_orig) # For final cropping in original dimensions

            image_rgb_processed = cv2.cvtColor(image_bgr_processed, cv2.COLOR_BGR2RGB) # For UniK3D and point colors
        except Exception as e:
            self.logger.error(f"Failed to load and prepare image {image_path}: {e}", exc_info=True); return [], ""

        conversations = data_dict.get("conversations")
        if not conversations or not isinstance(conversations, list) or not conversations[0].get("value"):
            self.logger.error("Invalid 'conversations' structure."); return [], ""
        
        human_query_full = conversations[0]["value"]
        human_query_for_llm = human_query_full.split("<image>\n", 1)[-1] if "<image>\n" in human_query_full else human_query_full

        llm_parsed_regions, remaked_query = self.query_parser_llm.get_class_names_and_regions_from_query(human_query_for_llm)
        if not llm_parsed_regions: 
            self.logger.error("LLM failed to parse regions or returned no regions."); return [], remaked_query
        
        rle_masks_data = data_dict.get("rle")
        if not isinstance(rle_masks_data, list): 
            self.logger.error("Missing or invalid 'rle' field."); return [], remaked_query

        image_tensor = torch.from_numpy(image_rgb_processed).permute(2,0,1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            outputs = self.unik3d_model.infer(image_tensor, camera=None, normalize=True)
        points_3d_global = outputs["points"].squeeze().permute(1,2,0).cpu().numpy() # Shape (h_proc, w_proc, 3)
        
        if not self._validate_dimensions(image_bgr_processed, points_3d_global, rle_masks_data, llm_parsed_regions):
            self.logger.error("Dimension validation failed - aborting processing")
            return [], remaked_query
        
        detected_object_instances = []
        min_initial_pts = self.cfg.get("min_points_threshold", 10)
        min_processed_pts = self.cfg.get("min_points_threshold_after_denoise", 5)
        min_bbox_volume = self.cfg.get("bbox_min_volume_threshold", 1e-7)

        wis3d_instance = None
        if self.cfg.get("vis", False):
            filename_prefix = data_dict.get("id", "scene") + "_" + self.timestamp
            wis3d_instance = Wis3D(self.cfg.wis3d_folder, filename_prefix)
            if points_3d_global.shape[:2] == image_rgb_processed.shape[:2]: # Check against processed
                global_vertices = points_3d_global.reshape((-1, 3))
                global_colors_uint8 = image_rgb_processed.reshape(-1, 3) # Colors from processed image
                if len(global_vertices) > 0:
                    wis3d_instance.add_point_cloud(
                        vertices=global_vertices, colors=global_colors_uint8, 
                        name="unik3d_global_scene_pts_processed_dims"
                    )

        for i in range(len(llm_parsed_regions)):
            llm_region_info = llm_parsed_regions[i]
            rle_obj = rle_masks_data[i]
            rle_str = rle_obj.get("counts", None)
            class_name = llm_region_info["class_name"]
            description = llm_region_info["region_id"]
            
            # Mask in processed dimensions (h_proc, w_proc)
            mask_2d_processed_dims = self._process_rle_mask_with_coordinate_validation(
                rle_obj, h_proc, w_proc, i
            )
            if mask_2d_processed_dims is None:
                self.logger.error(f"Failed to process RLE mask for {description} (idx {i})")
                continue
            
            if not self._validate_mask_dimensions(mask_2d_processed_dims, image_bgr_processed.shape, i):
                self.logger.warning(f"Processed mask validation failed for {description} (idx {i})")
                continue # Skip if mask shape is wrong, but not necessarily if empty/small (handled later)
            
            if not self._validate_point_cloud_mask_alignment(points_3d_global, mask_2d_processed_dims, i):
                # This currently means shapes mismatch, which is critical.
                self.logger.error(f"Point cloud and processed mask alignment failed for {description} (idx {i})")
                continue

            # --- Convert 2D attributes to original image dimensions ---
            # 1. Resize processed mask to original dimensions
            mask_2d_orig_dims = cv2.resize(
                mask_2d_processed_dims.astype(np.uint8),
                (w_orig, h_orig), # Target original dimensions
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            # 2. Calculate BBox from processed mask, then scale it to original dimensions
            bbox_2d_np_proc = get_bounding_box_from_mask(mask_2d_processed_dims)
            bbox_2d_np_orig = None
            image_crop_pil_orig = None

            if bbox_2d_np_proc is not None:
                if w_proc == 0 or h_proc == 0: # Should be caught by earlier checks
                    self.logger.error(f"Processed image dimensions are zero for {description}, cannot scale bbox.");
                else:
                    scale_x = w_orig / w_proc
                    scale_y = h_orig / h_proc

                    x1_orig = int(round(bbox_2d_np_proc[0] * scale_x))
                    y1_orig = int(round(bbox_2d_np_proc[1] * scale_y))
                    x2_orig = int(round(bbox_2d_np_proc[2] * scale_x))
                    y2_orig = int(round(bbox_2d_np_proc[3] * scale_y))
                    bbox_2d_np_orig = np.array([x1_orig, y1_orig, x2_orig, y2_orig])
                
                    # 3. Crop from original PIL image using scaled bbox (original dimensions)
                    try:
                        # Ensure bbox coordinates are within image bounds
                        x1_c_orig = max(0, min(x1_orig, w_orig-1))
                        y1_c_orig = max(0, min(y1_orig, h_orig-1))
                        x2_c_orig = max(x1_c_orig+1, min(x2_orig, w_orig))  # Ensure x2 > x1
                        y2_c_orig = max(y1_c_orig+1, min(y2_orig, h_orig))  # Ensure y2 > y1

                        # Additional validation
                        crop_width = x2_c_orig - x1_c_orig
                        crop_height = y2_c_orig - y1_c_orig
                        
                        if crop_width > 0 and crop_height > 0:
                            self.logger.debug(f"Cropping {description} from ({x1_c_orig},{y1_c_orig}) to ({x2_c_orig},{y2_c_orig}), size: {crop_width}x{crop_height}")
                            image_crop_pil_orig = image_pil_orig.crop((x1_c_orig, y1_c_orig, x2_c_orig, y2_c_orig))
                        else:
                            self.logger.warning(f"Invalid crop dimensions for {description} (idx {i}): {crop_width}x{crop_height}")
                            image_crop_pil_orig = None
                    except Exception as e_crop:
                        self.logger.warning(f"Crop error on original image for {description} (idx {i}): {e_crop}")
                        image_crop_pil_orig = None
            else:
                self.logger.warning(f"No 2D bbox from processed mask for {description} (idx {i}). Original bbox and crop will be None.")
            
            # --- 3D Processing (uses processed_dims mask and points_3d_global) ---
            obj_points = points_3d_global[mask_2d_processed_dims]
            if len(obj_points) < min_initial_pts: 
                self.logger.debug(f"Too few initial points ({len(obj_points)} < {min_initial_pts}) for {description} (idx {i}) from processed mask."); 
                continue
            
            pcd_obj = o3d.geometry.PointCloud()
            pcd_obj.points = o3d.utility.Vector3dVector(obj_points)
            # Colors from processed image, corresponding to obj_points
            obj_colors_rgb_0_255 = image_rgb_processed[mask_2d_processed_dims] 
            if len(obj_colors_rgb_0_255) == len(obj_points): 
                pcd_obj.colors = o3d.utility.Vector3dVector(obj_colors_rgb_0_255 / 255.0)
            
            processed_pcd = process_pcd_for_unik3d(self.cfg, pcd_obj)

            if not processed_pcd.has_points() or len(processed_pcd.points) < min_processed_pts:
                self.logger.debug(f"Too few processed points ({len(processed_pcd.points) if processed_pcd.has_points() else 0} < {min_processed_pts}) for {description} (idx {i})."); 
                continue
            
            aabb_3d, obb_3d = get_bounding_box_for_unik3d(self.cfg, processed_pcd)
            if aabb_3d.is_empty() or aabb_3d.volume() < min_bbox_volume :
                self.logger.debug(f"Small/empty 3D bbox (Volume: {aabb_3d.volume() if not aabb_3d.is_empty() else 'empty'}) for {description} (idx {i})."); 
                continue
            
            det_obj = DetectedObject(
                class_name, description, 
                mask_2d_orig_dims,
                rle_str,
                bbox_2d_np_orig,            # BBox in original dimensions (or None)
                processed_pcd,              # 3D data relative to processed image
                obb_3d,                     # 3D data relative to processed image
                aabb_3d,                    # 3D data relative to processed image
                image_crop_pil_orig         # Crop from original image (or None)
            )
            detected_object_instances.append(det_obj)
            self.logger.info(f"Created DetectedObject for {description} ({class_name}) (idx {i}) with original-dim 2D data.")

            if wis3d_instance and processed_pcd.has_points():
                cmap = matplotlib.colormaps.get_cmap("turbo")
                instance_color_float_0_1 = np.array(cmap(i / max(1, len(llm_parsed_regions) -1) if len(llm_parsed_regions) > 1 else 0.5)[:3])
                
                pcd_to_vis = o3d.geometry.PointCloud(processed_pcd) # This is the processed 3D point cloud
                pcd_to_vis.paint_uniform_color(instance_color_float_0_1) 

                vertices_to_vis = np.asarray(pcd_to_vis.points)
                colors_to_vis_uint8 = (np.asarray(pcd_to_vis.colors) * 255).astype(np.uint8)

                if len(vertices_to_vis) > 0:
                    wis3d_instance.add_point_cloud(
                        vertices=vertices_to_vis, colors=colors_to_vis_uint8, 
                        name=f"{i:02d}_{class_name}_reg_{description}_3Dpts"
                    )
                
                aa_center,_,aa_extent = axis_aligned_bbox_to_center_euler_extent_for_unik3d(aabb_3d.get_min_bound(), aabb_3d.get_max_bound())
                wis3d_instance.add_boxes(positions=np.array([aa_center]), eulers=np.array([(0,0,0)]), extents=np.array([aa_extent]), name=f"{i:02d}_{class_name}_aa_bbox3D_reg_{description}")
                if not obb_3d.is_empty() and np.all(np.array(obb_3d.extent) > 1e-6):
                    or_center,or_eulers,or_extent = oriented_bbox_to_center_euler_extent_for_unik3d(obb_3d.center, obb_3d.R, obb_3d.extent)
                    wis3d_instance.add_boxes(positions=np.array([or_center]), eulers=np.array([or_eulers]), extents=np.array([or_extent]), name=f"{i:02d}_{class_name}_or_bbox3D_reg_{description}")
        
        self.logger.info(f"Initial object detection finished. Generated {len(detected_object_instances)} objects.")
        
        # Add mask visualization if objects were detected
        if detected_object_instances:
            vis_save_path = os.path.join(self.cfg.get("log_dir", "./temp_outputs/log_focused"), 
                                        f"mask_visualization_{data_dict.get('id', 'scene')}_{self.timestamp}.jpg")
            self.visualize_all_masks_on_original(image_bgr_orig, detected_object_instances, vis_save_path)

        return detected_object_instances, remaked_query

    def merge_regions_with_identical_masks(self, detected_objects: list[DetectedObject], query: str) -> tuple[str, list[DetectedObject]]:
        """
        Merge regions with identical RLE masks and use LLM to refine the query.
        Returns both refined query and updated detected objects list.
        """
        if not detected_objects:
            return query, detected_objects
        
        # First check if there are any identical masks using compare_mask function
        has_identical_masks = False
        for i in range(len(detected_objects)):
            for j in range(i + 1, len(detected_objects)):
                obj1 = detected_objects[i]
                obj2 = detected_objects[j]
                if obj1.rle_mask_2d is not None and obj2.rle_mask_2d is not None:
                    similarity = self.compare_mask(obj1.rle_mask_2d, obj2.rle_mask_2d)
                    if similarity == 1.0:  # Identical masks found
                        has_identical_masks = True
                        break
            if has_identical_masks:
                break
        
        # If no identical masks found, return original query and objects
        if not has_identical_masks:
            self.logger.info("No identical RLE masks found. Keeping original query and objects.")
            return query, detected_objects
        
        self.logger.info("Identical RLE masks detected. Starting query refinement process.")
        
        # Find groups of objects with identical RLE masks
        mask_groups = {}
        for i, obj in enumerate(detected_objects):
            if obj.rle_mask_2d is not None:
                mask_key = str(obj.rle_mask_2d)  # Convert to string for hashing
                if mask_key not in mask_groups:
                    mask_groups[mask_key] = []
                mask_groups[mask_key].append((i, obj))
        
        # Find groups with multiple objects (identical masks)
        merge_groups = {}
        objects_to_remove = set()
        
        for mask_key, objects in mask_groups.items():
            if len(objects) > 1:
                # Group objects by their indices for merging
                indices = [idx for idx, obj in objects]
                class_names = [obj.class_name for idx, obj in objects]
                
                # Keep the first object as the merged one, mark others for removal
                primary_idx = min(indices)
                secondary_indices = [idx for idx in indices if idx != primary_idx]
                
                merge_groups[primary_idx] = {
                    'class_names': class_names,
                    'indices_to_remove': secondary_indices,
                    'primary_object': objects[0][1]  # First object in the group
                }
                objects_to_remove.update(secondary_indices)
        
        if not merge_groups:
            return query, detected_objects  # No identical masks found
        
        # Create new detected objects list
        new_detected_objects = []
        region_mapping = {}
        new_region_counter = 0
        
        for i, obj in enumerate(detected_objects):
            if i not in objects_to_remove:
                if i in merge_groups:
                    # Create merged object with combined class name
                    merged_obj = copy.deepcopy(obj)
                    
                    # Combine class names
                    class_names = merge_groups[i]['class_names']
                    unique_classes = list(dict.fromkeys(class_names))  # Preserve order, remove duplicates
                    
                    if len(unique_classes) == 1:
                        merged_obj.class_name = unique_classes[0]
                    else:
                        merged_obj.class_name = " with a ".join(unique_classes)
                    
                    new_detected_objects.append(merged_obj)
                    region_mapping[i] = new_region_counter
                    new_region_counter += 1
                else:
                    # Keep original object
                    new_detected_objects.append(copy.deepcopy(obj))
                    region_mapping[i] = new_region_counter
                    new_region_counter += 1
        
        # Generate LLM prompt for query refinement
        prompt = self._generate_merge_prompt(query, merge_groups, region_mapping, detected_objects)
        print(f"Generated prompt for LLM:\n{prompt}")
        self.logger.info("Generated prompt for LLM to refine query with merged regions.")
        # Use LLM to refine the query
        refined_query = self.query_parser_llm.llm_for_query_refinement(prompt)
        
        return refined_query, new_detected_objects

    def _generate_merge_prompt(self, original_query: str, merge_groups: dict, region_mapping: dict, detected_objects: list) -> str:
        """
        Generate prompt for LLM to refine the query with merged regions.
        """
        prompt = f"""
    **Original query:**
    {original_query}
    
    **Merge instructions:**
    """
        
        for primary_idx, merge_info in merge_groups.items():
            class_names = merge_info['class_names']
            indices_to_remove = merge_info['indices_to_remove']
            
            # Create combined class name
            unique_classes = list(dict.fromkeys(class_names))  # Preserve order, remove duplicates
            if len(unique_classes) == 1:
                combined_name = unique_classes[0]
            else:
                combined_name = " with a ".join(unique_classes)
            
            new_region_num = region_mapping[primary_idx]
            
            prompt += f" Merge <region{primary_idx}>"
            for idx in indices_to_remove:
                prompt += f" and <region{idx}>"
            prompt += f" into a single <region{new_region_num}> with class name '{combined_name}'"
        
        prompt += f"""

    **Region renumbering:**
    """
        for old_idx, new_idx in region_mapping.items():
            if old_idx not in merge_groups:  # Don't repeat merge information
                class_name = detected_objects[old_idx].class_name
                prompt += f"\n- <region{old_idx}> becomes <region{new_idx}> (class: {class_name})"
        
        prompt += f"""

    **Please rewrite the query follow these instructions:**
    1. Merged regions using their combined class names.
    2. Renumbered regions according to the mapping.
    3. Remove references to merged regions that are no longer needed.
    4. Return only the refined query without any explanation.
    
    **Example:**
    Original query: Given buffer zones <region0> <region1> and transporter <region2>, which pallets <region3> <region4> <region5> is inside the buffer in the left.
    
    Merge Instruction: Merge <region2> and <region3> with class name 'transporter with a pallet'
    
    Region renumbering:
    - <region0> becomes <region0> (class: buffer)
    - <region1> becomes <region1> (class: buffer)
    - <region4> becomes <region3> (class: pallet)
    - <region5> becomes <region4> (class: pallet)
    
    Answer: Given buffer zones <region0> <region1> and transporter with a pallet <region2>, which pallets <region3> <region4>  is inside the buffer in the left.
    
    """
        
        return prompt

    def refine_query(self, detected_objects: list[DetectedObject], query: str) -> tuple[str, list[DetectedObject]]:
        """
        Refine the original query based on detected objects.
        Enhanced version that merges regions with identical RLE masks.
        Returns both refined query and updated detected objects list.
        """
        if not detected_objects:
            return query, detected_objects
        
        # Merge regions with identical masks
        refined_query, new_detected_objects = self.merge_regions_with_identical_masks(detected_objects, query)
        
        self.logger.info(f"Query refinement complete. Original objects: {len(detected_objects)}, New objects: {len(new_detected_objects)}")
        
        return refined_query, new_detected_objects

    def __del__(self):
        if hasattr(self, 'query_parser_llm') and self.query_parser_llm:
            if hasattr(self.query_parser_llm, 'pipeline') and self.query_parser_llm.pipeline is not None:
                del self.query_parser_llm.pipeline 
                self.query_parser_llm.pipeline = None
            del self.query_parser_llm
            self.query_parser_llm = None 
            self.logger.info("Query Parser LLM client instance and its pipeline marked for release.")

        if hasattr(self, 'unik3d_model') and self.unik3d_model is not None:
            try:
                if hasattr(self.unik3d_model, 'device') and self.unik3d_model.device.type == 'cuda': 
                    self.unik3d_model.cpu()
            except Exception as e:
                self.logger.warning(f"Could not move UniK3D model to CPU: {e}")
            del self.unik3d_model
            self.unik3d_model = None
            self.logger.info("UniK3D model resources released.")
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("Cleaned up resources.")
        
    def visualize_all_masks_on_original(self, image_bgr_orig: np.ndarray, detected_objects: list[DetectedObject], save_path: str = None):
        """
        Visualize all detected object masks overlaid on the original image
        """
        if not detected_objects:
            self.logger.warning("No detected objects to visualize")
            return None
        
        # Create visualization image
        vis_image = image_bgr_orig.copy()
        h_orig, w_orig = vis_image.shape[:2]
        
        # Create color map for different objects
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('tab20')  # Get distinct colors
        
        # Create overlay image for transparency
        overlay = np.zeros_like(vis_image, dtype=np.uint8)
        
        for i, obj in enumerate(detected_objects):
            if obj.segmentation_mask_2d is not None:
                # Get color for this object
                color_norm = cmap(i / max(len(detected_objects), 1))
                color_bgr = (int(color_norm[2]*255), int(color_norm[1]*255), int(color_norm[0]*255))  # RGB to BGR
                
                # Verify mask dimensions
                if obj.segmentation_mask_2d.shape != (h_orig, w_orig):
                    self.logger.warning(f"Mask shape {obj.segmentation_mask_2d.shape} != image shape ({h_orig}, {w_orig}) for {obj.description}")
                    continue
                
                # Apply colored mask
                mask_indices = obj.segmentation_mask_2d
                overlay[mask_indices] = color_bgr
                
                # Draw bounding box if available
                if obj.bounding_box_2d is not None:
                    x1, y1, x2, y2 = obj.bounding_box_2d.astype(int)
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color_bgr, 2)
                    
                    # Add label
                    label = f"{i}: {obj.class_name}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(vis_image, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color_bgr, -1)
                    cv2.putText(vis_image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        # Blend overlay with original image
        alpha = 0.3  # Transparency
        vis_image = cv2.addWeighted(vis_image, 1-alpha, overlay, alpha, 0)
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, vis_image)
            self.logger.info(f"Saved mask visualization to: {save_path}")
        
        return vis_image

    def compare_mask(self, rle_mask1: np.ndarray, rle_mask2: np.ndarray) -> float:
        if rle_mask1 == rle_mask2:
            self.logger.info("RLE masks are identical.")
            return 1
        else:
            return 0

if __name__ == "__main__":
    CONFIG_DIR = "configs"
    CONFIG_FILE_NAME = "v2_hf_llm.py" 
    
    DEFAULT_GENERATOR_CONFIG = "config/focused_config.py"
    if os.path.exists(DEFAULT_GENERATOR_CONFIG):
        CONFIG_FILE_PATH = DEFAULT_GENERATOR_CONFIG
        CONFIG_DIR = os.path.dirname(CONFIG_FILE_PATH) 
        CONFIG_FILE_NAME = os.path.basename(CONFIG_FILE_PATH)
        print(f"Using existing config file: {CONFIG_FILE_PATH}")
    else:
        CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, CONFIG_FILE_NAME) 
        print(f"Default generator config {DEFAULT_GENERATOR_CONFIG} not found.")
        print(f"Will attempt to use/create: {CONFIG_FILE_PATH}")


    if not os.path.exists(CONFIG_DIR): os.makedirs(CONFIG_DIR)
    if not os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, "w") as f:
            f.write(f"""
# Config for focused_datagen_module.py ({CONFIG_FILE_NAME})
llm_model_name_hf = "Qwen/Qwen2.5-7B-Instruct" 
unik3d_model_size = "Large" 

logger_name = "FocusedSGGTest"
log_dir = "./temp_outputs/logs_focused" 
wis3d_folder = "./temp_outputs/wis3d_focused" 
vis = True 
image_resize_height = 0 # Set to 0 or original height (e.g. 1080) to disable resizing for testing scaling

pcd_sor_neighbors = 20; pcd_sor_std_ratio = 1.5; pcd_voxel_size = 0.01
dbscan_remove_noise = True; dbscan_eps = 0.05; dbscan_min_points = 10
obb_robust = True
min_points_threshold = 10 # Min points from processed mask
min_points_threshold_after_denoise = 5
bbox_min_volume_threshold = 1e-7
""")
        print(f"Created dummy config: {CONFIG_FILE_PATH}")

    sample_data_dict_from_user = {
        "id": "e6f08787cba0eb0e6f5386a81016cb94",
        "image": "054690.png", 
        "conversations": [
          {"from": "human", "value": "<image>\nGiven the available transporters <mask> <mask> and pallets <mask> <mask> <mask> <mask> <mask> <mask> <mask>, which pallet is the best choice for automated pickup by an empty transporter?"}
        ],
        # RLE sizes are [1080, 1920] - (height, width)
        "rle": [
            {
                "size": [
                1080,
                1920
                ],
                "counts": "XnRo03eQ16I2O2M2O1O1N2O1N3N1O1N2O1N2O1O1N3N1N2O1M3N2O1O1N2N3N1N10O10O1000O100000O01000O010000O0O200O01jNBRQO>mn0DRQO;on0W100O01000O0100000O01000O01000O010000O01000O010O10O1000O10O01000O01000000O2O1N2O1O1O1N2O1N3N0000O10001N10000MeMiPO[2Wo030O1O1O1O1N2001O000001N2O1O1O0O2O1O1N2O1N101N2O1O0ERPOkNPP1T1QPOkNPP1S1RPOkNoo0U1RPOjNoo0U1;I7Bkcek0"
            },
            {
                "size": [
                1080,
                1920
                ],
                "counts": "ofa_12cQ14M2M4M3M2M4M2M4M2N3L3N3L3N3M2M4M2M4M2N3L3N3M2M4M2000O101O010O010O010O01O010OWOePOUO\\o0g0hPOYOWo0e0jPO\\OWo0`0jPOCUo0;jPOQOI`0^o0;iPO0Vo0NjPOWOIf0]o00jPOZOJh0\\o0LjPO\\OIl0\\o0EkPO_OIo0\\o0@kPOAIR1[o0ZOlPODIU1\\o0TOkPOV1To0kNhPOFM[1\\o0nNdPOH0]1[o0POePOS1[o0jNePOZ1\\o0`010O0N3M2M3010O010O01O010O010O010O0001O010O01O010O10O00RQO]Mhn0c2VQO^Mjn0b2VQO_Mjn0a2TQO`Mln0`2SQObMln0e21O010O010O010O01O01O0O2O0O2N101O01O01000O01O00010O010O0100O010O0010N1MmPOcMRo0c2N100O100O0010O010O0010O01O00001O001O0O20O10JnPOgMon0Y2RQOfMnn0Z2QQOhMon0`2010O010O01NXMSQOg2nn01010O01O010O010O010O01O01O010O02N010O010O00010O010O010O0010O00O2N1N3L3L5K5L4K4M4K4L4M4K4M4K4M4K5L4M3N2N3L4MPii8"
            },
            {
                "size": [
                1080,
                1920
                ],
                "counts": "RcSh0;]Q1O06K6boN]OZo0j0`POXO^o0m0\\POUOco0Q1WPOQOfo0V1SPOlNlo0b1M2O1N3N1N2O1N2O1N2O1O2M2O1N3N1N2O2M2O1N3N1N4M2N1N2O2M2O1N3N1N2O2M101N2O1O2M2O1N3N1O1O2N1O1O2N1O2NgKhROS4Wm0mKjROU4Sm0jKPSOU4ol0kKRSOV4ll0iKVSOW4il0iKXSOW4gl0hK[SOY4cl0gK_SOX4Yl0oKhSOR4Vl0WLbSOi3\\l0k0O1N2O0O100O10000O100O010O01O0N3O0010O0100O010M210O010O010O010O010O1O010O010O010O0101N10000000001OO100O100nJ`SOk4al0TKaSOk4_l0UKcSOf4`l0YKbSOd4`l0\\KaSOb4`l0]KbSOb4^l0^KcSO`4b0_Knj01aTO_4a0`Knj00cTO^4?cKnj0OdTO]4>dK`k0W4bTOjK_k0U4`TOlKak0R4`TOnKak0Q4^TOPLck0n3^TORLck0m3\\TOTLek0j3\\TOVLek0i3ZTOXLhk0e3YTO[Lhk0d3WTO]Ljk0a3WTO_Ljk0`3UTOaLlk0]3UTOcLlk0\\3STOeLnk0Y3RTOhLok0W3QTOiLQl0T3oSOmLRl0Q3oSOoLRl0P3mSOPMUl0P3jSOPMWl0o2hSOQMZl0n2fSOQM\\l0o2bSORM_l0m2aSORMbl0l2]SOUMdl0j2\\SOUMfl0i2ZSOXMhl0f2XSOYM\\m0S2dROnMPn0^1oQObNcn0l0^QOTOTo08mPOHWo03kPOMVo01jPOOXo00hPOOZo0OfPO2\\o0KePO4]o0KbPO6jSeQ1"
            },
            {
                "size": [
                1080,
                1920
                ],
                "counts": "c_ie04Y12mN5^o0N`QOMVOe1em0dNoROH\\Od1em0jNiROACe1dm0oNdRO\\OGf1dm0UO^ROVONe1dm0ZOYROPO3e0H5lm0;TROkN8g0KNhm0h0^RO[ONHdm0Q1YROYO6Bam0^2]ROaMbm0c2]RO[Mcm0f2\\ROZMdm0g2]ROWMbm0l2]ROSMcm0n2]ROQMbm0Q3_ROmLam0S3_ROmLam0S3aROkL^m0V3cROiL]m0X3cROgL]m0Z3eROcLZm0_3?1O2M2O1O5J2O1O2M2O000O10000O010O10O100O2O1O1N2O1N10O10O10O010O10O10O01000O0100O01000O01000O010O10O10O2O1N2O1O]RO[LTm0c3mRO]LTm0b3lRO^LUm0`3kROaLVm0^3jRObLWm0]3hROdLYm0Z3hROfLYm0Y3gROgLZm0W3fROjL[m0U3eROkL\\m0T3cROmL^m0Q3cROoL^m0P3bROPM_m0n2aROSM_m0m2aROSM`m0l2_ROUMdm0g2]ROYMcm0g2]ROYMcm0f2]ROZMem0e2[RO[Mfm0d2YRO]Mhm0a2YRO^Mim0a2WRO_Mjm0_2VROaMkm0_2UROaMkm0_2TRObMlm0]2URObMkm0_2UROaMkm0^2URObMkm0OgQOX2>iMkm0OhQOW2<kMlm0NiQOU2<lMkm0OkQOT2:mMkm0OlQOR29oMlm0OlQOQ28PNkm0OoQOP25RNlm0NPROn15SNlm0OPROm14TNkm0OSROl11UNmm0OXROe1L\\Nlm0NbRO]1AfNmm0MkROS1YOoNmm0NRSOk0QOWOmm0MZSOe0hN^Oko0`0VPO@ko0?TPOBmo0<TPOCno0<RPODoo0;PPOERP19ooNGRP18moNITP1L^oN4?OUP1K^oN5=0kZWU1"
            },
            {
                "size": [
                1080,
                1920
                ],
                "counts": "WTdP1320]Q16O100O1O100O1O100O1O100O1G9\\Od0dMROYSOc1gl0]NdROX2[m0hMeROW2_OZMlm0>eROX2]m0iMbROX2]m0hMdROW2]m0hMcROX2]m0iMbROY2\\m0gMeROZ2Zm0eMfRO^2Xm0bMgRO`2Xm0_MiROa2Wm0^MiROc2Wm0]MhROd2Xm0[MiROe2Wm0ZMiROg2Wm0YMhROh2Xm0WMiROi2Wm0WMhROj2Xm0UMhROl2Xm0SMiROm2Wm0SMhROn2Xm0QMiROo2Wm0PMiROQ3Wm0oLhROR3Xm0mLiROS3Wm0lLiROU3Wm0kLhROV3Xm0iLiROW3Wm0hLiROY3Wm0gLhROZ3Xm0eLiRO[3Wm0eLhRO\\3im0100O1O100O1O100O1O100O100O1O100O1O100O1O100O1O100O1O10000nN`LjSOb3Ul0_LkSOa3Ul0^LkSOc3Tl0]LlSOd3Sl0]LmSOc3Sl0\\LmSOe3Sl0ZLnSOf3Rl0ZLmSOg3Sl0XLmSOi3Sl0VLnSOj3Rl0VLmSOk3Sl0TLmSOm3Sl0RLnSOn3Rl0RLmSOo3Sl0PLmSOQ4Sl0oKmSOQ4Sl0nKmSOS4Sl0lKmSOU4Sl0kKmSOU4Sl0jKmSOW4Sl0hKmSOY4Sl0gKmSOY4Sl0fKmSO[4Sl0dKmSO]4Sl0cKmSO]4Sl0bKmSO_4Sl0`KmSOa4Sl0_KmSOa4dl0100O1O100O1O100O1O100O1O100O1O100OkMWSO]Ohl0d0XSO\\Ohl0j2O1O100O100O1O10000001O1O1O1O1O2N1O1O1O1O1O1O2N1O1O1O1O1O1O1O2N1O1O1O1O1O1O2N1O1O1O1O1O1O2N1O1O1O1O1O1O1O2N1OO16J000000000000000000000000000000000000000000000000000]MfQOh1Zn0WNhQOh1Xn0WNjQOh1Vn0XNkQOg1Un0ZNkQOe1Un0\\NkQOc1Un0^NkQOa1Un0`NhQOVOOX2Yn0hNgQOW1Yn0jNgQOU1Yn0mNfQOR1Zn0oNfQOP1Zn0[NgQO3Oa1Zn0[NhQO5N^1[n0]NgQO6OW1^n0bNcQO9OQ1an0eN`QO<Oo0`n0eN`QO>0m0_n0eNaQO`0Ok0_n0FaQO;]n0FcQO:\\n0GdQO\\O1NZn0g0fQOVO43Un0h0gQORO84Qn0k0gQOQO83Qn0n0fQOoN92WUPg0"
            },
            {
                "size": [
                1080,
                1920
                ],
                "counts": "`dlX14P2`0UNOPn0EiSO>WNLkm0LlSO:WNKmm0LkSO:WNJmm0NjSOn0Vl0SOhSOn0Wl0TOhSOl0Wl0VOgSOk0Yl0VOeSOk0Zl0WOeSOi0Zl0YOdSOh0\\l0YObSOh0[k0jNmSO?f0h0\\k0iNoSO`0d0g0\\k0jNnSOb0d0e0^k0hNnSOe0b0d0_k0gNnSOh0b0a0`k0gNjSOm0d0=bk0eNgSOS1e09dk0cNeSOX1f05ek0cNaSO]1h01gk0aN^SOc1i0Mik0_N^SOf1h0Kjk0_N^SOg1f0Klk0]N^SOj1e0Imk0]N]SOl1f0Fmk0]N^SOn1e0Dmk0]N^SOQ2e0Amk0^N]SOS2f0^Omk0^N^SOU2e0\\Omk0^N^SOX2e0YOmk0_N^SOY2e0WOmk0_N^SO\\2f0SOlk0`N^SO_2f0POlk0aN^SO`2f0nNlk0aN^SOc2f0kNlk0aN^SOf2f0hNlk0bN^SOg2f0fNok0\\1QTOcNnk0_1RTO`Nnk0a1RTO^Nmk0d1STO[Nlk0g1TTOXNlk0i1TTOVN]k0SOmSOi2f0SN]k0TOlSOk2g0PN]k0TOmSOm2f0nM]k0TOmSOP3f0kM]k0UOmSOQ3f0iM]k0UOmSOT3f0fM]k0UOmSOW3f0cM]k0VOmSOX3f0aM]k0VOmSO[3f0^M]k0VOmSO^3f0[M]k0WOmSO_3f0YM]k0WOmSOb3f0VM]k0XOlSOd3g0SM]k0XOmSOf3f0QM]k0XOmSOi3f0nL]k0YOlSOk3g0kL]k0YOmSOm3f0iL`k0Y3`TOfL_k0\\3aTOcL_k0^3aTOaL^k0a3bTO^L^k0c3bTO\\L]k0f3cTOYLnj0J\\TOo3f0VLnj0K\\TOP4f0TLnj0K\\TOS4f0QLnj0L[TOU4g0nKnj0L\\TOW4f0lKnj0L\\TOZ4f0iKnj0M[TO\\4g0fKnj0M\\TO^4f0dKnj0M\\TOa4f0aKnj0N[TOc4h0]Kmj0O\\TOe4g0[Kmj0O\\TOh4g0XKmj00\\TOi4g0VKmj00\\TOl4g0SKmj00\\TOo4g0PKmj01\\TOP5^l0oJbSOS5`l02O1N2UTOhJSk0Z5kTOgJUk0Z5jTOfJdj0<lTOQ5>dJfj0m5XUOTJgj0n5WUOSJij0n5VUORJij0P6UUOQJjj0Q6TUOPJkj0R6TUOnIij0V6VUOjIgj0Z6YUOeIdj0_6\\UO`Iaj0d6_UO[I]j0j6cUOUI[j0n6eUOQIZj0Q7fUOnHYj0T751N2N2O1N2O0O1O01jKgUOm0Xj0TOhUO30[MXj0Q6hUOmIYj0h601O01O0O2N1N20101ZO`UOQJbj0n5_UOPJcj0P6]UOoIdj0Q6[UOnIfj0a62N2O1N2N1O10O0001O010O000010OfIRUOk5nj0VJSUOh5nj0WJTUOh5kj0YJVUOe5kj0ZJVUOQ5JVKPk0HXUOR5HVKoj0HZUOR5GVKnj0G]UOg5cj0XJ_UOf5bj0ZJ_UOd5bj0[J_UOd5bj0[J`UOd5`j0\\JaUOb5`j0]JbUOa5_j0^JcUO`5^j0`JcUOf4YOlKTk0]OeUOf4XOjKUk0_OdUOh4WOeKXk0CbUOh4UOcK[k0DbUOW5_j0iJbUOU5_j0jJcUOT5^j0kJdUOg0oNc3]k0fKdUOd0SOd3_l0YLdSOe3^l0WLfSOh3[l0TLiSOV3@SMil0DiSOY3_OQMdm0o2[ROPMgm0P3YROoLhm0_32N2N2O0O2N2N2O1N2N2N101N20000O2N0O1WO`QOYNbn0d1`QO[Nbn0b1aQO]N`n0`1cQO^N_n0_1dQO_N^n0]1fQOaN\\n0\\1gQObN[n0[1hQObNZn0[1jQObNYn0[1jQObNYn0[1jQObNYn0Z1cQOgNdn0V1[QOnNen0o0[QOSOfn0j0YQOZOfn0c0[QO@en0=[QOEfn08YQOLgn00ZQO2gn0KXQO9hn0DXQO>in0^OXQOe0gn0YOXQOj0in0SOWQOo0obZ?"
            },
            {
                "size": [
                1080,
                1920
                ],
                "counts": "lkaV1c3Un00O1O100K5K5L4L400O1O100O1O100O1O100O1O100O1001O1O1O1\\MeROi0\\m0UOfROj0\\m0SOfROl0[m0SOfROl0Zm0UOfROj0Ym0XOgROg0Xm0[OhROd0Xm0]OeROYNOX2[m0FeRO9Zm0IfRO6Zm0LeRO3Zm0OfRO0Ym0\\OhROROOa1Ym0\\OiROTON^1Ym0_OiROTOOW1[m0EfROVOOQ1]m0IdROXOOo0\\m0IdROZO0m0Zm0JfRO[OOk0Ym0l0gROUOWm0l0iROTOUm0n0kROUN1NRm0o1nROnM43ml0P2oROjM84hl0T2PSOhM83gl0X2PSOeM92gl0Z2PSOdM91fl0]2VSO]M45el0`2WSO\\M33fl0b2XSOZM32dl0f2YSOXM31dl0h2YSOWM30cl0l2YSOTM4Obl0o2ZSORM4Nbl0Q3ZSORM3Lbl0T3[SOPM4J`l0X3]SOmL3J`l0Z3]SOlL3I_l0]3^SOjL3H^l0`3_SOhL3G]l0d3_SOeL4F]l0;[SOh24XM3D]l0<^SOh22XM4B[l0`0`SOd23YM2B[l0b0`SOa25[M0AZl0e0aSO^26\\MNAZl0g0`SO^28ZMLB\\l0g0_SO^29ZMIB^l0Q4iSO]LGC_l0R4jSO[LFC`l0S4kSO]LTl0e3lSOZLSl0h3mSOWLSl0j3mSOULRl0]3cSObL:0Rl0^3fSObL8OQl0_3iSObL6NQl0`3jSObL6Lok0b3mSObL4Knk0c3PTObL0LPl0a3RTOcLLMQl0c3RTOaLJMSl0d3TTO^LHNTl0e3TTO]LFOUl0f3UTOkLjk0W3VTOhLjk0Y3VTOfLik0\\3WTOcLik0^3WTOaLjk0o2kSOdL;<kk0n2lSOgL99lk0n2mSOiL77nk0o2lSOjL65Pl0o2lSOlL34Sl0n2lSOnL04Ul0m2lSOoLM4Zl0l2iSOPMK4^l0m2gSOnLI5bl0m2eSOnLH5dl0m2dSOcM^l0]2bSOaM`l0_2`SO_Mbl0a2^SO]Mdl0S2RSO^M:>el0S2RSO_M9<gl0S2RSOaM7:il0S2RSOcM67jl0T2RSOeM46kl0T2RSOgM14ol0S2RSOiMM4Sm0S2PSOiML3Vm0T2lROkML2Ym0S2gROoMNNcl0]OSTOf2XOSN0Jfl0^ORTOS4ok0nKQTOP4Pl0QLPTOn3Pl0SLPTOk3Ql0VLoSOQ2YO^Oil0bNnSOo1ZO]Oil0eNmSOl1\\O^Ogl0gNmSOi1]O_Ogl0iNlSOf1]OAhl0jNkSOd1\\OBjl0kNjSOb1[ODkl0kNjSOb1YOBnl0mNiSOR3Xl0oLhSOo2Yl0RMgSOm2Yl0TMgSOj2Zl0WMfSOg2[l0ZMeSOd2\\l0]MdSOk0XO6Um0POcSOh0ZO7Sm0ROcSOe0\\O7Rm0UObSOc0\\O7Sm0WOaSOa0[O8Um0XO`SO`0ZO8Vm0YO`SO?XO8Ym0ZO_SOn1bl0SN_SOj1bl0WN^SOh1bl0YN^SOe1cl0\\N]SOb1dl0_N\\SO_1el0bN[SO]1el0dN[SOZ1fl0gNZSOW1gl0jNWSOV1jl0kNUSOT1ll0mNRSOT1nl0mNPSOS1Qm0nNnROQ1Sm0POkROP1Vm0QOhROP1R1gMej0Z1WTOo0T1hMej0Z1VTOm0U1jMej0Z1TTOl0X1jMdj0[1RTOl0Y1jMej0[1QTOj0Z1lMej0[1PTOh0[1nMej0[1PTOe0[1QNej0[1PTOb0\\1SNdj0\\1PTO`0[1UNej0\\1PTO=[1XNej0\\1PTO:[1[Nej0\\1PTO7[1^Nej0\\1PTO5[1_Nej0]1PTO2[1bNej0]1PTOO[1eNej0]1PTOL[1hNej0]1PTOJZ1jNfj0]1PTOG[1lNej0^1PTOD[1oNej0\\1RTOCY1ROej0Z1TTOCV1TOfj0Y1UTOAV1VOej0`0lSOE;c0T1YOej0`0lSOB=c0R1\\Oej0T1[TO^OP1_Oej0S1\\TOjNBN\\15gj0R1]TOhNCOZ16gj0R1^TOeNE0W19gj0P1`TOdNEOV1=bb^b0"
            },
            {
                "size": [
                1080,
                1920
                ],
                "counts": "fRla03eQ14L3L3YOLfoN5XP1KhoN8VP1KgoN5YP1OcoN1\\P13`oNN`P1f0O1DhNUPO[1ho0jNTPOW1jo0mNRPOV1lo0<O1N3N4L3L2O2N1N10000O2O2M2O1O2M2O2N1N5kMTMcSO2S1m2Wk0PMgSO7n0k2Yk0nLiSO;i0i2]k0lLiSO`0f0e2`k0jLkSOe0`0b2ek0iLkSOj0;]2jk0iLjSOo08X2nk0hLkSOT12U2Sl0gLkSOX1NQ2Wl0gLjSO]1Kk1gl0Y1USOgKkl0Y4USOfKkl0Z4USOgKkl0Y4USOgKkl0b40GUSOgKkl0Y4USOfKll0c4OGUSOgKkl0b41FTSOhKkl0Y4USOfKll0c40FTSOhKkl0b41FTSOhKkl0Y4USOfKll0c40FTSOhKkl0b41FTSOhKll0X4TSOgKll0c40FTSOhKll0X4TSOgKll0Y4TSOhKll0X4TSOgKml0b4OGTSOhKll0X4TSOgKml0X4SSOiKll0X4TSOgKml0b40FSSOiKnl0V4RSOiKPm0_41EoROlKSm0\\42DkROQL>HXl0_4iSOaKXl0^4hSObKYl0\\4gSOeK[l0Y4eSOgK[l0Y4dSOhK]l0V4dSOjK]l0U4bSOlK`l0R4`SOnKal0P4`SOPLal0o3^SORLcl0c3oRO`L?Mcl0k3\\SOVLel0_3mROdL>Mfl0g3[SOYLgl0[3jROhL>Nil0Y3iROhL?Oil0a3VSO`Lkl0U3gROlL>Oll0T3fROlL?Oml0S3dROoL>Ool0Q3cROoL?OQm0Y3nROhLSm0m2_ROTM>OTm0l2^ROTM?OUm0U3jROlLWm0i2[ROWM?OXm0R3gROoLZm0f2XRO[M>O\\m0d2VRO\\M?O]m0m2bROTM_m0a2SRO_M?O`m0j2_ROWMbm0^2PROcM>Ndm0^2nQOcM?Ogm0c2XRO^Mhm0X2jQOhM?Ogm0OoQOX29jMhm0NSROj1GYN>Nhm00SROi1GXN?Ogm0OTROS24oMmn0S1ePOmN?Omn0]1RQOdNnn0MePOn0OVO>Non00gPOg0LZO?Onn01kPOj06VOon01nPO;ED?Oon05kPO`05\\Oho0b0XPO]Oio0b0XPO^Ojo0`0UPOAlo0=UPOBmo0=RPODPP1:PPOEQP1:PPOFRP18moNITP15moNJho0JTPO;3Lio0JVPO71NaViX1"
            },
            {
                "size": [
                1080,
                1920
                ],
                "counts": "XnRo03eQ16I2O2M2O1O1N2O1N3N1O1N2O1N2O1O1N3N1N2O1M3N2O1O1N2N3N1N10O10O1000O100000O01000O010000O0O200O01jNBRQO>mn0DRQO;on0W100O01000O0100000O01000O01000O010000O01000O010O10O1000O10O01000O01000000O2O1N2O1O1O1N2O1N3N0000O10001N10000MeMiPO[2Wo030O1O1O1O1N2001O000001N2O1O1O0O2O1O1N2O1N101N2O1O0ERPOkNPP1T1QPOkNPP1S1RPOkNoo0U1RPOjNoo0U1;I7Bkcek0"
            }
            ]
    }
    
    DEMO_IMAGE_DIR = "./demo_images_focused" 
    if not os.path.exists(DEMO_IMAGE_DIR):
        os.makedirs(DEMO_IMAGE_DIR)
    
    image_filename = sample_data_dict_from_user["image"]
    TARGET_IMAGE_PATH = '/root/Hachiman/SpatialRGPT/dataset_pipeline/054690.png'

    if not os.path.exists(TARGET_IMAGE_PATH):
        # RLE size is [height, width]
        img_h, img_w = sample_data_dict_from_user["rle"][0]["size"] 
        blank_image = np.full((img_h, img_w, 3), (200, 200, 200), dtype=np.uint8) 
        cv2.imwrite(TARGET_IMAGE_PATH, blank_image)
        print(f"Created dummy image for sample data: {TARGET_IMAGE_PATH} (Size: {img_w}x{img_h})")
    
    sample_data_dict_from_user["image"] = TARGET_IMAGE_PATH 

    main_logger = setup_logger_simple("MainTestFocusedSGG")
    generator = None
    try:
        generator = GeneralizedSceneGraphGenerator(
            config_path=CONFIG_FILE_PATH 
        )
        main_logger.info("GeneralizedSceneGraphGenerator initialized.")

        main_logger.info(f"\n--- Testing process_json_with_llm_classes with user-provided sample data ---")
        main_logger.info(f"Using image: {sample_data_dict_from_user['image']}")
        main_logger.info(f"Config setting image_resize_height: {generator.cfg.get('image_resize_height')}")
        
        if generator.query_parser_llm: 
            detected_objects_list, refined_query = generator.process_json_with_llm_classes(sample_data_dict_from_user)
            serialized_list = [obj.serialize() for obj in detected_objects_list]
            import pickle
            with open('stage1.pkl', 'wb') as f:
                pickle.dump(serialized_list, f)
            print(f"\nRefined query: {refined_query}")
            if detected_objects_list:
                main_logger.info(f"\nGenerated {len(detected_objects_list)} DetectedObject instances:")
                for i, obj in enumerate(detected_objects_list):
                    main_logger.info(f"  Object {i+1}: {obj}") # Repr now includes mask shape
                    if obj.image_crop_pil:
                        try:
                            crop_dir = "./temp_outputs/crops_focused"
                            if not os.path.exists(crop_dir): os.makedirs(crop_dir)
                            crop_filename = f"crop_ID{sample_data_dict_from_user['id']}_{obj.description}_{obj.class_name}_orig_dim.png"
                            obj.image_crop_pil.save(os.path.join(crop_dir, crop_filename))
                            main_logger.info(f"    Saved original-dim crop: {crop_filename} (Size: {obj.image_crop_pil.size})")
                        except Exception as e_crop:
                            main_logger.warning(f"Could not save crop for {obj.description}: {e_crop}")
                    if obj.segmentation_mask_2d is not None:
                        main_logger.info(f"    Mask original-dim shape: {obj.segmentation_mask_2d.shape}, Sum: {obj.segmentation_mask_2d.sum()}")
                    if obj.bounding_box_2d is not None:
                         main_logger.info(f"    BBox original-dim: {obj.bounding_box_2d.tolist()}")

            else:
                main_logger.warning("No DetectedObjects were generated from process_json_with_llm_classes.")
        else:
            main_logger.error("Query Parser LLM was not initialized. Cannot run the main test.")

    except SkipImageException as e_skip:
        main_logger.warning(f"Image processing was skipped: {e_skip}")
    except Exception as e_main:
        main_logger.error(f"An error occurred in the main execution: {e_main}", exc_info=True)
    finally:
        if generator:
            del generator 
        main_logger.info("\nTest run complete. Check logs and temp_outputs_focused directory.")





