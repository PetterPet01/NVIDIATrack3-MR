# predefined_modules.py
import re
import torch
import numpy as np
import io
import json
from PIL import Image # Make sure PIL is imported if not already globally
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    GenerationConfig,
)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
# from unidepth.models import UniDepthV2 # Remove UniDepthV2 import
from unik3d.models import UniK3D # <-- Add UniK3D import
import groundingdino.datasets.transforms as T

from VADAR.prompts.vqa_prompt import (
    VQA_PROMPT_CLEVR,
    VQA_PROMPT_GQA,
    VQA_PROMPT_GQA_HOLISTIC,
    VQA_PROMPT,
)
from .engine_utils import * # Assuming this has set_devices, Generator, BytesIO, base64, html_embed_image, dotted_image, box_image
from groundingdino.util.inference import load_model, predict


# Helper function to instantiate UniK3D model
def _instantiate_unik3d_model(model_size_str="Large", device="cuda"):
    """
    Instantiates the UniK3D model.
    model_size_str: "Large", "Base", "Small" (controls 'l', 'b', 's' in vit type)
    """
    type_ = model_size_str[0].lower() # L, B, S
    name = f"unik3d-vit{type_}"
    try:
        model = UniK3D.from_pretrained(f"lpiccinelli/{name}")
        model.resolution_level = 9 # As per run_with_unik3d.py
        model.interpolation_mode = "bilinear" # As per run_with_unik3d.py
        model = model.to(device).eval()
        print(f"UniK3D model '{name}' initialized on {device}")
        return model
    except Exception as e:
        print(f"Error instantiating UniK3D model {name}: {e}")
        raise

class PredefinedModule:
    def __init__(self, name, trace_path=None):
        self.trace_path = trace_path
        self.name = name

    def write_trace(self, html):
        if self.trace_path:
            with open(self.trace_path, "a+") as f:
                f.write(f"{html}\n")


class OracleModule(PredefinedModule):
    def __init__(self, name, trace_path=None):
        super().__init__(name, trace_path)
        self.reference_image = None
        self.scene_json = None
        self.oracle = None

    def set_oracle(self, oracle, reference_image, scene_json):
        self.oracle = oracle
        self.reference_image = reference_image
        self.scene_json = scene_json

    def clear_oracle(self):
        self.reference_image = None
        self.scene_json = None
        self.oracle = None


class LocateModule(OracleModule):
    def __init__(
        self,
        dataset,
        grounding_dino=None,
        molmo_processor=None,
        molmo_model=None,
        trace_path=None,
    ):
        super().__init__("loc", trace_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset

        if self.dataset in ["clevr", "gqa"]:
            self.molmo_processor = molmo_processor
            self.molmo_model = molmo_model
        else:
            self.grounding_dino = grounding_dino
            self.BOX_THRESHOLD = 0.25
            self.TEXT_TRESHOLD = 0.25

    def _extract_points(self, molmo_output, image_w, image_h):
        all_points = []
        for match in re.finditer(
            r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"',
            molmo_output,
        ):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                x = int(point[0] * image_w)
                y = int(point[1] * image_h)
                all_points.append([x, y])

        # convert all points to int
        return all_points

    def _parse_bounding_boxes(self, boxes, width, height):
        if len(boxes) == 0:
            return []

        bboxes = []
        for box in boxes:
            cx, cy, w, h = box
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            bboxes.append(
                [
                    int(x1 * width),
                    int(y1 * height),
                    int(x2 * width),
                    int(y2 * height),
                ]
            )
        return bboxes

    def transform_image(self, og_image):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        og_image = og_image.convert("RGB")
        img = np.asarray(og_image)
        im_t, _ = transform(og_image, None)
        return img, im_t

    def execute_pts(self, image, object_prompt):
        original_object_prompt = object_prompt
        if self.oracle:
            pts = self.oracle.locate(image, object_prompt, self.scene_json)
        else:
            if object_prompt[-1] != "s":
                object_prompt = object_prompt + "s"
            inputs = self.molmo_processor.process(
                images=[image],
                text="point to the " + object_prompt,
            )
            with torch.no_grad():
                inputs = {
                    k: v.to(self.molmo_model.device).unsqueeze(0)
                    for k, v in inputs.items()
                }
                output = self.molmo_model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                    tokenizer=self.molmo_processor.tokenizer,
                )
                generated_tokens = output[0, inputs["input_ids"].size(1) :]
                generated_text = self.molmo_processor.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                pts = self._extract_points(generated_text, image.size[0], image.size[1])

        if len(pts) == 0:
            self.write_trace(f"<p> No points found<p>")
            return []

        # trace
        if self.oracle:
            self.write_trace(f"<p>Locate [Oracle]: {original_object_prompt}<p>")
        else:
            self.write_trace(f"<p>Locate: {original_object_prompt}<p>")
        dotted_im = dotted_image(image, pts)
        dotted_html = html_embed_image(dotted_im)
        self.write_trace(dotted_html)
        if len(pts) > 1 and original_object_prompt[-1] != 's':
            original_object_prompt += 's'
        self.write_trace(f"<p>{len(pts)} {original_object_prompt} found<p>")
        self.write_trace(f"<p>Points: {pts}<p>")
        return pts

    def execute_bboxs(self, image, object_prompt):
        original_object_prompt = object_prompt
        width, height = image.size
        prompt = f"{object_prompt.replace(' ', '-')} ."
        _, img_gd = self.transform_image(image)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16): # Assuming cuda is available
            boxes, logits, phrases = predict(
                model=self.grounding_dino,
                image=img_gd,
                caption=prompt,
                box_threshold=self.BOX_THRESHOLD,
                text_threshold=self.TEXT_TRESHOLD,
                device=self.device, # Use self.device
            )
        bboxes = self._parse_bounding_boxes(boxes, width, height)

        if len(bboxes) == 0:
            self.write_trace(f"<p> No objects found<p>")
            return []

        # trace
        if self.oracle:
            self.write_trace(f"<p>Locate [Oracle]: {original_object_prompt}<p>")
        else:
            self.write_trace(f"<p>Locate: {original_object_prompt}<p>")
        boxed_image = box_image(image, bboxes)
        boxed_html = html_embed_image(boxed_image)
        self.write_trace(boxed_html)
        if len(bboxes) > 1 and original_object_prompt[-1] != 's':
            original_object_prompt += 's'
        self.write_trace(f"<p>{len(bboxes)} {original_object_prompt} found<p>")
        self.write_trace(f"<p>Boxes: {bboxes}<p>")

        return bboxes


class VQAModule(OracleModule):
    def __init__(
        self,
        dataset="omni3d",
        sam2_predictor=None,
        device=None,
        trace_path=None,
        api_key_path="./api.key",
    ):
        super().__init__("vqa", trace_path)
        self.generator = InternVLGenerator(model_path="OpenGVLab/InternVL3-9B", precision="bf16", use_8bit=False, multi_gpu=False)
        self.dataset = dataset

        if self.dataset in ["clevr", "gqa"]:
            self.sam2_predictor = sam2_predictor
            self.device = device

    def _get_prompt(self, question, holistic=False):
        if self.dataset == "clevr":
            return VQA_PROMPT_CLEVR.format(question=question)
        elif self.dataset == "gqa":
            if holistic:
                print("using gqa vqa prompt holistic")
                return VQA_PROMPT_GQA_HOLISTIC.format(question=question)
            else:
                return VQA_PROMPT_GQA.format(question=question)
        else:
            return VQA_PROMPT.format(question=question)

    def _get_bbox(self, mask, margin=20):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols): # Handle empty mask
            return [0,0,0,0]
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add margin
        rmin = max(0, rmin - margin)
        cmin = max(0, cmin - margin)
        rmax = min(mask.shape[0] - 1, rmax + margin)
        cmax = min(mask.shape[1] - 1, cmax + margin)

        return [cmin, rmin, cmax, rmax]

    def execute_pts(self, image, question, x, y):
        if self.oracle:
            answer = self.oracle.answer_question(x, y, question, self.scene_json)
            boxed_image = image # For trace consistency
        else:
            if int(x) == 0 and int(y) == 0: # Assuming (0,0) means holistic
                answer = self.predict(image, question, holistic=True)
                boxed_image = image
            else:
                with torch.no_grad():
                    sam_inpt_pts = np.array([[int(x), int(y)]])
                    sam_inpt_label = np.array([1])  # foreground label
                    self.sam2_predictor.set_image(np.array(image.convert("RGB"))) # Ensure RGB for SAM

                    masks, scores, logits = self.sam2_predictor.predict(
                        point_coords=sam_inpt_pts,
                        point_labels=sam_inpt_label,
                        multimask_output=True,
                    )

                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                
                # Check if any mask is found
                if len(masks) == 0 or scores[0] < 0.1: # Threshold for considering a mask valid
                    print("VQAModule: No reliable mask found from SAM. Using holistic VQA.")
                    answer = self.predict(image, question, holistic=True)
                    boxed_image = image # For trace
                else:
                    # Logic to combine masks or pick the best one
                    if len(scores) > 1 and scores[1] > 0.3: # Heuristic from original code
                        box1 = self._get_bbox(masks[0])
                        box2 = self._get_bbox(masks[1])
                        box = [
                            min(box1[0], box2[0]),
                            min(box1[1], box2[1]),
                            max(box1[2], box2[2]),
                            max(box1[3], box2[3]),
                        ]
                    else:
                        box = self._get_bbox(masks[0])
                    
                    # Crop image based on box for VQA prediction
                    # Ensure box coordinates are valid before cropping
                    if box[2] > box[0] and box[3] > box[1]:
                        cropped_image = image.crop(box)
                        boxed_image = box_image(image, [box]) # For trace
                        answer = self.predict(cropped_image, question)
                    else:
                        print("VQAModule: Invalid box from SAM mask. Using holistic VQA.")
                        answer = self.predict(image, question, holistic=True)
                        boxed_image = image # For trace


        # trace
        im_html = html_embed_image(image, 300)
        if self.oracle:
            self.write_trace(f"<p>Question [Oracle]: {question}</p>")
        else:
            self.write_trace(f"<p>Question: {question}</p>")
        
        # Dotted image based on input points
        dotted_im = dotted_image(image, [[x, y]])
        dotted_im_html = html_embed_image(dotted_im, 300)
        self.write_trace(dotted_im_html)

        # Show the (potentially) boxed/cropped image used for prediction
        if not self.oracle and boxed_image != image : # if it's not holistic
            boxed_trace_im_html = html_embed_image(boxed_image, 300)
            self.write_trace(f"<p>Region sent to VQA model:</p>")
            self.write_trace(boxed_trace_im_html)

        self.write_trace(f"<p>Answer: {answer}<p>")

        return answer.lower()

    def execute_bboxs(self, image, question, bbox):
        if bbox is None or (bbox[0]==0 and bbox[1]==0 and bbox[2]==0 and bbox[3]==0) : # Treat None or all-zero bbox as holistic
            answer = self.predict(image, question, holistic=True)
            boxed_image_for_trace = image
        else:
            # Ensure box coordinates are valid for cropping
            cmin, rmin, cmax, rmax = bbox
            if cmax > cmin and rmax > rmin:
                cropped_image = image.crop(bbox)
                answer = self.predict(cropped_image, question)
                boxed_image_for_trace = box_image(image, [bbox]) # For trace
            else:
                print("VQAModule: Invalid bbox provided. Using holistic VQA.")
                answer = self.predict(image, question, holistic=True)
                boxed_image_for_trace = image


        # trace
        im_html = html_embed_image(image, 300)
        self.write_trace(im_html)
        if self.oracle:
            self.write_trace(f"<p>Question [Oracle]: {question}</p>")
        else:
            self.write_trace(f"<p>Question: {question}</p>")

        # Show the (potentially) boxed/cropped image used for prediction
        if boxed_image_for_trace != image: # if it's not holistic
            boxed_trace_im_html = html_embed_image(boxed_image_for_trace, 300)
            self.write_trace(f"<p>Region sent to VQA model:</p>")
            self.write_trace(boxed_trace_im_html)
        
        self.write_trace(f"<p>Answer: {answer}<p>")
        return answer.lower()

    def remove_substring(self, output, substring):
        if substring in output:
            return output.replace(substring, "")
        else:
            return output

    def predict(self, img, question, holistic=False):
        prompt = self._get_prompt(question, holistic)
        
        # Ensure image is PIL for BytesIO saving
        if not isinstance(img, Image.Image):
            try:
                img = Image.fromarray(img) # If it's a numpy array
            except Exception as e:
                print(f"Error converting image to PIL for VQA: {e}. Using blank image.")
                img = Image.new('RGB', (100,100), color='grey')


        buffered = BytesIO()
        img.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ]
        output, _ = self.generator.generate("", messages)
        output = self.remove_substring(output, "```python")
        output = self.remove_substring(output, "```")
        
        try:
            answer = re.findall(r"<answer>(.*?)</answer>", output, re.DOTALL)[0].lower()
        except IndexError:
            print(f"VQAModule: Could not parse <answer> tag from LLM output: {output}")
            answer = "error: could not parse answer" # Or some other default
        return answer


class DepthModule(OracleModule):
    def __init__(
        self,
        # unidepth_model, # Removed
        unik3d_model,   # Added
        device,
        trace_path=None,
    ):
        super().__init__("depth", trace_path)
        # self.unidepth_model = unidepth_model # Removed
        self.unik3d_model = unik3d_model     # Added
        self.device = device

    def execute_pts(self, image, x, y):
        # image is a PIL Image
        if self.oracle:
            depth = self.oracle.depth(x, y, self.scene_json)
            preds_for_trace = None # No direct prediction map from oracle
        else:
            with torch.no_grad():
                # Prepare image for UniK3D (PIL to NumPy RGB, then to Tensor)
                image_np_rgb = np.array(image.convert("RGB")) # H, W, C
                
                # Ensure it's 3 channels if it was grayscale and converted
                if image_np_rgb.ndim == 2: # Should not happen after .convert("RGB") but as safeguard
                    image_np_rgb = np.stack((image_np_rgb,)*3, axis=-1)
                
                # Permute to C, H, W, add batch dim, convert to float, send to device
                image_tensor = torch.from_numpy(image_np_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                
                outputs = self.unik3d_model.infer(image_tensor, camera=None, normalize=True)
                # outputs["points"] is (1, 3, H, W) for (X,Y,Z)
                # Squeeze batch dim, permute to (H, W, 3)
                points_3d_per_pixel = outputs["points"].squeeze(0).permute(1, 2, 0).cpu().numpy()
                
                # Extract Z coordinate as depth (H, W)
                preds = points_3d_per_pixel[:, :, 2] 
                
                # Ensure x, y are within bounds of preds
                h_preds, w_preds = preds.shape
                # image.size is (width, height), so map x to w_preds, y to h_preds
                # Assuming x,y are in original image coordinates, and preds map directly
                safe_y = min(max(0, int(y)), h_preds - 1)
                safe_x = min(max(0, int(x)), w_preds - 1)
                depth = preds[safe_y, safe_x]
                preds_for_trace = preds # For visualization

        if self.oracle:
            self.write_trace(f"<p>Get Depth [Oracle]: ({x}, {y})<p>")
        else:
            self.write_trace(f"<p>Get Depth: ({x}, {y})<p>")
        
        dotted_im = dotted_image(image, [[x, y]]) # Use original image for point marking
        dotted_html = html_embed_image(dotted_im)
        self.write_trace(dotted_html)
        
        if preds_for_trace is not None:
            # Normalize depth map for visualization if needed by dotted_image/html_embed_image
            # This helps if dotted_image expects values in a certain range (e.g. 0-255 or 0-1)
            vis_preds = preds_for_trace.copy()
            if vis_preds.max() > vis_preds.min(): # Avoid division by zero for flat depth maps
                 vis_preds = (vis_preds - vis_preds.min()) / (vis_preds.max() - vis_preds.min() + 1e-6)
            
            dotted_depth_map_im = dotted_image(vis_preds, [[x,y]]) # Use normalized depth map for visualization
            dotted_depth_map_html = html_embed_image(dotted_depth_map_im)
            self.write_trace(dotted_depth_map_html)
            
        self.write_trace(f"<p>Depth: {depth}<p>")
        return depth

    def execute_bboxs(self, image, bbox):
        # image is a PIL Image
        x_mid = (bbox[0] + bbox[2]) / 2
        y_mid = (bbox[1] + bbox[3]) / 2

        if self.oracle:
            depth = self.oracle.depth(x_mid, y_mid, self.scene_json) # Assuming oracle can handle bbox center
            preds_for_trace = None
        else:
            with torch.no_grad():
                image_np_rgb = np.array(image.convert("RGB"))
                if image_np_rgb.ndim == 2:
                    image_np_rgb = np.stack((image_np_rgb,)*3, axis=-1)
                
                image_tensor = torch.from_numpy(image_np_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                
                outputs = self.unik3d_model.infer(image_tensor, camera=None, normalize=True)
                points_3d_per_pixel = outputs["points"].squeeze(0).permute(1, 2, 0).cpu().numpy()
                preds = points_3d_per_pixel[:, :, 2]
                
                h_preds, w_preds = preds.shape
                safe_y_mid = min(max(0, int(y_mid)), h_preds - 1)
                safe_x_mid = min(max(0, int(x_mid)), w_preds - 1)
                depth = preds[safe_y_mid, safe_x_mid]
                preds_for_trace = preds

        if self.oracle:
            self.write_trace(f"<p>Depth [Oracle] at bbox center: ({x_mid:.2f}, {y_mid:.2f})<p>")
        else:
            self.write_trace(f"<p>Depth at bbox center: ({x_mid:.2f}, {y_mid:.2f})<p>")

        dotted_im = dotted_image(image, [[x_mid, y_mid]])
        dotted_html = html_embed_image(dotted_im)
        self.write_trace(dotted_html)
        
        if preds_for_trace is not None:
            vis_preds = preds_for_trace.copy()
            if vis_preds.max() > vis_preds.min():
                 vis_preds = (vis_preds - vis_preds.min()) / (vis_preds.max() - vis_preds.min() + 1e-6)
            
            dotted_depth_map_im = dotted_image(vis_preds, [[x_mid, y_mid]])
            dotted_depth_map_html = html_embed_image(dotted_depth_map_im)
            self.write_trace(dotted_depth_map_html)
            
        self.write_trace(f"<p>Depth: {depth}<p>")
        return depth


class SameObjectModule(OracleModule):
    def __init__(
        self, dataset="omni3d", sam2_predictor=None, device=None, trace_path=None
    ):
        super().__init__("same_object", trace_path)
        self.dataset = dataset

        if self.dataset in ["clevr", "gqa"]:
            self.sam2_predictor = sam2_predictor
            self.device = device

    def _get_bbox(self, mask, margin=20): # Copied from VQAModule, can be utility
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols): return [0,0,0,0]
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmin = max(0, rmin - margin)
        cmin = max(0, cmin - margin)
        rmax = min(mask.shape[0] - 1, rmax + margin)
        cmax = min(mask.shape[1] - 1, cmax + margin)
        return [cmin, rmin, cmax, rmax]


    def get_iou(self, box1, box2):
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
        iou = area_inter / area_union if area_union != 0 else 0
        return iou

    def _get_mask(self, sam_inpt_pts, sam_inpt_label):
        with torch.no_grad():
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=sam_inpt_pts,
                point_labels=sam_inpt_label,
                multimask_output=True,
            )
            if len(masks) == 0: return None # Handle no mask found
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
        return masks[0]

    def execute_bboxs(self, image, bbox1, bbox2):
        answer = self.get_iou(bbox1, bbox2) > 0.92 # Threshold from original
        boxed_image = box_image(image, [bbox1, bbox2])
        im_html = html_embed_image(boxed_image, 300)
        self.write_trace(im_html)
        self.write_trace(f"<p>{answer}<p>")
        return answer

    def execute_pts(self, image, x_1, y_1, x_2, y_2):
        if self.oracle:
            answer = self.oracle.same_object(x_1, y_1, x_2, y_2, self.scene_json)
            # For trace consistency, we might need dummy bboxes if oracle doesn't provide them
            obj_1_bbox_trace = [int(x_1)-5, int(y_1)-5, int(x_1)+5, int(y_1)+5] # Example dummy
            obj_2_bbox_trace = [int(x_2)-5, int(y_2)-5, int(x_2)+5, int(y_2)+5] # Example dummy
        else:
            sam_inpt_label = np.array([1])  # foreground label
            obj_1_sam_inpt_pts = np.array([[int(x_1), int(y_1)]])
            obj_2_sam_inpt_pts = np.array([[int(x_2), int(y_2)]])
            
            self.sam2_predictor.set_image(np.array(image.convert("RGB"))) # Ensure RGB

            obj_1_mask = self._get_mask(obj_1_sam_inpt_pts, sam_inpt_label)
            obj_2_mask = self._get_mask(obj_2_sam_inpt_pts, sam_inpt_label)

            if obj_1_mask is None or obj_2_mask is None:
                print("SameObjectModule: Could not get masks for one or both points.")
                answer = False # Or handle error appropriately
                obj_1_bbox_trace = [int(x_1)-5, int(y_1)-5, int(x_1)+5, int(y_1)+5] 
                obj_2_bbox_trace = [int(x_2)-5, int(y_2)-5, int(x_2)+5, int(y_2)+5]
            else:
                obj_1_bbox = self._get_bbox(obj_1_mask)
                obj_2_bbox = self._get_bbox(obj_2_mask)
                answer = self.get_iou(obj_1_bbox, obj_2_bbox) > 0.92
                obj_1_bbox_trace = obj_1_bbox
                obj_2_bbox_trace = obj_2_bbox


        if self.oracle:
            self.write_trace(f"<p>Same Object [Oracle]: ({x_1}, {y_1}) and ({x_2}, {y_2})<p>")
        else:
            self.write_trace(f"<p>Same Object: ({x_1}, {y_1}) and ({x_2}, {y_2})<p>")
        
        # Ensure bboxes for trace are valid for box_image
        if not (obj_1_bbox_trace[2] > obj_1_bbox_trace[0] and obj_1_bbox_trace[3] > obj_1_bbox_trace[1]):
            obj_1_bbox_trace = [int(x_1)-1, int(y_1)-1, int(x_1)+1, int(y_1)+1] # minimal valid
        if not (obj_2_bbox_trace[2] > obj_2_bbox_trace[0] and obj_2_bbox_trace[3] > obj_2_bbox_trace[1]):
            obj_2_bbox_trace = [int(x_2)-1, int(y_2)-1, int(x_2)+1, int(y_2)+1] # minimal valid

        boxed_image = box_image(image, [obj_1_bbox_trace, obj_2_bbox_trace])
        im_html = html_embed_image(boxed_image, 300)
        self.write_trace(im_html)
        self.write_trace(f"<p>Answer: {answer}<p>")
        return answer


class Get2DObjectSize(OracleModule): # Changed to OracleModule for consistency if oracle might provide this
    def __init__(
        self, dataset="omni3d", sam2_predictor=None, device=None, trace_path=None
    ):
        super().__init__("get_2D_object_size", trace_path) # Name matches class
        self.dataset = dataset

        if self.dataset in ["clevr", "gqa"]:
            self.sam2_predictor = sam2_predictor
            self.device = device
            
    def _get_bbox(self, mask, margin=20): # Copied from VQAModule
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols): return [0,0,0,0]
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmin = max(0, rmin - margin)
        cmin = max(0, cmin - margin)
        rmax = min(mask.shape[0] - 1, rmax + margin)
        cmax = min(mask.shape[1] - 1, cmax + margin)
        return [cmin, rmin, cmax, rmax]

    def execute_bboxs(self, image, bbox):
        # image is PIL, bbox is [x1, y1, x2, y2]
        width = abs(bbox[0] - bbox[2])
        height = abs(bbox[1] - bbox[3])

        # trace
        self.write_trace(f"<p>Get 2D Object Size for Bbox: {bbox}<p>")
        boxed_image = box_image(image, [bbox])
        boxed_im_html = html_embed_image(boxed_image, 300)
        self.write_trace(boxed_im_html)
        self.write_trace(f"<p>Width: {width}, Height: {height}<p>")

        return width, height

    def execute_pts(self, image, x, y):
        # image is PIL, x,y are coords
        if self.oracle:
            # Assuming oracle can provide width, height from point.
            # This is a hypothetical oracle behavior.
            width, height = self.oracle.get_2d_object_size(x,y, self.scene_json)
            box_for_trace = [int(x)-width//2, int(y)-height//2, int(x)+width//2, int(y)+height//2]
        else:
            with torch.no_grad():
                sam_inpt_pts = np.array([[int(x), int(y)]])
                sam_inpt_label = np.array([1])  # foreground label
                self.sam2_predictor.set_image(np.array(image.convert("RGB"))) # Ensure RGB

                masks, scores, logits = self.sam2_predictor.predict(
                    point_coords=sam_inpt_pts,
                    point_labels=sam_inpt_label,
                    multimask_output=True,
                )
            
            if len(masks) == 0:
                 print("Get2DObjectSize: No mask found from SAM.")
                 width, height = 0,0
                 box_for_trace = [int(x)-1, int(y)-1, int(x)+1, int(y)+1] # Minimal box for trace
            else:
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                
                # Heuristic for combining masks from original code
                if len(scores) > 1 and scores[1] > 0.3:
                    box1 = self._get_bbox(masks[0])
                    box2 = self._get_bbox(masks[1])
                    box = [
                        min(box1[0], box2[0]), min(box1[1], box2[1]),
                        max(box1[2], box2[2]), max(box1[3], box2[3]),
                    ]
                elif len(scores) > 2 and scores[2] > 0.2: # Another heuristic from original
                    box1 = self._get_bbox(masks[0])
                    box2 = self._get_bbox(masks[1])
                    box3 = self._get_bbox(masks[2])
                    box = [
                        min(box1[0], box2[0], box3[0]), min(box1[1], box2[1], box3[1]),
                        max(box1[2], box2[2], box3[2]), max(box1[3], box2[3], box3[3]),
                    ]
                else:
                    box = self._get_bbox(masks[0])
                
                box_for_trace = box
                width = abs(box[0] - box[2])
                height = abs(box[1] - box[3])
        
        # Ensure box_for_trace is valid for box_image
        if not (box_for_trace[2] > box_for_trace[0] and box_for_trace[3] > box_for_trace[1]):
            box_for_trace = [int(x)-1, int(y)-1, int(x)+1, int(y)+1]


        # trace
        if self.oracle:
            self.write_trace(f"<p>Get 2D Object Size [Oracle]: ({x}, {y})<p>")
        else:
            self.write_trace(f"<p>Get 2D Object Size: ({x}, {y})<p>")
        
        boxed_image = box_image(image, [box_for_trace])
        boxed_im_html = html_embed_image(boxed_image, 300)
        self.write_trace(boxed_im_html)
        self.write_trace(f"<p>Width: {width}, Height: {height}<p>")

        return width, height


class ResultModule(PredefinedModule):
    def __init__(self, trace_path=None):
        super().__init__("result", trace_path)

    def execute_pts(self, var):
        self.write_trace(f"<p>Result: {var}<p>")
        return str(var)

    def execute_bboxs(self, var):
        self.write_trace(f"<p>Result: {var}<p>")
        return str(var)


class ModulesList:
    def __init__(self, models_path=None, trace_path=None, dataset="omni3d", api_key_path="./api.key", unik3d_model_size="Large"):
        set_devices() # Assuming this sets up CUDA if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset
        self.unik3d_model_size = unik3d_model_size # Store for DepthModule

        if dataset in ["clevr", "gqa"]:
            self.sam2_checkpoint = f"{models_path}/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
            self.sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml" # Path should be correct relative to execution
            sam_model = build_sam2(self.sam2_model_cfg, self.sam2_checkpoint, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(sam_model)
            print("SAM2 Initialized")
            self.molmo_processor = AutoProcessor.from_pretrained(
                "allenai/Molmo-7B-D-0924",
                trust_remote_code=True,
                torch_dtype="auto", # Changed to auto for flexibility
                device_map="auto",
            )
            self.molmo_model = AutoModelForCausalLM.from_pretrained(
                "allenai/Molmo-7B-D-0924",
                trust_remote_code=True,
                torch_dtype="auto", # Changed to auto
                device_map="auto",
            )
            print("Molmo Initialized")
        else: # omni3d or other bbox based
            self.grounding_dino = load_model(
                f"{models_path}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                f"{models_path}/GroundingDINO/weights/groundingdino_swint_ogc.pth"
            )
            print("GroundingDINO Initialized")
            # For omni3d, SAM might not be explicitly initialized here but could be if needed by some modules
            # For now, following original structure. If VQA/SameObject for omni3d needs SAM, it's missing.
            # Assuming VQA/SameObject for omni3d don't use SAM based on current structure.
            self.sam2_predictor = None # For omni3d, SAM is not primarily used by these modules based on original logic

        # Initialize UniK3D instead of UniDepth
        # self.unidepth_model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14").to(self.device) # Removed
        self.unik3d_model = _instantiate_unik3d_model(model_size_str=self.unik3d_model_size, device=self.device) # Added

        self.modules = self.get_module_list(self.dataset, trace_path, api_key_path)
        self.module_names = [module.name for module in self.modules]
        self.module_executes = self.get_module_executes(self.dataset)

    def get_module_executes(self, dataset):
        if dataset in ["clevr", "gqa"]:
            return {
                self.module_names[i]: self.modules[i].execute_pts
                for i in range(len(self.modules))
            }
        else: # omni3d or other bbox based
            return {
                self.module_names[i]: self.modules[i].execute_bboxs
                for i in range(len(self.modules))
            }

    def get_module_list(self, dataset, trace_path, api_key_path):
        if dataset in ["clevr", "gqa"]:
            return [
                LocateModule(
                    dataset=dataset,
                    molmo_processor=self.molmo_processor,
                    molmo_model=self.molmo_model,
                    trace_path=trace_path,
                ),
                VQAModule(
                    dataset=dataset,
                    sam2_predictor=self.sam2_predictor,
                    device=self.device,
                    trace_path=trace_path,
                    api_key_path=api_key_path,
                ),
                DepthModule(self.unik3d_model, self.device, trace_path), # Pass UniK3D
                SameObjectModule(
                    dataset=dataset,
                    sam2_predictor=self.sam2_predictor,
                    device=self.device,
                    trace_path=trace_path,
                ),
                Get2DObjectSize(
                    dataset=dataset,
                    sam2_predictor=self.sam2_predictor,
                    device=self.device,
                    trace_path=trace_path,
                ),
                ResultModule(trace_path),
            ]
        else: # omni3d or other bbox based
            return [
                LocateModule(
                    dataset=dataset,
                    grounding_dino=self.grounding_dino,
                    trace_path=trace_path,
                ),
                VQAModule( # Note: VQAModule for omni3d doesn't get SAM based on current structure
                    dataset=dataset, 
                    sam2_predictor=None, # Explicitly None for omni3d
                    device=self.device,
                    trace_path=trace_path, 
                    api_key_path=api_key_path
                ),
                DepthModule(self.unik3d_model, self.device, trace_path), # Pass UniK3D
                SameObjectModule( # Note: SameObjectModule for omni3d doesn't get SAM
                    dataset=dataset, 
                    sam2_predictor=None, # Explicitly None for omni3d
                    device=self.device,
                    trace_path=trace_path
                ),
                Get2DObjectSize( # Note: Get2DObjectSize for omni3d doesn't get SAM
                    dataset=dataset, 
                    sam2_predictor=None, # Explicitly None for omni3d
                    device=self.device,
                    trace_path=trace_path
                ),
                ResultModule(trace_path),
            ]

    def set_trace_path(self, trace_path):
        for module in self.modules:
            module.trace_path = trace_path

    def set_oracle(self, oracle, reference_image, scene_json):
        for module in self.modules:
            if hasattr(module, "set_oracle"):
                module.set_oracle(oracle, reference_image, scene_json)

    def clear_oracle(self):
        for module in self.modules:
            if hasattr(module, "set_oracle"):
                module.clear_oracle()

