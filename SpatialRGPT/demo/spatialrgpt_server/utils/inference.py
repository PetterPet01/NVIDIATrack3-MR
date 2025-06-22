import re
import torch
import numpy as np
from PIL import Image

# Assuming these are available in the PYTHONPATH as per project structure
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, process_images, process_regions, tokenizer_image_token

def run_vlm_inference(
    request, all_models, raw_image, seg_masks, colorized_depth
):
    """
    Main inference function, adapted from Gradio demo.
    """
    tokenizer = all_models["tokenizer"]
    model = all_models["spatialrgpt_model"]
    image_processor = all_models["image_processor"]
    
    input_str = request.messages[-1].content
    if not isinstance(input_str, str): # Handle multipart content
        text_parts = [item.text for item in input_str if item.type == 'text']
        input_str = "\n".join(text_parts)

    use_depth = "<depth>" in input_str
    
    query_base = re.sub(r"<region\d+>", "<mask> <depth>" if use_depth else "<mask>", input_str)

    # For now, we assume each call is a new conversation.
    # A more advanced server could manage conversation state with session IDs.
    conv = conv_templates[request.conv_mode].copy()
    query = DEFAULT_IMAGE_TOKEN + "\n" + query_base
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # --- Prepare Tensors ---
    device = model.device
    dtype = torch.bfloat16 if request.use_bfloat16 and torch.cuda.is_bf16_supported() else torch.float16
    if str(device) == 'cpu':
        dtype = torch.float32

    original_model_dtype = next(model.parameters()).dtype
    if original_model_dtype != dtype and str(device) != 'cpu':
        model.to(dtype=dtype)

    pil_raw_image = Image.fromarray(raw_image)
    images_tensor = process_images([pil_raw_image], image_processor, model.config).to(device, dtype=dtype)
    
    pil_colorized_depth = Image.fromarray(colorized_depth)
    depths_tensor = process_images([pil_colorized_depth], image_processor, model.config).to(device, dtype=dtype)

    # --- Mask Processing ---
    current_region_tags = re.findall(r"<region(\d+)>", input_str)
    current_region_indices = [int(tag) for tag in current_region_tags]

    final_masks_for_model = None
    if seg_masks and current_region_indices:
        # BUG WORKAROUND from Gradio demo for cv2.resize issue in process_regions
        np_masks_for_processing = [np.array(Image.fromarray(m * 255)) for m in seg_masks]
        
        all_masks_tensor = process_regions(
            np_masks_for_processing, image_processor, model.config
        ).to(device, dtype=dtype)

        # Select only the masks mentioned in the current prompt
        indices_to_pass = [idx for idx in current_region_indices if 0 <= idx < all_masks_tensor.size(0)]
        if indices_to_pass:
            final_masks_for_model = all_masks_tensor[indices_to_pass]

    # --- Run Generation ---
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[images_tensor],
            depths=[depths_tensor],
            masks=[final_masks_for_model] if final_masks_for_model is not None else None,
            do_sample=True if request.temperature > 0 else False,
            temperature=request.temperature,
            max_new_tokens=request.max_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    # Restore model dtype
    if original_model_dtype != dtype and str(device) != 'cpu':
        model.to(dtype=original_model_dtype)
    
    # --- Process and Remap Output ---
    outputs_raw = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs_stripped = outputs_raw.strip().removesuffix(stop_str).strip()
    
    # Remap model's output indices [0], [1]... to the original prompt's <regionX>
    mapping_dict = {str(out_idx): str(in_idx) for out_idx, in_idx in enumerate(current_region_indices)}
    if mapping_dict:
        remapped_output = re.sub(
            r"\[([0-9]+)\]", 
            lambda x: f"[{mapping_dict[x.group(1)]}]" if x.group(1) in mapping_dict else x.group(0), 
            outputs_stripped
        )
    else:
        remapped_output = outputs_stripped

    return remapped_output