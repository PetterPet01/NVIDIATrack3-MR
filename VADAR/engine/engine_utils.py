import numpy as np
# from openai import OpenAI # No longer needed for the Generator
import json
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import time
import torch

# New imports for local Qwen model
from transformers import AutoModelForCausalLM, AutoTokenizer


def correct_indentation(code_str):
    lines = code_str.split("\n")
    tabbed_lines = ["\t" + line for line in lines]
    tabbed_text = "\n".join(tabbed_lines)
    return tabbed_text


def replace_tabs_with_spaces(code_str):
    return code_str.replace("\t", "    ")


def untab(text):
    lines = text.split("\n")
    untabbed_lines = []
    for line in lines:
        if line.startswith("\t"):
            untabbed_lines.append(line[1:])
        elif line.startswith("    "):
            untabbed_lines.append(line[4:])
        else:
            untabbed_lines.append(line)
    untabbed_text = "\n".join(untabbed_lines)
    return untabbed_text


def get_methods_from_json(api_json):
    methods = []
    namespace = {}
    for method_info in api_json:

        signature = method_info["signature"]
        if "def" in method_info["implementation"]:
            full_method = method_info["implementation"]
        else:
            full_method = signature + "\n" + method_info["implementation"]
        methods.append(full_method)

    return methods, namespace


class Generator:
    def __init__(self, model_name="Qwen/Qwen1.5-0.5B-Chat", temperature=0.7, device_preference=None, max_new_tokens=1024):
        """
        Initializes the Generator with a local Qwen model.

        Args:
            model_name (str): Hugging Face model identifier for Qwen (e.g., "Qwen/Qwen1.5-0.5B-Chat")
                              or path to a local model directory.
            temperature (float): Sampling temperature for generation.
            device_preference (str, optional): Preferred device ("cuda", "mps", "cpu").
                                               If None, auto-detects.
            max_new_tokens (int): Maximum number of new tokens to generate.
        """
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
        print(f"Generator: Selected device: {self.device}")

        # Load tokenizer and model from Hugging Face or local path
        # Qwen models (especially Qwen1.5 and later) often require trust_remote_code=True
        print(f"Generator: Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Ensure pad_token is set. Some models might not have it set by default.
        # Using EOS token as PAD token if PAD is not defined.
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                print(f"Generator: Tokenizer pad_token was None, set to eos_token ('{self.tokenizer.eos_token}')")
            else:
                # Fallback if eos_token is also None (very unlikely for LLMs)
                print("Generator: Tokenizer pad_token and eos_token are None. Adding a new pad token '[PAD]'.")
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        print(f"Generator: Tokenizer pad_token: '{self.tokenizer.pad_token}', ID: {self.tokenizer.pad_token_id}")


        print(f"Generator: Loading model {self.model_name} onto device {self.device}...")
        # torch_dtype="auto" lets transformers choose the best precision (e.g., bfloat16 on Ampere GPUs)
        # device_map=str(self.device) attempts to load the model directly onto the specified device.
        # For multi-GPU, device_map="auto" is more common.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map=str(self.device), # Pass the determined device string
            trust_remote_code=True
        )
        
        # Verify model's primary device after loading with device_map
        # For models spread across multiple GPUs, self.model.device might show the device of the first part.
        print(f"Generator: Model loaded. Main model device: {self.model.device}")

        self.model.eval()  # Set the model to evaluation mode

    def remove_substring(self, output, substring):
        if substring in output:
            return output.replace(substring, "")
        else:
            return output

    def generate(self, prompt=None, messages=None):
        """
        Generates text using the local Qwen model.

        Args:
            prompt (str, optional): The user's prompt. If messages are also provided,
                                    this prompt is treated as the latest user message.
            messages (list, optional): A list of message dictionaries, e.g.,
                                       [{"role": "system", "content": "You are helpful."},
                                        {"role": "user", "content": "Hello!"}].
                                       If None, a new conversation starts with the prompt.

        Returns:
            tuple: (generated_text, new_conversation_history)
                   - generated_text (str): The model's response.
                   - new_conversation_history (list): The updated list of messages,
                                                      including the assistant's response.
        """
        current_conversation = []
        if messages:
            current_conversation = list(messages)  # Work with a copy

        if prompt:
            # If a prompt is given, add it as the latest user message
            current_conversation.append({"role": "user", "content": prompt})
        elif not current_conversation:
            # Neither prompt nor messages were provided
            raise ValueError("Generator.generate: Either 'prompt' or 'messages' must be provided.")

        # Apply the chat template to format the input for the model
        # `add_generation_prompt=True` tells the tokenizer to add tokens that indicate
        # the start of the assistant's response.
        try:
            input_ids = self.tokenizer.apply_chat_template(
                current_conversation,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device) # Ensure input tensor is on the same device as the model
        except Exception as e:
            print(f"Generator: Error applying chat template: {e}")
            print(f"Current conversation causing error: {current_conversation}")
            raise

        # Prepare generation arguments
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            # "eos_token_id": self.tokenizer.eos_token_id, # Often handled by model/tokenizer config
        }

        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["do_sample"] = True
            gen_kwargs["top_p"] = 0.9  # Common default for top_p when sampling
        else: # Greedy decoding if temperature is 0 or less
            gen_kwargs["do_sample"] = False
        
        # Some models might issue warnings if both temperature and top_p are set with do_sample=False
        # but transformers usually handles this fine.

        try:
            generated_ids = self.model.generate(
                input_ids,
                **gen_kwargs
            )
            # The generated_ids include the input_ids. We need to decode only the new tokens.
            # For chat models, the response starts after the input sequence.
            response_ids = generated_ids[0][input_ids.shape[-1]:]
            result = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        except Exception as e:
            print(f"Generator: Error during model.generate: {e}")
            # Unlike API calls, local errors (e.g., OOM) are unlikely to be fixed by a simple retry.
            # Re-raising the exception is generally safer.
            raise

        # Clean up common markdown formatting, if any
        result = self.remove_substring(result, "```python")
        result = self.remove_substring(result, "```")

        # Append the assistant's response to the conversation history
        new_conversation_history = list(current_conversation) # Start from this call's input history
        new_conversation_history.append(
            {
                "role": "assistant",
                "content": result,
            }
        )
        return result, new_conversation_history


def docstring_from_json(json_file):
    with open(json_file, "r") as file:
        api_data = json.load(file)

    docstring = ""
    for module in api_data.get("modules", []):
        docstring += f'"""\n'
        docstring += f"{module['description']}\n\n"
        if module["arguments"]:
            docstring += "Args:\n"
            for arg in module["arguments"]:
                docstring += (
                    f"    {arg['name']} ({arg['type']}): {arg['description']}\n"
                )
        if "returns" in module:
            docstring += f"\nReturns:\n"
            docstring += f"    {module['returns']['type']}: {module['returns']['description']}\n\"\"\""
        docstring += f"\n{module['name']}("
        args = [arg["name"] for arg in module["arguments"]]
        docstring += ", ".join(args) + ")\n\n"

    return docstring.strip()

def depth_to_grayscale(depth_map):
    # Ensure depth_map is a NumPy array of type float (if not already)
    depth_map = np.array(depth_map, dtype=np.float32)

    # Get the minimum and maximum depth values
    d_min = np.min(depth_map)
    d_max = np.max(depth_map)
    
    # Avoid division by zero if the image is constant
    if d_max - d_min == 0:
        normalized = np.zeros_like(depth_map)
    else:
        normalized = (depth_map - d_min) / (d_max - d_min)
    
    # Scale to 0-255 and convert to unsigned 8-bit integer
    grayscale = (normalized * 255).astype(np.uint8)
    
    return grayscale


def box_image(img, boxes):
    img1 = img.copy()
    draw = ImageDraw.Draw(img1)
    for box in boxes:
        x_0, y_0, x_1, y_1 = box[0], box[1], box[2], box[3]
        draw.rectangle([x_0, y_0, x_1, y_1], outline="red", width=8)

    return img1


def dotted_image(img, points):
    # Scale dot size based on image width
    if isinstance(img, np.ndarray):
        img_width = img.shape[1]
        np_img = img.copy()
        img = Image.fromarray(np_img)
        if img.mode == 'F': # Likely a depth map
            img = depth_to_grayscale(np_img) # Convert to grayscale first
            img = Image.fromarray(img)
            img = img.convert('RGB') # Then convert to RGB for colored dots
    else:
        img_width = img.size[0]

    
    dot_size = max(1, int(img_width * 0.01)) # 1% of image width, ensure at least 1px
    img1 = img.copy()
    if img1.mode != 'RGB': # Ensure image is RGB for colored dots
        img1 = img1.convert('RGB')
        
    draw = ImageDraw.Draw(img1)
    for pt in points:
        x = pt[0]
        y = pt[1]

        # Define ellipse bounding box: (x0, y0, x1, y1)
        # x0 = x - radius, y0 = y - radius, x1 = x + radius, y1 = y + radius
        ellipse_bbox = (x - dot_size, y - dot_size, x + dot_size, y + dot_size)
        draw.ellipse(
            ellipse_bbox,
            fill="red",
            outline="black",
        )
    return img1


def html_embed_image(img, size=300):
    img = img.copy()
    img.thumbnail((size, size))
    with BytesIO() as buffer:
        if img.mode == 'F': # Handle single-channel float images (like depth maps)
            # Convert to a displayable format, e.g., grayscale then RGB
            grayscale_img_array = depth_to_grayscale(np.array(img))
            img_to_save = Image.fromarray(grayscale_img_array).convert('RGB')
        elif img.mode == 'L': # Handle grayscale images
             img_to_save = img.convert('RGB') # Convert to RGB if it's L
        elif img.mode == 'RGBA':
             img_to_save = img.convert('RGB') # Convert RGBA to RGB (removes alpha)
        else:
            img_to_save = img

        if img_to_save.mode != 'RGB' and img_to_save.mode != 'L' and img_to_save.mode != 'P': # common saveable modes
             # If still not a common mode, attempt conversion to RGB as a fallback
             try:
                 img_to_save = img_to_save.convert('RGB')
             except Exception as e:
                 print(f"Warning: Could not convert image mode {img_to_save.mode} for saving. Error: {e}")
                 # Fallback: create a dummy image or return an error placeholder
                 # For now, let JPEG try to handle it or fail.
                 pass

        img_to_save.save(buffer, "jpeg")
        base64_img = base64.b64encode(buffer.getvalue()).decode()
    return (
        f'<img style="vertical-align:middle" src="data:image/jpeg;base64,{base64_img}">'
    )
    

class TimeoutException(Exception):
    pass


def set_devices():
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # Check if mps is available
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        # torch.autocast("cuda", dtype=torch.bfloat16).__enter__() # This is a global setting, usually fine
        # For more controlled casting, it's better to manage dtypes at model loading or operation level.
        # Transformers' `torch_dtype="auto"` handles this well.
        pass # Autocast can be enabled here if desired globally
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if device.type == "cuda" and torch.cuda.get_device_properties(0).major >= 8: # Check for Ampere or newer
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nNote: Support for MPS devices is under active development in PyTorch and Transformers. "
            "Performance and numerical stability may vary compared to CUDA. "
            "See PyTorch and Hugging Face Transformers documentation for MPS specifics."
        )
    return device


def timeout_handler(signum, frame):
    raise TimeoutException(
        "The script took too long to run. There is likely a Recursion Error. Ensure that you are not calling a method with infinite recursion."
    )


def remove_python_text(output):
    substring = "```python"
    if substring in output:
        output = output.replace(substring, "")

    substring = "```"
    if substring in output:
        output = output.replace(substring, "")

    return output