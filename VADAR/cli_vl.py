import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import argparse
import math
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# Constants for image preprocessing
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def dynamic_preprocess(image, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(1, max_num + 1) 
        for i in range(1, n + 1) for j in range(1, n + 1) 
        if i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
    closest_ratio = min(target_ratios, key=lambda r: abs(r[0]/r[1] - aspect_ratio))
    target_width = image_size * closest_ratio[0]
    target_height = image_size * closest_ratio[1]
    blocks = closest_ratio[0] * closest_ratio[1]

    resized_img = image.resize((target_width, target_height))
    images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        images.append(resized_img.crop(box))
    if use_thumbnail and len(images) > 1:
        images.append(image.resize((image_size, image_size)))
    return images

def load_image(image_path, input_size=448, max_num=12):
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, max_num=max_num, image_size=input_size)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)

def main():
    parser = argparse.ArgumentParser(description="CLI demo for InternVL3-9B")
    parser.add_argument("--image", type=str, help="Path to image file (optional)")
    args = parser.parse_args()

    # Load tokenizer and model
    model_name = "OpenGVLab/InternVL3-9B"
    commit_hash = "7cd614e9f065f6234d57280a1c395008c0b3a996"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False, revision=commit_hash)

    print("Loading model (this may take a few moments)...")
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        revision=commit_hash
    ).eval().cuda()

    print("✅ Model loaded successfully.")

    # Prepare image input if given
    pixel_values = None
    if args.image:
        print(f"Processing image: {args.image}")
        try:
            pixel_values = load_image(args.image).to(torch.bfloat16).cuda()
            print(f"Loaded image with {pixel_values.shape[0]} tiles.")
        except Exception as e:
            print(f"[Error] Could not process image: {e}")
            exit(1)

    # Chat loop
    history = None
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    print("\n💬 Chat started. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        if pixel_values is not None:
            user_input = "<image>\n" + user_input

        response, history = model.chat(
            tokenizer,
            pixel_values=pixel_values if pixel_values is not None else None,
            question=user_input,
            generation_config=generation_config,
            history=history,
            return_history=True
        )

        print(f"Assistant: {response}")

if __name__ == "__main__":
    main()