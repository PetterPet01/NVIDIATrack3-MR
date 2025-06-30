import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Qwen3Chatbot:
    def __init__(self, model_name="Qwen/Qwen3-30B-A3B"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure a GPU is installed and accessible.")

        print("✅ Using CUDA device:", torch.cuda.get_device_name(0))

        # Use bfloat16 if supported (better performance on A100)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"🧠 Using precision: {dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="flash_attention_2"  # Flash Attention 2 for faster attention
        ).to("cuda")

    def chat(self, user_input, thinking=True):
        messages = [{"role": "user", "content": user_input}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")  # Move inputs to GPU

        # Use recommended sampling parameters for each mode
        gen_kwargs = {
            "max_new_tokens": 512,
            "do_sample": True
        }
        if thinking:
            gen_kwargs.update({
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0
            })
        else:
            gen_kwargs.update({
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0
            })

        outputs = self.model.generate(**inputs, **gen_kwargs)
        output_ids = outputs[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # Try to parse thinking content if present
        try:
            thinking_start = response.index("<tool_response>")
            thinking_content = response[:thinking_start].strip()
            final_response = response[thinking_start + len("<tool_response>"):].strip()
        except ValueError:
            thinking_content = ""
            final_response = response.strip()

        return thinking_content, final_response


def main():
    parser = argparse.ArgumentParser(description="Qwen3 Chatbot CLI")
    parser.add_argument("--thinking", action="store_true", help="Use thinking mode (default)")
    parser.add_argument("--no-thinking", dest="thinking", action="store_false", help="Use non-thinking mode")
    parser.set_defaults(thinking=True)
    args = parser.parse_args()

    print(f"Initializing Qwen3-30B-A3B in {'thinking' if args.thinking else 'non-thinking'} mode...")

    try:
        bot = Qwen3Chatbot(model_name='Qwen/Qwen3-8B')
        print("✅ Ready to chat! Type 'exit' to quit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    break
                thinking_content, response = bot.chat(user_input, thinking=args.thinking)
                if thinking_content:
                    print("🧠 Thinking Content:")
                    print(thinking_content)
                    print()
                print("🤖 Qwen3:", response)
            except KeyboardInterrupt:
                print("\n👋 Exiting chat.")
                break
            except Exception as e:
                print(f"❌ Error during generation: {e}")
                break
    except RuntimeError as e:
        print(f"🚫 Fatal Error: {e}")


if __name__ == "__main__":
    main()