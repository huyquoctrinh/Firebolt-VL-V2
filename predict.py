import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token
from transformers.generation.streamers import TextIteratorStreamer

from PIL import Image

import requests
from io import BytesIO

from threading import Thread
import os

MODEL_PATH = "/home/mamba/ML_project/Testing/Huy/joint_vlm/LLaVA/checkpoints/llava-v1.5-7b-moe-finetune/checkpoint-15000"


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


class Predictor:
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        disable_torch_init()
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=MODEL_PATH,
            model_base=None,
            model_name="llava-v1.5-7b-moe",
            load_8bit=False,
            load_4bit=False,
        )

    def predict(
        self,
        image,
        prompt,
        top_p=1.0,
        temperature=0.2,
        max_tokens=1024,
    ):
        """Run a single prediction on the model"""

        conv_mode = "v1"
        conv = conv_templates[conv_mode].copy()

        image_data = load_image(str(image))
        image_tensor = self.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().cuda()

        # just one turn, always prepend image token
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=20.0)
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                use_cache=True)
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to use for text generation")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()

    predictor = Predictor()
    predictor.setup()
    # for output in predictor.predict(args.image, args.prompt, args.top_p, args.temperature, args.max_tokens):
        # print("Output:", output)
    output = predictor.predict(args.image, args.prompt, args.top_p, args.temperature, args.max_tokens)
    print("Output:", output)
    print()
