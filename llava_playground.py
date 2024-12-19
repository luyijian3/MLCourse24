import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

def load_llava(model_path):
    model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    processor = LlavaNextProcessor.from_pretrained(model_path, patch_size=model.config.vision_config.patch_size, vision_feature_select_strategy=model.config.vision_feature_select_strategy)
    model.to("cuda")
    return model, processor, model.device
# prepare image and text prompt, using the appropriate prompt template
#url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
#image = Image.open('./images/chart/pie1.jpg')

def response_llava(model, processor, image, text):
# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image"},
            ],
        },
    ]
    image = Image.open(image)
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=300)
    return processor.decode(output[0], skip_special_tokens=True).replace(text,'')


if __name__ == "__main__":
    model, processor, device = load_llava('/data2/MODELS/llava-v1.6-mistral-7b-hf')
    print(f'The model is on {device}')
    response = response_llava(model, processor, './images/chart/bar1.jpg', 'Generate the detailed image description: ')
    print(response)
