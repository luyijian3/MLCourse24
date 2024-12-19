import transformers
import torch

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"

# huggingface-cli download MMInstruction/Qwen-VL-ArXivCap --cache-dir /data2/MODELS

prompt_category = 'Pick the image category from the following list according to the image caption: chart, people, artistic, table, document, object, place, animal. \
                                        Try to fit each image in the above categories, for example, cat is in animal, book and car are in object, map and logo are in document. \
                                        But if you think the correct category is not included in the list, use up to 2 words to name the category. Do not return image or anything else. \
                                        Image caption: '
def load_llama(model_id = "/data3/MODELS//Meta-Llama-3-8B-Instruct"):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline

def llama_response(pipeline, image_caption):
    messages = [
        {"role": "system", "content": "You are an expert in image labeling."},
        {"role": "user", "content": prompt_category + image_caption},
    ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=pipeline.tokenizer.eos_token_id
    )
    return outputs[0]["generated_text"][-1]['content']
