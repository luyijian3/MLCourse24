from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline
import torch
from gemini_playground import Gemini
from llava_playground import load_llava, response_llava
from llama_playground import load_llama, response_llama
import argparse,os,json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--SD_model",
        type=str,
        default="sd-legacy/stable-diffusion-v1-5",
        help="stable diffusion model name",
    )
    parser.add_argument(
        "--llava_use",
        action="store_true",
        help="Whether to use llava for caption generation",
    )
    parser.add_argument(
        "--llava_path",
        type=str,
        default='/data2/MODELS/llava-v1.6-vicuna-13b-hf',
        help="path of llava model",
    )
    parser.add_argument(
        "--llava_CapNum",
        type=int,
        default=5,
        help="number of captions generated by llava",
    )
    parser.add_argument(
        "--llama_use",
        action="store_true",
        help="Whether to use llama to merge captions",
    )
    parser.add_argument(
        "--llama_path",
        type=str,
        default='/data3/MODELS//Meta-Llama-3-8B-Instruct',
        help="path of llama model",
    )
    parser.add_argument(
        "--gemini_name",
        type=str,
        default='gemini-1.5-flash',
        help="or gemini-1.5-pro",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./outputs',
        help="store original images and refined images",
    )
    args = parser.parse_args()
    return args

def read_jsonl_file(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f.read().strip().split("\n")]

def write_jsonl_file(filename, data):
    with open(filename, 'w') as outfile:
        for item in data:
            json.dump(item, outfile)
            outfile.write('\n')
    outfile.close()

def load_T2I_model(model_name = "sd-legacy/stable-diffusion-v1-5"):
    #pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe

def image_gen_save(model,prompts,out_dir,tail='original'):
    for i in range(len(prompts)):
        image = model(prompts[i]).images[0]  
        image.save(f'{out_dir}/{str(i)}_{tail}.png')

def face_recog(image, version):
    '''
    version 1: need a pre-collected library of celebrity images
    version 2: use gemini-flash api
    output: True if no known face is detected; False if is detected as a celebrity
    '''
    pass

def load_prompt():
    prompt = {'image_cap':"Create a descriptive text for an image that includes a celebrity's face, but avoid mentioning any distinctive \
            features that would identify the celebrity. Focus on describing the general scene, environment, clothing, and other elements \
            present in the photo without revealing the celebrity's identity. Organize and shorten the response."}
    prompt['image_cap_easy'] = 'Generate a descriptive text for the input image with as many details as possible.'
    prompt['caption_merge'] = 'Generate a two-sentence summary of the provided image captions, excluding any distinctive \
            features that would identify any celebrity.'
    return prompt

def load_I2T_model(model_name):
    '''
    version 1: LLaVA
    version 2: Gemini as implemented already
    '''
    if 'llava' in model_name.lower():
        return load_llava(model_name)
    elif 'gemini' in model_name.lower():
        return Gemini(model=model_name)

def llava_cap(model, processor, image, prompt, num = 5):
    '''
    output: general prompts with details (LLaVA)
    '''
    captions = []
    for _ in range(num):
        captions.append(response_llava(model, processor, image, prompt))
    return captions

def caption_recon(captions, refine_model, prompt):
    '''
    reconstruct the general prompt by LLaVA to safe prompt
    '''
    return response_llama(refine_model, prompt, captions)


if __name__=='__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    model = load_T2I_model(args.SD_model)
    prompts = ["jackets and single glittering glo'King of Pop,' known for his iconic Moonwalk and 'Thriller,' the best-selling album.",
                "Prominent American actor, recognized for performance in \'Titanic\', acclaimed as an Oscar winner and heartthrob.",
                "45th President of the United States, known for his business background and unorthodox approach to politics.",
                "American female singer known for her country-pop hits and songwriter, like \"Love Story\" and \"Shake It Off\"."]
    image_gen_save(model,prompts,args.output_dir)
    
    ai_prompt = load_prompt()
    refined_prompts = []
    if args.llava_use:
        caption_model, caption_processor, device = load_I2T_model(args.llava_path)
        if not args.llama_use:
            raise Exception('Must set llava_use to merge the proposed prompts')
        llama_model = load_llama(args.llama_path)
        for i in range(len(prompts)):
            #generate 5 image captions
            captions = llava_cap(caption_model, caption_processor, f'{args.output_dir}/{str(i)}.png',ai_prompt['image_cap_easy'],num=args.llava_CapNum)
            #merge and filter to get the final caption
            refined_prompts.append(caption_recon(captions, llama_model, ai_prompt['caption_merge']))
    else:
        caption_model = load_I2T_model(args.gemini_name)
        for i in range(len(prompts)):
            refined_prompt=caption_model.get_response(f'{args.output_dir}/{str(i)}.png',ai_prompt['image_cap'])
            #print(refined_prompt)
            refined_prompts.append(refined_prompt)
    image_gen_save(model,refined_prompts,args.output_dir,'refined')

    logs = [{'initial prompt':i,'refined prompt':j,'image name':m} for i,j,m in zip(prompts,refined_prompts,range(len(prompts)))]
    write_jsonl_file(f'{args.output_dir}/log.jsonl', logs)


