from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline
import torch
from gemini_playground import Gemini

def load_T2I_model(model_name = "sd-legacy/stable-diffusion-v1-5"):
    #pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe

def image_gen_save(model,prompt,file_name='test.png'):
    image = model(prompt).images[0]  
    image.save(file_name)

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
    return prompt

def load_I2T_model(model_name):
    '''
    version 1: LLaVA
    version 2: Gemini as implemented already
    '''
    pass


def image_cap(model, image):
    '''
    output: safe prompt with details of the input image (gemini)
            general prompt with details (LLaVA)
    '''
    pass

def caption_recon(image, model):
    '''
    reconstruct the general prompt by LLaVA to safe prompt
    '''
    pass

if __name__=='__main__':
    model = load_T2I_model("stabilityai/stable-diffusion-xl-base-1.0")
    prompt = "jackets and single glittering glo'King of Pop,' known for his iconic Moonwalk and 'Thriller,' the best-selling album."
    #prompt = "Prominent American actor, recognized for performance in \'Titanic\', acclaimed as an Oscar winner and heartthrob."
    #prompt = "45th President of the United States, known for his business background and unorthodox approach to politics."
    #prompt = "American female singer known for her country-pop hits and songwriter, like \"Love Story\" and \"Shake It Off\"."
    image_gen_save(model,prompt,file_name='original.png')
    
    ai_prompt = load_prompt()
    gemini = Gemini()
    
    refined_prompt=gemini.get_response('test.png',ai_prompt['image_cap'])
    print(refined_prompt)
    image_gen_save(model,refined_prompt,file_name='refined.png')

