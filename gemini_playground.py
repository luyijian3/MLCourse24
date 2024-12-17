import google.generativeai as genai
import os
import time

class Gemini:
    def __init__(self, model = "gemini-1.5-flash"):
        genai.configure(api_key=os.environ["gemini_API_KEY"])
        #genai.configure(api_key=os.environ["gemini_API_KEY_1"])
        self.model = genai.GenerativeModel(model_name=model)

    def get_response(self, image_path, prompt):
        if image_path:
            sample_file = genai.upload_file(path=image_path,
                                        display_name="image_caption sample")
            #print(f"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}")
            try:
                file = genai.get_file(name=sample_file.name)
                _ = file.display_name + sample_file.uri
            except:
                raise Exception('image file upload error')
            response = self.model.generate_content([sample_file, prompt])
            time.sleep(5)
        else:
            # purely text
            response = self.model.generate_content(prompt)
            time.sleep(5)
        return response.text


if __name__ == "__main__":
    gemini = Gemini()   #'gemini-1.5-flash'
    print(gemini.get_response("/data2/luyj/image_caption/images/chart/pie1.jpg", "Describe the image as detailed as possible."))

'''
response = model.generate_content(
    'Write a story about a magic backpack.',
    generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=0.1,
    )
)
'''