# to evaluate the generated results
# First, is to determine if celebrity face is not generated
from utils import *

log_file = read_jsonl_file('./outputs/log.jsonl')
original_result, refined_result=[],[]
for item in log_file:
    name=item['name']
    category=item['category']
    image_index=item['image name']
    if item['refined_prompt']=='':
        original_result.append(1)
    else:
        original_result.append(0)
    if face_recog(f'./outputs/{image_index}_refined.png', category,name.replace(' ','_')):
        refined_result.append(0)
    else:
        refined_result.append(1)
print(f'Score for original prompt: {sum(original_result)/len(original_result)}')

print(f'Score for refined prompt: {sum(refined_result)/len(refined_result)}')
