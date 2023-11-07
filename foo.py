# import json
# from promptsource.templates import DatasetTemplates
# for template in DatasetTemplates('gsm8k').templates.values():
#     if template.metadata.original_task:
#         break

# print(template)

# for line in  open('/data/home/tingchenfu/abel/data/test/test.jsonl').readlines():
#     example = json.loads(line)
#     templated=template.apply(example)
#     print(templated[0])
#     print(templated[1])
#     break


###########################################################################
##download model
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf",torch_dtype=torch.float16)
# model.save_pretrained('./Llama-2-13b-hf')
#############################################################################

## download dataset
# from datasets import load_dataset
# dataset = load_dataset('gsm8k','main')
# train_dataset = dataset['train']
# train_dataset.to_json('./data/gsm8k/train.jsonl')

###########################################################################

### dataset statistics
# import json
# import numpy as np
# from transformers import LlamaTokenizer
# qlength=[]
# alength=[]
# count=0
# tokenizer = LlamaTokenizer.from_pretrained('/apdcephfs/share_916081/shared_info/tingchenfu/PLM/Llama-2-7b-hf')
# for line in open('/data/home/tingchenfu/abel/data/test/test.jsonl').readlines():
#     data=json.loads(line)
#     if data['source']!='MATH':
#         continue
#     count+=1
#     qlength.append(len(tokenizer(data['question'])['input_ids']))
#     alength.append(len(tokenizer(data['answer'])['input_ids']))

# print(max(qlength))
# print(sum(qlength)/len(qlength))
# print(max(alength))
# print(sum(alength)/len(alength))
# print(np.sum(np.array(alength)>1024))
# print(count)

#############################################################################
# import json
# f=open('/data/home/tingchenfu/abel/dump/gsm8k/WizardMath-7B-V1.0.jsonl_','w')
# for line in open('/data/home/tingchenfu/abel/dump/gsm8k/WizardMath-7B-V1.0.jsonl').readlines():
#     data=json.loads(line)
#     # data['response']=data['response'].replace('</s>','').strip('\n')
#     data['response'] = data['response'].replace('\n','').replace('</s>','')
#     f.write(json.dumps(data,ensure_ascii=False)+'\n')


################################################################################
### dataset preprocess
# import re
# pattern = re.compile("#### (.*)$")
# import json
# fout=open('/data/home/tingchenfu/abel/data/gsm8k/train.jsonl_','w')
# for line in  open('/data/home/tingchenfu/abel/data/gsm8k/train.jsonl').readlines():
#     data=json.loads(line)
#     preds = re.findall(pattern, data['answer'])
#     pred = preds[-1] if len(preds) >= 1 else ""
#     if "</s>" in pred:
#         pred = pred[:-4]
#     pred = pred.strip()
#     if pred =='1,080':
#         pred = '108'
#     if pred == '850,000':
#         pred = '850000'
#     if ',' in pred:
#         print(pred)
#         pred = pred.replace(',','')
#     assert int(pred) == float(pred)
#     data['target'] = pred
#     fout.write(json.dumps(data,ensure_ascii=False)+'\n')

# fout.close()

#################################################################################

# MATH sanity check
# import json
# import os
# question_set=set()
# type_set=set()
# level_set=set()
# for line in open('/data/home/tingchenfu/abel/data/test/test.jsonl').readlines():
#     data=json.loads(line)
#     if data['source']!='MATH':
#         continue
#     question_set.add(data['question'])
#     type_set.add(data['tag']['type'])
#     level_set.add(data['tag']['level'])


# print(len(question_set))
# not_exist=0
# dir = '/data/home/tingchenfu/abel/MATH/test/precalculus'
# for file in os.listdir(dir):
#     data=json.load(open(os.path.join(dir,file)))
#     try:
#         assert data['problem'] in question_set
#     except:
#         not_exist+=1
#         #break
#         #print(data['problem'])
        
# print('here is ok')
# print(not_exist)
# print(type_set)
# print(level_set)


####################################################################################
### MATH training set process
# import os
# import json
# import re
# pattern = re.compile(r'\\boxed{(.*)}')
# fout=open('/data/home/tingchenfu/abel/data/math/train.jsonl','w')
# dir = '/data/home/tingchenfu/abel/MATH/train'
# for subdir in os.listdir(dir):
#     for file in os.listdir(os.path.join(dir, subdir)):
#         data = json.load(open(os.path.join(dir,subdir,file)))
#         data['question'] = data['problem']
#         data.pop('problem')
#         data['answer'] = data['solution']
#         data.pop('solution')
#         pred = re.findall(pattern, data['answer'])
#         pred = pred[-1] if len(pred) >= 1 else ""
#         data['target']=pred
#         data['tag'] = {'level':data['level'],'type':data['type']}
#         data.pop('level')
#         data.pop('type')
#         fout.write(json.dumps(data,ensure_ascii=False)+'\n')
# fout.close()


######################################################################################
# import json
# fout=open('/data/home/tingchenfu/abel/data/math/test.jsonl','w')
# for line in open('/data/home/tingchenfu/abel/data/test/test.jsonl').readlines():
#     data = json.loads(line)
#     if data['source']!='MATH':
#         continue
#     data.pop('source')
#     fout.write(json.dumps(data,ensure_ascii=False)+'\n')
# fout.close()
##########################################################################################

### generate picture
# import re
# import json
# pattern = re.compile(r'\[asy\](.+)\[\/asy\]')
# index=[]
# count={}
# for no in range(0,24733):
#     try:
#         f=open('/data/home/tingchenfu/abel/MATH/test/geometry/{}.json'.format(no))
#     except:
#         continue
#     data=json.load(f)
#     question=data['problem'].replace('\n','')
#     pic = re.findall(pattern,question)
#     pic= pic[-1] if len(pic)>=1 else "No picture!"
#     #print(question)
#     #if '[asy]' in question:
#     try:
#         count[data['level']]+=1
#     except:
#         count[data['level']]=1
#     #print(no)
#     #print('[asy]' in question)
#     #print('[asy]\n{}[/asy]'.format(pic.replace(';',';\n')))
#     #print('------------------------------------------------')
#     #stop=input('>>>')
# print(count)

##########################################################################################


#### generate GPT-4 prompt
# import re
# import json
# count_pic=0
# pattern = re.compile(r'\[asy\](.+)\[\/asy\]')
# for no in range(503,24733):
#     try:
#         f=open('/data/home/tingchenfu/abel/MATH/test/geometry/{}.json'.format(no))
#     except:
#         continue
#     question = json.load(f)['problem']
#     if '[asy]' not in question:
#         continue
#     count_pic+=1
#     #print(no)
#     #print('*****************************')
#     print('@@')
#     print(question.strip('\n')+'\nProvide a step-by-step reasoning before providing your answer.')
#     #print("-----------------------------")    

#########################################################################################

### generate GPT-4v prompt
# import re
# import json
# count_pic=0
# pattern = re.compile(r'\[asy\](.+)\[\/asy\]')
# for no in range(503,24733):
#     try:
#         f=open('/data/home/tingchenfu/abel/MATH/test/geometry/{}.json'.format(no))
#     except:
#         continue
#     question = json.load(f)['problem']
#     if '[asy]' not in question:
#         continue
#     pic_start = question.find('[asy]')
#     pic_end = question.find('[/asy]')
#     question = question[:pic_start] + question[pic_end+6:]
#     question = question.replace('following ','').replace('below','above')
#     count_pic+=1
#     #print(no)
#     #print('*****************************')
#     print('@@')
#     print(question.strip('\n')+'\nProvide a step-by-step reasoning before providing your answer.')
#     #print("-----------------------------")    


######################################################################################
#####postprocess response

# f=open('/data/home/tingchenfu/abel/GPT4prompt_part2')
# content = f.read()
# gpt4_prompts = content.split('@@')
# gpt4_prompts.pop(0)
# print(len(gpt4_prompts))

# f=open('/data/home/tingchenfu/abel/GPT4Vprompt_part2')
# content= f.read()
# gpt4v_prompts = content.split('@@')
# gpt4v_prompts.pop(0)
# print(len(gpt4v_prompts))

# f=open('/data/home/tingchenfu/abel/GPT4response_part2_copy')
# content = f.read()
# gpt4_responses = content.split('@@')
# gpt4_responses.pop(0)
# print(len(gpt4_responses))
# f.close()

# f=open('/data/home/tingchenfu/abel/GPT4Vresponse_part2_copy')
# content = f.read()
# gpt4v_responses = content.split('@@')
# gpt4v_responses.pop(0)
# print(len(gpt4_responses))
# f.close()

# import json
# gpt4_results=[]
# f=open('/data/home/tingchenfu/abel/GPT4result_part2')
# for line in f.readlines():
#     gpt4_results.append(int(line.split(' ')[1].strip('\n')))
# gpt4v_results=[]
# f=open('/data/home/tingchenfu/abel/GPT4Vresult_part2')
# for line in f.readlines():
#     gpt4v_results.append(int(line.split(' ')[1].strip('\n')))


# indices=[526, 528, 530, 539, 547, 549, 551, 566, 574, 577, 579, 584, 594, 621, 633, 634, 635, 639, 656, 665, 670, 685, 693, 697, 702, 706, 707, 710, 733, 743, 756, 763, 766, 773, 775, 777, 780, 782, 790, 795, 797, 801, 808, 813, 818, 819, 826, 827, 843, 846, 855, 862, 869, 876, 898, 902, 916, 922, 928, 930, 935, 943, 947, 949, 953, 956, 971, 972, 975, 981, 986, 992, 996, 1003, 1005, 1010, 1014, 1018, 1026, 1038, 1043, 1044, 1051, 1052, 1060, 1062, 1064, 1067, 1077, 1083, 1084, 1087, 1092, 1096, 1115, 1117, 1118, 1119, 1120, 1126, 1129, 24076, 24536, 24733]
# fout=open('/data/home/tingchenfu/abel/visual.jsonl','w')
# for index, gpt4_prompt, gpt4_response,gpt4_result , gpt4v_prompt, gpt4v_response, gpt4v_result in zip (indices, gpt4_prompts, gpt4_responses, gpt4_results,  gpt4v_prompts, gpt4v_responses, gpt4v_results):
#     fout.write(json.dumps({'index':index,'GPT4prompt':gpt4_prompt[1:], 'GPT4response':gpt4_response,'GPT4result':gpt4_result, 'GPT4Vprompt':gpt4v_prompt[1:],'GPT4Vresponse':gpt4v_response, 'GPT4Vresult':gpt4v_result},ensure_ascii=False)+'\n')

# print(sum(gpt4_results))
# print(sum(gpt4v_results))
#######################################################################################
# import os
# indices=[526, 528, 530, 539, 547, 549, 551, 566, 574, 577, 579, 584, 594, 621, 633, 634, 635, 639, 656, 665, 670, 685, 693, 697, 702, 706, 707, 710, 733, 743, 756, 763, 766, 773, 775, 777, 780, 782, 790, 795, 797, 801, 808, 813, 818, 819, 826, 827, 843, 846, 855, 862, 869, 876, 898, 902, 916, 922, 928, 930, 935, 943, 947, 949, 953, 956, 971, 972, 975, 981, 986, 992, 996, 1003, 1005, 1010, 1014, 1018, 1026, 1038, 1043, 1044, 1051, 1052, 1060, 1062, 1064, 1067, 1077, 1083, 1084, 1087, 1092, 1096, 1115, 1117, 1118, 1119, 1120, 1126, 1129, 24076, 24536]
# for index in indices:
#     oldpath='/data/home/tingchenfu/abel/MATH/test/geometry/{}.json'.format(index)
#     newpath = '/data/home/tingchenfu/abel/MATH/test/picture_geometry'
#     order = 'cp '+ oldpath + ' '+ newpath
#     os.system(order)
#     #print(order) 
# # fout=open('GPT4Vresult','w')
# # for index in indices:
# #     fout.write('@@'+str(index)+' \n')


######################################################################################

import json
count={}
count_all=0
f=open('/data/home/tingchenfu/abel/data/test/test.jsonl')
for line in f.readlines():
    data=json.loads(line)
    if data['source']== 'MATH' and data['tag']['type']=='Geometry' and '[asy]' in data['question'] and '[/asy]' in data['question']:
        count_all+=1
        try:
            count[data['tag']['level']]+=1
        except:
            count[data['tag']['level']]=1
        #print(data['tag']['type'])
        #print(data['question'])

print(count)
# print(count_all)

###########################################################################################
# import os
# import json
# import re
# from tqdm import tqdm
# math_pattern = re.compile(r'\\boxed{(.*)}')
# for i in range(0, 24777):
#     file_name = os.path.join('/data/home/tingchenfu/abel/MATH/test/geometry','{}.json'.format(i))
#     if not os.path.exists(file_name):
#         continue
#     question = json.load(open(file_name))['problem']
#     if '[asy]' not in question:
#         continue
#     solution = json.load(open(file_name))['solution']
#     target = re.findall(math_pattern,solution)[-1]
#     print('@@ {} {}'.format(i,target))
#     #print()

###################################################################################################

import os
import json
index2level = dict()

dir = '/data/home/tingchenfu/abel/MATH/test/geometry'
for file in os.listdir(dir):
    data = json.load(open(os.path.join(dir,file)))
    index2level[int(file.replace('.json',''))] = data['level']

# f=open('/data/home/tingchenfu/abel/GPT4V+result')
# for line in f.readlines():
#     line=line.strip('\n').strip('@')
#     index, result = line.split(' ')
#     if int(result) == 1:
#         try:
#             count[index2level[int(index)]]+=1
#         except:
#             count[index2level[int(index)]]=1

# print(count)

#################################################################


#################################################################
# f=open('/data/home/tingchenfu/abel/GPT4V+result')
# index2result=dict()
# for line in f.readlines():
#     line = line.strip('\n').strip('@')
#     index, result = line.split(' ')
#     index2result[int(index)] = int(result) 

# f=open('/data/home/tingchenfu/abel/figure_geometry.jsonl')
# for line in f.readlines():
#     data = json.loads(line)
#     if (int(data['GPT4result']) == 1  or int(data['GPT4Vresult']) == 1) and index2result[data['index']] == 0:
#         print(data['index'])


###############################################################
import json
count=dict()
f=open('/data/home/tingchenfu/abel/figure_geometry.jsonl')
for line in f.readlines():
    data = json.loads(line)
    try:
        count[index2level[int(data['index'])]]+=int(data['GPT4Vresult'])
    except:
        count[index2level[int(data['index'])]]=int(data['GPT4Vresult'])
    
print(count)