from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM
from peft import PeftModel
import os
import re
import json
import jsonlines
import argparse
import torch
from tqdm import tqdm
import sys
import pdb
from math_normalization import *
from str2bool import str2bool
gsm8k_pattern  = re.compile(r"#### (.*)$")
math_pattern = re.compile(r'\\boxed{(.*)}')

def get_results(pred_file, dataset_name):
    def test_answer(pred_str, ans_str):
        if "Question" in pred_str:
            pred_str = pred_str.split("Question")[0]
        if 'question' in pred_str:
            pred_str = pred_str.split('question')[0]

        preds = re.findall(gsm8k_pattern, pred_str)
        pred = preds[-1] if len(preds) >= 1 else ""

        if pred == "":
            preds = re.findall(math_pattern,pred_str)
            pred = preds[-1] if len(preds) >= 1 else ""

        if pred.endswith('</s>'):
            pred = pred[:-4]
        if pred.endswith('</s'):
            pred = pred[:-3]
        if pred.endswith('</'):
            pred = pred[:-2]
        
        gold = ans_str
        pred = normalize_final_answer(pred)
        gold = normalize_final_answer(gold)
        return check_sympy_equivalence(gold, pred), pred, gold
    
    def parse_pred_ans(preds_str, golds_str, properties_list):
        num_q = 0
        acc = 0
        results = []
        preds = []
        golds = []
        correct_table = {}
        cnt_table = {}
        source_set = set()
        for pred_str, gold_str, properties in tqdm(zip(preds_str, golds_str, properties_list), total=len(preds_str)):
            num_q += 1
            result, pred, gold = test_answer(pred_str, gold_str)
            results.append(result)
            preds.append(pred)
            golds.append(gold)
            if result:
                acc += 1
            source = properties['source']
            tag = properties['tag']
            source_set.add(source)
            if source not in correct_table.keys():
                correct_table[source] = 1 if result else 0
                cnt_table[source] = 1
            else:
                correct_table[source] = (correct_table[source] + 1) if result else correct_table[source]
                cnt_table[source] += 1
            for key in tag.keys():
                value = tag[key]
                value = source+","+key+"__"+value
                if value not in correct_table.keys():
                    correct_table[value] = 1 if result else 0
                    cnt_table[value] = 1
                else:
                    correct_table[value] = (correct_table[value] + 1) if result else correct_table[value]
                    cnt_table[value] += 1
        print('num_q %d correct %d ratio %.4f' % (num_q, acc, float(acc / num_q)))
        acc_table = {}
        for key in correct_table.keys():
            acc_table[key] = correct_table[key] / cnt_table[key]
        acc_table = list(zip(acc_table.keys(), acc_table.values()))
        acc_table.sort(key=lambda x: x[0])
        for key, acc in acc_table:
            if key in source_set:
                print(key+" : "+str(acc))
            else:
                print("    " + key.split(",")[-1]+ " : " + str(acc))
        return results, preds, golds

    if dataset_name in ['all', 'gsm8k', 'math', 'mathgpt', 'gsm8k_robust']:
        golds_str = []
        properties = []
        with open('./data/test/test.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if dataset_name != "all":
                    if json.loads(line)['source'].lower() == dataset_name:
                        golds_str.append(json.loads(line)['target'])
                        properties.append({"source": json.loads(line)['source'], "tag": json.loads(line)['tag']})
                else:
                    golds_str.append(json.loads(line)['target'])
                    properties.append({"source": json.loads(line)['source'], "tag": json.loads(line)['tag']})
        preds_str = []
        with open(pred_file, 'r', encoding='utf-8') as f:
            for line in f:
                preds_str.append(json.loads(line)['response'])
        results, preds, golds = parse_pred_ans(preds_str, golds_str, properties)
        with open(pred_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        for i, line in enumerate(data):
            line['pred'] = preds[i]
            line['gold'] = golds[i]
            line['result'] = results[i]

        # Save the updated list of dictionaries back to the jsonl file
        with open(pred_file, 'w') as file:
            for item in data:
                file.write(json.dumps(item) + '\n')

    else:
        raise NotImplementedError("Evaluation not supported.")


def get_raw_inputs(dataset_name):
    # in this function, we will get the raw queries for a target dev set
    data = []
    if dataset_name in ['all', 'gsm8k', 'math', 'mathgpt', 'gsm8k_robust']:
        with open('./data/test/test.jsonl') as f:
            for line in jsonlines.Reader(f):
                data.append(line)
        if dataset_name != 'all':
            data = [line for line in data if line['source'].lower() == dataset_name]
    else:
        raise ValueError

    prompt_list = [line['question'] for line in data]
    return prompt_list

def get_example(dataset_name,n_example,seed=None):
    
    examples=[]
    for line in open('./data/{}/train.jsonl'.format(dataset_name)).readlines():
        examples.append(json.loads(line))
    if seed:
        return examples[seed:(seed+n_example)%len(examples)]
    else:
        return random.choices(examples,k=n_example)
    


template_mapping = {
    "math-single": "Question:\n{question}\nAnswer:\nLet's think step by step. {answer}",
    "alpaca": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response: Let’s think step by step. {answer}"
}

if __name__ == '__main__':
    # set args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--peft_model_path',type=str,default=None)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--output_file', type=str, default='output.json')
    parser.add_argument('--stop', type=str, nargs='+', default=[], help="you can pass one or multiple stop strings to halt the generation process.")
    parser.add_argument('--dataset_name', type=str, default='all')
    parser.add_argument('--prompt_type', type=str, default='math-single')
    parser.add_argument('--sample_num', type=int, default=-1, )
    parser.add_argument('--eval_only', type=str2bool, default=False)
    parser.add_argument('--max_num_batched_tokens', type=int, default=4096)
    parser.add_argument('--n_example', type=int, default=0)
    parser.add_argument("--seed", type=int,default=0)
    parser.add_argument("--figure_geometry_only",type=str2bool,default=False)
    args = parser.parse_args()

    for key,value in vars(args).items():
        print("{} == {}".format(key,value))
    import random
    random.seed(args.seed)

    # if args.eval_only == False:
    if not os.path.exists(args.output_file):
        print("not cached! decode")
        # part 1 we set the model
        if args.peft_model_path:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
            peft_model = PeftModel.from_pretrained(model, args.peft_model_path)
            peft_model = peft_model.merge_and_unload()
            peft_model.save_pretrained(os.path.join(args.peft_model_path,'peft_intergrated'))
            print("PEFT intergrated!!")
           
        num_gpus = torch.cuda.device_count()
        another_args = {'max_num_batched_tokens': args.max_num_batched_tokens} 
        llm = LLM(model =  os.path.join(args.peft_model_path,'peft_intergrated') if args.peft_model_path else args.model_name_or_path,
                tokenizer= args.model_name_or_path, 
                tensor_parallel_size=num_gpus,
                **another_args)
        print('>>>>>> model loaded')
        # part 2 we set the sampling params
        sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                                            stop=args.stop, presence_penalty=args.presence_penalty,
                                            frequency_penalty=args.frequency_penalty)

        # part 3 we prepare raw queries and wrap them with target prompt
        raw_queries = get_raw_inputs(args.dataset_name)
        if args.figure_geometry_only:
            assert 'geometry' in args.output_file
            raw_queries = [x for x in raw_queries if '[asy]' in x]
        


        template = template_mapping[args.prompt_type]
        prefix=''
        examples = get_example(args.dataset_name,args.n_example)
        for example in examples:
            prefix += template.format(question = example['question'], answer = example['answer'])
        processed_prompts = [prefix+  template.format(question = query, answer ='') for query in raw_queries]
        processed_prompts = processed_prompts[:args.sample_num] if args.sample_num > 0 else processed_prompts
        print(processed_prompts[:2])
        # part 4 we generate, note that vllm is async so extra sorting is needed
        outputs = llm.generate(processed_prompts, sampling_params)
        sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
        print('>>>>>> generation done')

        # part 5 we save the results, always be {'id':id,'response':response}
        # if dir of output file is not exist, it will be created automatically
        if not os.path.exists(os.path.dirname(args.output_file)):
            os.makedirs(os.path.dirname(args.output_file))
        with open(args.output_file, "w") as f:
            for id, output in enumerate(sorted_outputs):
            # note that `prompt`s are the wrapped ones
                f.write(json.dumps({'id': id, 'prompt': output.prompt, 'response': output.outputs[0].text}) + '\n')
        print('>>>>>> writing prediction done')

    # part 6 evaluate, I guess this should be done in a separate script
    get_results(args.output_file, args.dataset_name)
    print('>>>>>> evaluation done')

    os.system('rm -rf '+os.path.join(args.peft_model_path,'peft_intergrated'))
    print('>>>>>> PEFT intergrated model removed')