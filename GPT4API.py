import openai

openai.api_key = "MgxxsVTy2w05wkpW2UkgnUTZbtO9nzzQ"
openai.api_base = "https://gptproxy.llmpaas.woa.com/v1" #只增加这一行即可
# ret = openai.Completion.create(model="gpt-4", prompt="Say this is a test", max_tokens=7, temperature=0)
# print(ret)


def generate_text(text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": text}
        ],
        temperature=1
    )
    return response

question='''
In the diagram, $\triangle ABC$ is right-angled at $C$. Also, points $M$, $N$ and $P$ are the midpoints of sides $BC$, $AC$ and $AB$, respectively.  If the area of $\triangle APN$ is $2\mbox{ cm}^2$, then what is the area, in square centimeters, of $\triangle ABC$? [asy]
size(6cm);
import olympiad;
pair c = (0, 0); pair a = (0, 4); pair n = (0, 2); pair m = (4, 0); pair b = (8, 0); pair p = n + m;
draw(a--b--c--cycle); draw(n--p--m);
draw(rightanglemark(a,c,b));
label("$C$", c, SW); label("$A$", a, N); label("$N$", n, W); label("$M$", m, S); label("$B$", b, E); label("$P$", p, NE);
[/asy]
Provide a step-by-step reasoning before providing your answer.
'''

import json
import os
import re
import time
from tqdm import tqdm
import sys
math_pattern = re.compile(r'\\boxed{(.*)}')
fout = open('./GPT4API_prealgebra.jsonl','w')


for i in tqdm(range(970, 2090)):
    file_name = os.path.join('/data/home/tingchenfu/abel/MATH/test/prealgebra','{}.json'.format(i))
    if not os.path.exists(file_name):
        continue
    question = json.load(open(file_name))['problem']
    solution = json.load(open(file_name))['solution']
    target = re.findall(math_pattern,solution)[-1]
    response = generate_text(question)['choices'][0]['message']['content']

    fout.write(json.dumps({'index':i, 'GPT4APIresponse':response,'target':target},ensure_ascii=False)+'\n')
    fout.flush()