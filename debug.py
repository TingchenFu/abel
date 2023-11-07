import json
f=open('/data/home/tingchenfu/abel/MATH/test/geometry/1018.json')
question= json.load(f)['problem']
print(question)