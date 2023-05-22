import json
from collections import Counter

with open(r'E:\Code\FinancialReportingTyping\data\train_data.json', encoding='utf-8') as f:
    train_data = json.load(f)


print(f'训练样本总数为：{len(train_data)}')
c = Counter([x['type'] for x in train_data])  # 各个类别的样本数量
for ele, freq in c.items():
    print(ele,freq)

# with open(r'E:\Code\FinancialReportingTyping\data\eval_data.json', encoding='utf-8') as f:
#     eval_data = json.load(f)
# print(len(eval_data))
# print(eval_data[:3])