#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

# 读取第一行数据
with open('/Users/owan/code_project/30_GraduationThesis/1_work1/InstructOCR/data/nfv5_3125/train.txt', 'r') as f:
    line = f.readline()
    data = json.loads(line)

print("Entity字段示例：")
print("=" * 50)

for i, ann in enumerate(data['annotations'][:5]):
    print(f"标注 {i+1}:")
    print(f"  文本: {ann['text']}")
    print(f"  实体: {ann['entity']}")
    print()

print("Entity_dict字段内容：")
print("=" * 50)
for key, value in data.get('entity_dict', {}).items():
    print(f"{key}: {value}")

print("\n分析entity字段的作用：")
print("=" * 50)

# 统计entity标签类型
entity_types = set()
for ann in data['annotations']:
    entity_types.update(ann['entity'])

print(f"发现的实体标签类型: {sorted(entity_types)}")

# 查找有实体标签的标注
entity_annotations = []
for ann in data['annotations']:
    if any(tag != 'O' for tag in ann['entity']):
        entity_annotations.append(ann)

print(f"\n包含实体标签的标注数量: {len(entity_annotations)}")
if entity_annotations:
    print("\n实体标注示例:")
    for i, ann in enumerate(entity_annotations[:3]):
        print(f"  {i+1}. 文本: '{ann['text']}' -> 实体: {ann['entity']}")