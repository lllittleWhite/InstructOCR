#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 nfv5_3125 数据集从 JSON 格式转换为 InstructOCR 项目所需的 CSV 格式

输入格式：每行一个JSON对象，包含：
- file_name: 图像文件路径
- annotations: 标注列表，每个标注包含polygon、text、entity

输出格式：CSV文件，格式为：
图像文件名,标注字符串
其中标注字符串格式为：x1,y1,x2,y2,x3,y3,x4,y4&rec&文本内容&&tab&&下一个标注...
"""

import json
import csv
import os
from typing import List, Tuple

def polygon_to_quad(polygon: List[float]) -> List[float]:
    """
    将多边形坐标转换为四边形坐标
    如果polygon有超过8个坐标点，取前4个点
    如果少于8个坐标点，尝试补全或使用现有点
    """
    if len(polygon) >= 8:
        # 取前4个点 (x1,y1,x2,y2,x3,y3,x4,y4)
        return polygon[:8]
    elif len(polygon) == 6:
        # 3个点，复制最后一个点
        return polygon + polygon[-2:]
    elif len(polygon) == 4:
        # 2个点，假设是矩形的对角点
        x1, y1, x2, y2 = polygon
        return [x1, y1, x2, y1, x2, y2, x1, y2]
    elif len(polygon) == 2:
        # 1个点，创建一个小矩形
        x, y = polygon
        return [x, y, x+10, y, x+10, y+10, x, y+10]
    else:
        # 异常情况，返回默认值
        return [0, 0, 10, 0, 10, 10, 0, 10]

def convert_json_to_csv(json_file: str, csv_file: str) -> None:
    """
    将JSON格式的标注文件转换为CSV格式
    """
    print(f"正在转换 {json_file} -> {csv_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f_in, \
         open(csv_file, 'w', newline='', encoding='utf-8') as f_out:
        
        csv_writer = csv.writer(f_out, delimiter='\t')
        
        processed_images = 0
        total_annotations = 0
        
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())
                
                file_name = data['file_name']
                annotations = data.get('annotations', [])
                
                if not annotations:
                    print(f"警告：第{line_num}行图像 {file_name} 没有标注")
                    continue
                
                # 构建标注字符串
                annotation_parts = []
                
                for ann in annotations:
                    polygon = ann.get('polygon', [])
                    text = ann.get('text', '').strip()
                    
                    if not polygon or not text:
                        continue
                    
                    # 转换坐标为四边形
                    quad_coords = polygon_to_quad(polygon)
                    
                    # 格式化坐标（保留整数）
                    coords_str = ','.join([str(int(round(coord))) for coord in quad_coords])
                    
                    # 构建单个标注字符串：坐标&rec&文本
                    annotation_str = f"{coords_str}&rec&{text}"
                    annotation_parts.append(annotation_str)
                
                if annotation_parts:
                    # 用 &&tab&& 连接多个标注
                    full_annotation = '&&tab&&'.join(annotation_parts)
                    
                    # 写入CSV
                    csv_writer.writerow([file_name, full_annotation])
                    
                    processed_images += 1
                    total_annotations += len(annotation_parts)
                
            except json.JSONDecodeError as e:
                print(f"错误：第{line_num}行JSON解析失败: {e}")
            except Exception as e:
                print(f"错误：处理第{line_num}行时出错: {e}")
    
    print(f"转换完成：")
    print(f"  - 处理图像数量: {processed_images}")
    print(f"  - 总标注数量: {total_annotations}")
    print(f"  - 输出文件: {csv_file}")

def main():
    # 数据集路径
    data_dir = "/Users/owan/code_project/30_GraduationThesis/1_work1/InstructOCR/data/nfv5_3125"
    
    # 转换训练集
    train_json = os.path.join(data_dir, "train.txt")
    train_csv = os.path.join(data_dir, "nfv5_train.csv")
    convert_json_to_csv(train_json, train_csv)
    
    print()
    
    # 转换测试集
    test_json = os.path.join(data_dir, "test.txt")
    test_csv = os.path.join(data_dir, "nfv5_test.csv")
    convert_json_to_csv(test_json, test_csv)
    
    print("\n=== 转换完成 ===")
    print("CSV格式说明：")
    print("- 每行格式：图像文件名<TAB>标注字符串")
    print("- 标注字符串格式：x1,y1,x2,y2,x3,y3,x4,y4&rec&文本内容")
    print("- 多个标注用 &&tab&& 分隔")
    print("\n生成的文件：")
    print(f"- {train_csv}")
    print(f"- {test_csv}")

if __name__ == "__main__":
    main()