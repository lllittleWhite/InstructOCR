#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON to CSV Converter for InstructOCR Dataset
将COCO格式的JSON数据转换为InstructOCR所需的CSV格式
"""

import json
import csv
import os
from pathlib import Path

def decode_rec_to_text(rec_list):
    """
    将rec编码列表转换为文本
    这里使用简单的ASCII解码，实际项目中可能需要更复杂的字符映射
    """
    try:
        # 过滤掉填充值（通常是96或其他特殊值）
        valid_chars = [c for c in rec_list if c != 96 and c > 0]
        # 转换为字符
        text = ''.join([chr(c) for c in valid_chars if 32 <= c <= 126])
        return text if text else "TEXT"
    except:
        return "TEXT"

def convert_json_to_csv(json_file_path, output_csv_path):
    """
    将JSON格式的标注文件转换为CSV格式
    
    Args:
        json_file_path: 输入的JSON文件路径
        output_csv_path: 输出的CSV文件路径
    """
    
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建图像ID到文件名的映射
    image_id_to_filename = {}
    for image in data['images']:
        image_id_to_filename[image['id']] = image['file_name']
    
    # 按图像分组标注
    image_annotations = {}
    
    # 检查是否有annotations字段
    if 'annotations' not in data:
        print("警告: JSON文件中没有找到annotations字段")
        print("可用的字段:", list(data.keys()))
        
        # 如果没有annotations，创建空的CSV文件
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 只写入图像文件名，没有标注
            for image in data['images']:
                writer.writerow([image['file_name'], ""])
        
        print(f"已创建空标注的CSV文件: {output_csv_path}")
        return
    
    # 处理标注数据
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(annotation)
    
    # 写入CSV文件
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        for image_id, filename in image_id_to_filename.items():
            if image_id in image_annotations:
                # 构建标注字符串
                annotation_parts = []
                
                for ann in image_annotations[image_id]:
                    # 获取坐标信息 - 优先使用bezier_pts
                    if 'bezier_pts' in ann and ann['bezier_pts']:
                        # 使用贝塞尔曲线坐标，转换为四边形
                        bezier_pts = ann['bezier_pts']
                        if len(bezier_pts) >= 16:  # 贝塞尔曲线有16个点
                            # 取贝塞尔曲线的四个控制点作为四边形顶点
                            x1, y1 = bezier_pts[0], bezier_pts[1]    # 第一个点
                            x2, y2 = bezier_pts[6], bezier_pts[7]    # 第四个点
                            x3, y3 = bezier_pts[8], bezier_pts[9]    # 第五个点
                            x4, y4 = bezier_pts[14], bezier_pts[15]  # 第八个点
                            coords_str = f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4}"
                        else:
                            # 如果贝塞尔点不足，使用bbox
                            bbox = ann.get('bbox', [0, 0, 100, 100])
                            x, y, w, h = bbox
                            coords_str = f"{x},{y},{x+w},{y},{x+w},{y+h},{x},{y+h}"
                    elif 'segmentation' in ann and ann['segmentation']:
                        # 使用分割坐标
                        coords = ann['segmentation'][0]  # 取第一个多边形
                        if len(coords) >= 8:  # 确保至少有4个点（8个坐标）
                            coords_str = ','.join(map(str, coords[:8]))  # 只取前8个坐标
                        else:
                            # 如果坐标不足8个，从bbox生成
                            bbox = ann.get('bbox', [0, 0, 100, 100])
                            x, y, w, h = bbox
                            coords_str = f"{x},{y},{x+w},{y},{x+w},{y+h},{x},{y+h}"
                    elif 'bbox' in ann:
                        # 从边界框生成四边形坐标
                        bbox = ann['bbox']
                        x, y, w, h = bbox
                        coords_str = f"{x},{y},{x+w},{y},{x+w},{y+h},{x},{y+h}"
                    else:
                        # 默认坐标
                        coords_str = "0,0,100,0,100,100,0,100"
                    
                    # 获取文本内容 - 优先使用rec字段
                    if 'rec' in ann and ann['rec']:
                        text = decode_rec_to_text(ann['rec'])
                    elif 'text' in ann:
                        text = ann['text']
                    elif 'caption' in ann:
                        text = ann['caption']
                    else:
                        text = "TEXT"
                    
                    # 跳过无效文本
                    if text == "###" or not text.strip():
                        continue
                    
                    # 构建单个标注: "坐标&rec&文本"
                    annotation_parts.append(f"{coords_str}&rec&{text}")
                
                # 用&&tab&&连接多个标注
                annotation_string = "&&tab&&".join(annotation_parts)
                
                # 写入CSV行: [文件名, 标注字符串]
                writer.writerow([filename, annotation_string])
            else:
                # 没有标注的图像
                writer.writerow([filename, ""])
    
    print(f"转换完成: {json_file_path} -> {output_csv_path}")
    print(f"处理了 {len(image_id_to_filename)} 张图像")
    print(f"处理了 {len(data.get('annotations', []))} 个标注")

def main():
    """
    主函数 - 转换ICDAR2015数据集
    """
    
    # 设置文件路径
    base_dir = Path("/Users/owan/code_project/30_GraduationThesis/1_work1/InstructOCR/data/icdar2015")
    
    # 转换训练集
    train_json = base_dir / "ic15_train.json"
    train_csv = base_dir / "ic15_train.csv"
    
    if train_json.exists():
        print("正在转换训练集...")
        convert_json_to_csv(train_json, train_csv)
    else:
        print(f"训练集JSON文件不存在: {train_json}")
    
    # 转换测试集
    test_json = base_dir / "ic15_test.json"
    test_csv = base_dir / "ic15_test.csv"
    
    if test_json.exists():
        print("正在转换测试集...")
        convert_json_to_csv(test_json, test_csv)
    else:
        print(f"测试集JSON文件不存在: {test_json}")
    
    print("\n转换完成！")
    print("\nCSV格式说明:")
    print("每行格式: 图像文件名,标注字符串")
    print("标注字符串格式: x1,y1,x2,y2,x3,y3,x4,y4&rec&文本内容&&tab&&下一个标注...")
    print("\n示例:")
    print('img_1.jpg,"377,117,463,117,465,130,378,130&rec&PIZZA&&tab&&493,81,519,81,519,95,493,95&rec&Hut"')

if __name__ == "__main__":
    main()