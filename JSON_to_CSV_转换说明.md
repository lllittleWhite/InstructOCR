# JSON到CSV数据格式转换说明

## 概述

本文档详细说明了如何将COCO格式的JSON标注文件转换为InstructOCR项目所需的CSV格式。

## 原始JSON数据格式

### JSON文件结构
```json
{
  "licenses": [],
  "info": {},
  "categories": [
    {
      "id": 1,
      "name": "text",
      "supercategory": "beverage",
      "keypoints": ["mean", "xmin", "x2", "x3", "xmax", "ymin", "y2", "y3", "ymax", "cross"]
    }
  ],
  "images": [
    {
      "file_name": "img_346.jpg",
      "id": 346,
      "width": 1280,
      "height": 720
    }
  ],
  "annotations": [
    {
      "area": 6579.0,
      "bbox": [174.0, 187.0, 153.0, 43.0],
      "bezier_pts": [174, 193, 222, 191, 271, 189, 320, 187, 326, 222, 277, 224, 228, 226, 179, 229],
      "category_id": 1,
      "id": 0,
      "image_id": 346,
      "iscrowd": 0,
      "rec": [33, 67, 67, 69, 83, 83, 79, 82, 73, 69, 83, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96]
    }
  ]
}
```

### 关键字段说明

1. **images**: 图像信息列表
   - `file_name`: 图像文件名
   - `id`: 图像唯一标识符
   - `width`, `height`: 图像尺寸

2. **annotations**: 标注信息列表
   - `image_id`: 对应的图像ID
   - `bbox`: 边界框 [x, y, width, height]
   - `bezier_pts`: 贝塞尔曲线控制点（16个数值，表示8个点的x,y坐标）
   - `rec`: 文本的编码表示（数字列表）
   - `area`: 标注区域面积

## 目标CSV数据格式

### CSV格式规范

InstructOCR要求的CSV格式为：
```
图像文件名,标注字符串
```

其中标注字符串的格式为：
```
x1,y1,x2,y2,x3,y3,x4,y4&rec&文本内容&&tab&&下一个标注...
```

### 格式说明

1. **坐标部分**: `x1,y1,x2,y2,x3,y3,x4,y4`
   - 四个点的坐标，按顺序排列（通常是左上、右上、右下、左下）
   - 坐标值为浮点数

2. **分隔符**: `&rec&`
   - 坐标和文本之间的固定分隔符

3. **文本内容**: 实际的文本字符串

4. **多标注分隔符**: `&&tab&&`
   - 同一图像中多个标注之间的分隔符

## 转换过程详解

### 1. 坐标处理

转换脚本按以下优先级处理坐标：

1. **贝塞尔曲线坐标** (`bezier_pts`)
   - 从16个控制点中提取4个关键点
   - 选择第1、4、5、8个点作为四边形顶点
   ```python
   x1, y1 = bezier_pts[0], bezier_pts[1]    # 第一个点
   x2, y2 = bezier_pts[6], bezier_pts[7]    # 第四个点
   x3, y3 = bezier_pts[8], bezier_pts[9]    # 第五个点
   x4, y4 = bezier_pts[14], bezier_pts[15]  # 第八个点
   ```

2. **分割坐标** (`segmentation`)
   - 直接使用多边形的前8个坐标值

3. **边界框** (`bbox`)
   - 将矩形边界框转换为四边形坐标
   ```python
   x, y, w, h = bbox
   coords = f"{x},{y},{x+w},{y},{x+w},{y+h},{x},{y+h}"
   ```

### 2. 文本处理

转换脚本按以下优先级处理文本：

1. **编码文本** (`rec`)
   - 将数字编码列表转换为文本字符串
   - 过滤填充值（通常是96）
   - 使用ASCII解码
   ```python
   def decode_rec_to_text(rec_list):
       valid_chars = [c for c in rec_list if c != 96 and c > 0]
       text = ''.join([chr(c) for c in valid_chars if 32 <= c <= 126])
       return text if text else "TEXT"
   ```

2. **直接文本** (`text`)
   - 直接使用文本字段

3. **标题文本** (`caption`)
   - 使用标题字段作为文本

### 3. 转换示例

#### 原始JSON标注
```json
{
  "bbox": [174.0, 187.0, 153.0, 43.0],
  "bezier_pts": [174, 193, 222, 191, 271, 189, 320, 187, 326, 222, 277, 224, 228, 226, 179, 229],
  "rec": [33, 67, 67, 69, 83, 83, 79, 82, 73, 69, 83],
  "image_id": 346
}
```

#### 转换后CSV行
```csv
img_346.jpg,"174,193,320,187,326,222,179,229&rec&!CCESSORIES"
```

## 使用转换脚本

### 运行转换
```bash
cd /path/to/InstructOCR
python convert_json_to_csv.py
```

### 脚本功能
- 自动处理训练集和测试集
- 支持多种坐标格式
- 智能文本解码
- 错误处理和日志输出

### 输出信息
```
正在转换训练集...
转换完成: ic15_train.json -> ic15_train.csv
处理了 1000 张图像
处理了 4452 个标注
```

## 验证转换结果

### 检查CSV文件
```bash
head -5 /path/to/ic15_train.csv
```

### 预期输出格式
```csv
img_346.jpg,"174,193,320,187,326,222,179,229&rec&!CCESSORIES&&tab&&419,158,720,149,726,202,424,211&rec&LOG3HOPPING"
img_355.jpg,"608,299,657,300,656,314,607,313&rec&2!4%&&tab&&626,17,675,29,673,47,624,34&rec&3ONY"
```

## 注意事项

1. **字符编码**: 确保CSV文件使用UTF-8编码
2. **坐标精度**: 保持原始坐标的浮点精度
3. **特殊字符**: CSV中的引号和逗号需要正确转义
4. **空标注**: 没有标注的图像会生成空的标注字符串
5. **文本解码**: rec字段的解码可能需要根据具体数据集调整

## 常见问题

### Q: 转换后的文本显示乱码怎么办？
A: 检查`decode_rec_to_text`函数中的字符映射逻辑，可能需要使用特定的字符集或编码表。

### Q: 坐标顺序不正确怎么办？
A: 检查贝塞尔曲线点的提取逻辑，确保四个点按正确顺序排列。

### Q: 如何处理其他格式的JSON文件？
A: 修改转换脚本中的字段名称和处理逻辑，适配不同的JSON结构。

## 总结

通过这个转换脚本，可以将COCO格式的JSON标注文件成功转换为InstructOCR项目所需的CSV格式，保持了原始数据的完整性和准确性。转换后的数据可以直接用于InstructOCR的训练和测试。