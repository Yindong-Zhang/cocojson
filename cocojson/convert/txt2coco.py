#!/usr/bin/python

import sys
import os
import json
import glob
from pathlib import Path

START_BOUNDING_BOX_ID = 1
# PRE_DEFINE_CATEGORIES = None
# 如果需要，可以预定义类别及其ID
# PRE_DEFINE_CATEGORIES = {"person": 0, "car": 1, ...}
PRE_DEFINE_CATEGORIES = {
    'ignored regions': 0,
    'pedestrian': 1,
    'people': 2, 
    'bicycle': 3,
    'car': 4,
    'van': 5,
    'truck': 6,
    'tricycle': 7,
    'awning-tricycle': 8,
    'bus': 9,
    'motor': 10,
    'others': 11
}



def get_categories(txt_files):
    """从TXT文件列表中生成类别名称到ID的映射。
    
    参数:
        txt_files {list} -- TXT文件路径列表。
    
    返回:
        dict -- 类别名称到ID的映射。
    """
    classes_names = []
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:  # 确保有足够的部分
                    category = parts[5].strip()
                    classes_names.append(category)
    
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}


def convert(txt_files, json_file, img_ext=".jpg"):
    """将TXT标注文件转换为COCO JSON格式。
    
    参数:
        txt_files {list} -- TXT文件路径列表。
        json_file {str} -- 输出的COCO JSON文件路径。
        img_ext {str} -- 图像文件扩展名，默认为.jpg。
    """
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(txt_files)
    
    category_ids = set(categories.values())
    if not category_ids:
        raise ValueError("未找到任何类别")

    bnd_id = START_BOUNDING_BOX_ID
    
    for i, txt_file in enumerate(txt_files):
        # 获取对应的图像文件路径
        img_file = str(txt_file).replace('annotations', 'images').replace(".txt", img_ext)
        
        # 检查图像文件是否存在
        if not os.path.exists(img_file):
            print(f"警告: 文件 {img_file} 不存在")
            continue
        
        # 获取图像尺寸（这里需要实际读取图像文件获取尺寸）
        # 如果无法读取图像，可以从标注中估计或使用默认值
        try:
            from PIL import Image
            img = Image.open(img_file)
            width, height = img.size
        except:
            print(f"警告: 无法读取图像 {img_file} 的尺寸，使用默认值")
            width, height = 640, 480  # 默认尺寸
        
        image_id = i
        image = {
            "file_name": img_file,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)
        
        # 解析TXT文件中的标注
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    print(f"警告: 标注格式不正确 {line}")
                    continue
                
                try:
                    # 解析边界框坐标和类别
                    x = float(parts[0])
                    y = float(parts[1])
                    w = float(parts[2])
                    h = float(parts[3])
                    score = float(parts[4])  # 可选：置信度分数
                    category_id = int(parts[5].strip())
                    
                    # 可选：截断和遮挡信息
                    truncation = int(parts[6]) if len(parts) > 6 else 0
                    occlusion = int(parts[7]) if len(parts) > 7 else 0
                    
                    if category_id not in category_ids:
                        raise ValueError(f"未找到类别 {category_id} 的ID")
                    
                    # 创建标注对象
                    ann = {
                        "area": w * h,
                        "iscrowd": 0,
                        "image_id": image_id,
                        "bbox": [x, y, w, h],
                        "category_id": category_id,
                        "id": bnd_id,
                        "ignore": 0,
                        "segmentation": [],
                    }
                    
                    # 可选：添加额外信息
                    if score is not None:
                        ann["score"] = score
                    if truncation is not None:
                        ann["truncation"] = truncation
                    if occlusion is not None:
                        ann["occlusion"] = occlusion
                    
                    json_dict["annotations"].append(ann)
                    bnd_id += 1
                except Exception as e:
                    print(f"警告: 处理标注时出错 {line}: {e}")
    
    # 添加类别信息
    print(f"类别 {categories}")
    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)
    
    # 写入JSON文件
    with open(json_file, "w") as json_fp:
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
    
    return json_dict


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="将TXT标注文件转换为COCO格式。"
    )
    parser.add_argument("txt_dir", help="TXT文件目录路径。", type=str)
    parser.add_argument("json_file", help="输出的COCO格式JSON文件。", type=str)
    parser.add_argument("--img-ext", help="图像文件扩展名，默认为.jpg", default=".jpg", type=str)
    
    args = parser.parse_args()
    txt_files = glob.glob(os.path.join(args.txt_dir, "**/*.txt"), recursive=True)
    
    print(f"TXT文件数量: {len(txt_files)}")
    convert(txt_files, args.json_file, args.img_ext)
    print(f"转换成功: {args.json_file}")