"""
Extract a subset of COCO JSON based on a list of filename prefixes.
This tool allows you to extract images and their corresponding annotations
based on matching image filename prefixes, and optionally save the remaining data.
"""

from copy import deepcopy
from collections import defaultdict

from cocojson.utils.common import read_coco_json, write_json_in_place


def extract_by_prefix(cocojson, prefixes, output_name=None, save_remaining=True):
    """
    Extract images and annotations from COCO JSON based on filename prefixes.
    
    Args:
        cocojson: Path to the input COCO JSON file
        prefixes: List of filename prefixes to match
        output_name: Optional suffix for the output filename
        save_remaining: Whether to save the remaining data after extraction
    """
    coco_dict, setname = read_coco_json(cocojson)
    
    # 创建新的COCO字典
    extracted_dict = {
        "images": [],
        "annotations": [],
    }
    
    remaining_dict = {
        "images": [],
        "annotations": [],
    }
    
    # 复制基本信息到两个字典
    for dict_key in ["info", "licenses", "categories"]:
        if dict_key in coco_dict:
            extracted_dict[dict_key] = deepcopy(coco_dict[dict_key])
            remaining_dict[dict_key] = deepcopy(coco_dict[dict_key])
    
    # 建立新旧图片ID的映射
    extracted_img_ids = {}
    remaining_img_ids = {}
    new_extracted_id = 1
    new_remaining_id = 1
    
    # 根据前缀筛选图片
    for img in coco_dict["images"]:
        is_extracted = False
        for prefix in prefixes:
            if img["file_name"].startswith(prefix):
                new_img = deepcopy(img)
                extracted_img_ids[img["id"]] = new_extracted_id
                new_img["id"] = new_extracted_id
                extracted_dict["images"].append(new_img)
                new_extracted_id += 1
                is_extracted = True
                break
        
        if not is_extracted:
            new_img = deepcopy(img)
            remaining_img_ids[img["id"]] = new_remaining_id
            new_img["id"] = new_remaining_id
            remaining_dict["images"].append(new_img)
            new_remaining_id += 1
    
    # 处理标注
    new_extracted_ann_id = 1
    new_remaining_ann_id = 1
    
    for ann in coco_dict["annotations"]:
        if ann["image_id"] in extracted_img_ids:
            new_ann = deepcopy(ann)
            new_ann["id"] = new_extracted_ann_id
            new_ann["image_id"] = extracted_img_ids[ann["image_id"]]
            extracted_dict["annotations"].append(new_ann)
            new_extracted_ann_id += 1
        elif save_remaining and ann["image_id"] in remaining_img_ids:
            new_ann = deepcopy(ann)
            new_ann["id"] = new_remaining_ann_id
            new_ann["image_id"] = remaining_img_ids[ann["image_id"]]
            remaining_dict["annotations"].append(new_ann)
            new_remaining_ann_id += 1
    
    # 更新数据集描述
    if "info" in extracted_dict:
        extracted_dict["info"]["description"] = f"{setname}_extracted_by_prefix"
    if "info" in remaining_dict:
        remaining_dict["info"]["description"] = f"{setname}_remaining"
    
    # 写入新的JSON文件
    suffix = output_name if output_name else "extracted"
    write_json_in_place(cocojson, extracted_dict, append_str=suffix)
    
    if save_remaining:
        write_json_in_place(cocojson, remaining_dict, append_str="remaining")