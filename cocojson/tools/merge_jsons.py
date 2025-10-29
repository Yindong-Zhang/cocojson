'''
Merge multiple coco jsons. Handles potential ID conflicts by reassigning image_id and annotation_id to ensure uniqueness.

For example, for merging json files provided by https://www.sama.com/sama-coco-dataset/.
'''

from pathlib import Path 
from tqdm import tqdm 

from cocojson.utils.common import read_coco_json, write_json

def merge_jsons_files(
    jsons,
    output_json,
):
    coco_dicts = []
    for coco_json in jsons:
        coco_dicts.append(read_coco_json(coco_json)[0])

    out_dict = merge_jsons(coco_dicts)

    write_json(output_json, out_dict)

def merge_jsons(coco_dicts):
    merged_dict = coco_dicts[0].copy()
    
    # 初始化ID计数器
    max_img_id = max([img_dict['id'] for img_dict in merged_dict['images']]) if merged_dict['images'] else 0
    max_ann_id = max([ann_dict['id'] for ann_dict in merged_dict['annotations']]) if merged_dict['annotations'] else 0
    
    # 创建ID映射字典，用于处理ID冲突
    img_id_mapping = {}
    ann_id_mapping = {}
    
    # 为第一个字典建立初始映射
    for img_dict in merged_dict['images']:
        img_id_mapping[img_dict['id']] = img_dict['id']
    for ann_dict in merged_dict['annotations']:
        ann_id_mapping[ann_dict['id']] = ann_dict['id']
    
    for coco_dict in tqdm(coco_dicts[1:]):
        # 处理images
        for img_dict in coco_dict['images']:
            original_img_id = img_dict['id']
            
            # 如果ID冲突，分配新的ID
            if original_img_id in img_id_mapping:
                max_img_id += 1
                new_img_id = max_img_id
                img_id_mapping[original_img_id] = new_img_id
            else:
                img_id_mapping[original_img_id] = original_img_id
                max_img_id = max(max_img_id, original_img_id)
            
            # 更新image字典的ID
            img_dict_copy = img_dict.copy()
            img_dict_copy['id'] = img_id_mapping[original_img_id]
            merged_dict['images'].append(img_dict_copy)
        
        # 处理annotations
        for ann_dict in coco_dict['annotations']:
            original_ann_id = ann_dict['id']
            original_img_id = ann_dict['image_id']
            
            # 如果annotation ID冲突，分配新的ID
            if original_ann_id in ann_id_mapping:
                max_ann_id += 1
                new_ann_id = max_ann_id
                ann_id_mapping[original_ann_id] = new_ann_id
            else:
                ann_id_mapping[original_ann_id] = original_ann_id
                max_ann_id = max(max_ann_id, original_ann_id)
            
            # 更新annotation字典的ID
            ann_dict_copy = ann_dict.copy()
            ann_dict_copy['id'] = ann_id_mapping[original_ann_id]
            ann_dict_copy['image_id'] = img_id_mapping[original_img_id]
            merged_dict['annotations'].append(ann_dict_copy)
    
    return merged_dict