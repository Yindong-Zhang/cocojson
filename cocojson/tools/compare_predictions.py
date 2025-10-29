#!/usr/bin/env python3
"""
比较标注JSON和推理JSON，筛选出错误预测的图片

功能：
1. 比较标注框和推理框的IOU
2. 筛选高分错误：推理得分高但实际错误（IOU低）
3. 筛选低分漏检：推理得分低但实际应该有框（IOU高）
4. 将包含这两类错误预测的图片拷贝到指定目录

使用方法：
python compare_predictions.py --gt_json gt.json --pred_json pred.json --images_dir /path/to/images --output_dir /path/to/output --iou_threshold 0.5 --score_threshold 0.5
"""

import json
import os
import shutil
import argparse
from typing import Dict, List, Tuple, Set
from pathlib import Path


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    计算两个边界框的IOU
    
    Args:
        box1: [x1, y1, x2, y2] 格式的边界框
        box2: [x1, y1, x2, y2] 格式的边界框
    
    Returns:
        IOU值 (0-1之间)
    """
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算并集区域
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def convert_bbox_format(bbox: List[float], format_type: str = "xywh") -> List[float]:
    """
    转换边界框格式
    
    Args:
        bbox: 边界框坐标
        format_type: 输入格式 ("xywh" 或 "xyxy")
    
    Returns:
        [x1, y1, x2, y2] 格式的边界框
    """
    if format_type == "xywh":
        # COCO格式: [x, y, width, height] -> [x1, y1, x2, y2]
        return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    elif format_type == "xyxy":
        # 已经是 [x1, y1, x2, y2] 格式
        return bbox
    else:
        raise ValueError(f"不支持的格式类型: {format_type}")


def load_json(json_path: str) -> Dict:
    """加载JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_image_filename(image_id: int, images: List[Dict]) -> str:
    """根据image_id获取图片文件名"""
    for img in images:
        if img['id'] == image_id:
            return img['file_name']
    return None


def find_matching_annotations(gt_annots: List[Dict], pred_annots: List[Dict], 
                            iou_threshold: float = 0.5) -> Tuple[List[Tuple[Dict, Dict, float]], List[Dict], List[Dict]]:
    """
    找到匹配的标注和预测框，允许一对多，也就是一个gt 匹配多个 pred

    Args:
        gt_annots: 标注框列表
        pred_annots: 预测框列表
        iou_threshold: IOU阈值
        
    Returns:
        matched_pairs: [(gt_annot, pred_annot, iou), ...]
        unmatched_gt: 未匹配的GT标注
        unmatched_pred: 未匹配的预测标注
    """
    matched_pairs = []
    matched_gt_ids = set()
    matched_pred_ids = set()
    
    # 为每个GT标注找到所有匹配的预测框
    for gt_annot in gt_annots:
        for i, pred_annot in enumerate(pred_annots):
            if i in matched_pred_ids:
                continue
                
            # 转换边界框格式
            gt_bbox = convert_bbox_format(gt_annot['bbox'])
            pred_bbox = convert_bbox_format(pred_annot['bbox'])
            
            # 计算IOU
            iou = calculate_iou(gt_bbox, pred_bbox)
            
            # 如果IOU超过阈值，认为是匹配的
            if iou >= iou_threshold:
                matched_pairs.append((gt_annot, pred_annot, iou))
                matched_gt_ids.add(gt_annots.index(gt_annot))
                matched_pred_ids.add(i)
    
    # 找出未匹配的标注
    unmatched_gt = [gt_annots[i] for i in range(len(gt_annots)) if i not in matched_gt_ids]
    unmatched_pred = [pred_annots[i] for i in range(len(pred_annots)) if i not in matched_pred_ids]
    
    return matched_pairs, unmatched_gt, unmatched_pred


def analyze_predictions(gt_json_path: str, pred_json_path: str, 
                       ) -> Dict:
    """
    分析预测结果，按照4个角度进行难例挖掘：
    1. 高分误检：高分但未匹配的预测框
    2. 低分漏检：未匹配的GT标注
    3. 定位精度差：匹配但IOU较低的预测框
    4. 置信度背离：匹配但置信度较低的预测框
    
    Returns:
        包含4类难例信息的字典
    """
    # 加载JSON文件
    gt_data = load_json(gt_json_path)
    pred_data = load_json(pred_json_path)
    
    # 按图片ID分组标注
    gt_by_image = {}
    pred_by_image = {}
    
    for annot in gt_data['annotations']:
        img_id = annot['image_id']
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        gt_by_image[img_id].append(annot)
    
    for annot in pred_data['annotations']:
        img_id = annot['image_id']
        if img_id not in pred_by_image:
            pred_by_image[img_id] = []
        pred_by_image[img_id].append(annot)
    
    # 分析每张图片
    high_score_false_positives = []  # 高分误检
    low_score_misses = []            # 低分漏检
    poor_localization = []           # 定位精度差
    low_confidence_matches = []      # 置信度背离
    
    iou_threshold = 0.3
    ideal_iou_threshold = 0.6
    minimum_score_threshold = 0.4
    false_positive_score_threshold = 0.4

    print(f"匹配IOU阈值: {iou_threshold}")
    print(f"理想匹配IOU阈值: {ideal_iou_threshold}")
    print(f"最低得分阈值: {minimum_score_threshold}")
    print(f"误检得分阈值: {false_positive_score_threshold}")

    print(f"开始分析...")
    print(f"GT图片数量: {len(gt_by_image)}")
    print(f"预测图片数量: {len(pred_by_image)}")

    for img_id in set(gt_by_image.keys()) | set(pred_by_image.keys()):
        gt_annots = gt_by_image.get(img_id, [])
        pred_annots = pred_by_image.get(img_id, [])
        
        if not gt_annots and not pred_annots:
            continue
        
        
        # 找到匹配的标注
        matched_pairs, unmatched_gt, unmatched_pred = find_matching_annotations(
            gt_annots, pred_annots, iou_threshold=iou_threshold
        )
        
        # 1. 高分误检：高分但未匹配的预测框
        for pred_annot in unmatched_pred:
            score = pred_annot.get('score', 0)
            if score >= false_positive_score_threshold:
                high_score_false_positives.append({
                    'image_id': img_id,
                    'pred_annot': pred_annot,
                    'score': score
                })
        
        # 2. 低分漏检：未匹配的GT标注（应该有框但没检测到）
        for gt_annot in unmatched_gt:
            low_score_misses.append({
                'image_id': img_id,
                'gt_annot': gt_annot
            })
        
        # 3. 定位精度差：匹配但IOU较低的预测框
        for gt_annot, pred_annot, iou in matched_pairs:
            score = pred_annot.get('score', 0)
            if iou < ideal_iou_threshold:  # 使用更严格的IOU阈值
                poor_localization.append({
                    'image_id': img_id,
                    'gt_annot': gt_annot,
                    'pred_annot': pred_annot,
                    'iou': iou,
                    'score': score
                })
        
        # 4. 置信度背离：匹配但置信度较低的预测框
        for gt_annot, pred_annot, iou in matched_pairs:
            score = pred_annot.get('score', 0)
            if score < minimum_score_threshold:  # 使用更低的置信度阈值
                low_confidence_matches.append({
                    'image_id': img_id,
                    'gt_annot': gt_annot,
                    'pred_annot': pred_annot,
                    'iou': iou,
                    'score': score
                })
    
    return {
        'high_score_false_positives': high_score_false_positives,
        'low_score_misses': low_score_misses,
        'poor_localization': poor_localization,
        'low_confidence_matches': low_confidence_matches,
        'gt_data': gt_data,
        'pred_data': pred_data,
    }


def copy_error_images(analysis_result: Dict, images_dir: str, output_dir: str) -> None:
    """
    拷贝包含难例的图片到输出目录，按类别分文件夹存储
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建各类别的子目录
    categories = ['high_score_false_positives', 'low_score_misses', 'poor_localization', 'low_confidence_matches']
    category_names = ['高分误检', '低分漏检', '定位精度差', '置信度背离']
    
    for category in categories:
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
    
    # 收集所有有难例的图片ID，按类别分组
    error_images_by_category = {category: set() for category in categories}
    
    for category in categories:
        for error in analysis_result[category]:
            error_images_by_category[category].add(error['image_id'])
    
    # 拷贝图片到对应类别目录（raw 与 vis 两份）
    total_copied = 0
    # 预构建 image_id -> annots 的索引
    gt_annots_all = analysis_result['gt_data'].get('annotations', [])
    pred_annots_all = analysis_result['pred_data'].get('annotations', [])
    gt_by_img = {}
    for ann in gt_annots_all:
        gt_by_img.setdefault(ann.get('image_id'), []).append(ann)
    pred_by_img = {}
    for ann in pred_annots_all:
        pred_by_img.setdefault(ann.get('image_id'), []).append(ann)

    for i, category in enumerate(categories):
        category_dir = os.path.join(output_dir, category)
        raw_dir = os.path.join(category_dir, 'raw')
        vis_dir = os.path.join(category_dir, 'vis')
        copied_count = 0
        
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

        print(f"\n=== {category_names[i]} ===")
        
        for img_id in error_images_by_category[category]:
            # 从GT或预测数据中获取图片信息
            img_info = None
            for img in analysis_result['gt_data']['images']:
                if img['id'] == img_id:
                    img_info = img
                    break
            
            if img_info:
                basename = os.path.basename(img_info['file_name'])
                src_path = os.path.join(images_dir, img_info['file_name'])
                # 拷贝原图到 raw
                raw_dst_path = os.path.join(raw_dir, basename)
                try:
                    shutil.copy2(src_path, raw_dst_path)
                except Exception:
                    print(f"  拷贝原图失败: {src_path}")
                    # 源文件不存在或其他错误则跳过
                    continue

                # 生成可视化到 vis
                try:
                    from PIL import Image, ImageDraw
                    img = Image.open(src_path).convert('RGB')
                    draw = ImageDraw.Draw(img)
                    # 画 GT（绿色）
                    for ann in gt_by_img.get(img_id, []):
                        x1, y1, x2, y2 = convert_bbox_format(ann.get('bbox', [0,0,0,0]))
                        draw.rectangle([x1, y1, x2, y2], outline=(0,255,0), width=2)
                    # 画 Pred（红色），并写 score
                    for ann in pred_by_img.get(img_id, []):
                        x1, y1, x2, y2 = convert_bbox_format(ann.get('bbox', [0,0,0,0]))
                        draw.rectangle([x1, y1, x2, y2], outline=(255,0,0), width=2)
                        score = ann.get('score', None)
                        if score is not None:
                            text = f"{score:.2f}"
                            draw.text((x1, max(0, y1-12)), text, fill=(255,0,0))
                    vis_dst_path = os.path.join(vis_dir, basename)
                    img.save(vis_dst_path)
                except Exception:
                    # 若绘制失败，仅保留原图
                    print(f"  生成可视化失败: {src_path}")
                    pass

                copied_count += 1

        print(f"  {category_names[i]} 共拷贝了 {copied_count} 张图片")
        total_copied += copied_count
    
    print(f"\n总共拷贝了 {total_copied} 张包含难例的图片到 {output_dir}")


def save_coco_for_category(analysis_result: Dict, category_key: str, base_output_dir: str) -> None:
    """
    针对某一难例类别，保存过滤后的 COCO 标注与预测 JSON 到输出目录下的 cocojson 主文件夹，
    并将 images.file_name 改为复制后的相对路径：{category_key}/basename。

    生成位置：{base_output_dir}/cocojson/
      - {category_key}.gt.json
      - {category_key}.pred.json
    """
    cocojson_dir = os.path.join(base_output_dir, 'cocojson')
    os.makedirs(cocojson_dir, exist_ok=True)

    # 收集该类别涉及到的图片ID
    image_ids = set()
    for item in analysis_result.get(category_key, []):
        image_ids.add(item['image_id'])

    if not image_ids:
        return

    gt_data = analysis_result['gt_data']
    pred_data = analysis_result['pred_data']

    # 过滤 images
    def filter_images(images):
        return [img for img in images if img.get('id') in image_ids]

    # 过滤 annotations 按 image_id
    def filter_annots(annots):
        return [ann for ann in annots if ann.get('image_id') in image_ids]

    # 调整 images.file_name 到复制后的路径：{category_key}/raw/basename
    def filter_and_remap_filename(images):
        remapped = []
        for img in filter_images(images):
            img_copy = dict(img)
            basename = os.path.basename(img_copy.get('file_name', ''))
            img_copy['file_name'] = f"{category_key}/raw/{basename}"
            remapped.append(img_copy)
        return remapped

    # --- GT COCO ---
    gt_out = {
        'images': filter_and_remap_filename(gt_data.get('images', [])),
        'annotations': filter_annots(gt_data.get('annotations', [])),
    }
    if 'categories' in gt_data:
        gt_out['categories'] = gt_data['categories']
    if 'info' in gt_data:
        gt_out['info'] = gt_data['info']
    if 'licenses' in gt_data:
        gt_out['licenses'] = gt_data['licenses']

    with open(os.path.join(cocojson_dir, f'{category_key}.gt.json'), 'w', encoding='utf-8') as f:
        json.dump(gt_out, f, ensure_ascii=False)

    # --- Pred COCO ---
    pred_out = {
        'images': filter_and_remap_filename(pred_data.get('images', [])),
        'annotations': filter_annots(pred_data.get('annotations', [])),
    }
    if 'categories' in pred_data:
        pred_out['categories'] = pred_data['categories']
    if 'info' in pred_data:
        pred_out['info'] = pred_data['info']
    if 'licenses' in pred_data:
        pred_out['licenses'] = pred_data['licenses']

    with open(os.path.join(cocojson_dir, f'{category_key}.pred.json'), 'w', encoding='utf-8') as f:
        json.dump(pred_out, f, ensure_ascii=False)


def print_analysis_summary(analysis_result: Dict) -> None:
    """打印分析结果摘要"""
    high_score_false_positives = analysis_result['high_score_false_positives']
    low_score_misses = analysis_result['low_score_misses']
    poor_localization = analysis_result['poor_localization']
    low_confidence_matches = analysis_result['low_confidence_matches']
    
    print("\n=== 难例挖掘分析结果摘要 ===")
    print(f"1. 高分误检数量: {len(high_score_false_positives)}")
    print(f"2. 低分漏检数量: {len(low_score_misses)}")
    print(f"3. 定位精度差数量: {len(poor_localization)}")
    print(f"4. 置信度背离数量: {len(low_confidence_matches)}")
    
    # 显示各类别的详细信息
    categories = [
        ('高分误检', high_score_false_positives),
        ('低分漏检', low_score_misses),
        ('定位精度差', poor_localization),
        ('置信度背离', low_confidence_matches)
    ]
    
    for category_name, category_data in categories:
        if category_data:
            print(f"\n{category_name}详情:")
            for i, item in enumerate(category_data[:3]):  # 只显示前3个
                if category_name == '高分误检':
                    print(f"  图片ID {item['image_id']}: 得分={item['score']:.3f}")
                elif category_name == '低分漏检':
                    print(f"  图片ID {item['image_id']}: 类别ID={item['gt_annot']['category_id']}")
                elif category_name == '定位精度差':
                    print(f"  图片ID {item['image_id']}: 得分={item['score']:.3f}, IOU={item['iou']:.3f}")
                elif category_name == '置信度背离':
                    print(f"  图片ID {item['image_id']}: 得分={item['score']:.3f}, IOU={item['iou']:.3f}")
            
            if len(category_data) > 3:
                print(f"  ... 还有 {len(category_data) - 3} 个{category_name}")


def main():
    parser = argparse.ArgumentParser(description='比较标注JSON和推理JSON，进行4角度难例挖掘')
    parser.add_argument('gt_json', help='标注JSON文件路径')
    parser.add_argument('pred_json', help='推理JSON文件路径')
    parser.add_argument('--images_dir', required=False, default=None, help='图片目录路径')
    parser.add_argument('--output_dir', required=False, default=None, help='输出目录路径')
    parser.add_argument('--max_copy_num', type=int, default=1000, help='每类最多拷贝多少张图片（若为None则不限制）')
    parser.add_argument('--no_copy', action='store_true', help='不拷贝图片，只进行分析')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.gt_json):
        print(f"错误: 标注JSON文件不存在: {args.gt_json}")
        return
    
    if not os.path.exists(args.pred_json):
        print(f"错误: 推理JSON文件不存在: {args.pred_json}")
        return

    
    print("开始进行难例挖掘分析...")
    print("分析角度:")
    print("  1. 高分误检：高分但未匹配的预测框")
    print("  2. 低分漏检：未匹配的GT标注")
    print("  3. 定位精度差：匹配但IOU较低的预测框")
    print("  4. 置信度背离：匹配但置信度较低的预测框")
    
    # 分析预测结果
    analysis_result = analyze_predictions(
        args.gt_json, 
        args.pred_json, 
    )
    
    # 打印分析结果
    print_analysis_summary(analysis_result)

    # 如果设置了最大图片数（max_copy_num），则对每类只保留前 max_copy_num 张
    if args.max_copy_num is not None:
        for key in ['high_score_false_positives', 'low_score_misses', 'poor_localization', 'low_confidence_matches']:
            items = analysis_result.get(key, [])
            if len(items) > args.max_copy_num:
                analysis_result[key] = items[:args.max_copy_num]
    
    # 拷贝难例图片与保存对应COCO JSON
    if not args.no_copy and args.images_dir and args.output_dir:
        if not os.path.exists(args.images_dir):
            print(f"错误: 图片目录不存在: {args.images_dir}")
            return
        print(f"\n开始拷贝包含难例的图片到 {args.output_dir}...")
        copy_error_images(analysis_result, args.images_dir, args.output_dir)

        # 保存各类别的 COCO JSON（写入到 {output_dir}/cocojson/ 下，并将 file_name 改为 {category}/basename）
        categories = ['high_score_false_positives', 'low_score_misses', 'poor_localization', 'low_confidence_matches']
        for category in categories:
            save_coco_for_category(analysis_result, category, args.output_dir)
    
    print("\n难例挖掘分析完成!")


if __name__ == "__main__":
    main()
