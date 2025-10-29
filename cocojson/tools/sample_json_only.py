"""
对JSON文件进行采样，不改变图片位置。

只修改JSON文件内容，不复制或移动图片文件。
适用于只需要减少数据集大小但保持图片原位置的场景。
"""

from collections import defaultdict
from random import sample as _sample
from tqdm import tqdm
from pathlib import Path

from cocojson.utils.common import read_json, write_json


def sample_json_only(json_path, k=10, output_path=None, random_seed=None):
    """
    对JSON文件进行采样，不改变图片位置
    
    Args:
        json_path: 输入JSON文件路径
        k: 采样的图片数量
        output_path: 输出JSON文件路径，如果为None则自动生成
        random_seed: 随机种子，用于可重现的采样
    
    Returns:
        输出JSON文件路径
    """
    json_path = Path(json_path)
    
    if random_seed is not None:
        import random
        random.seed(random_seed)
    
    # 读取COCO格式的JSON文件
    coco_dict = read_json(json_path)
    
    assert k > 0, "采样数量必须大于0"
    
    # 获取所有图片
    all_images = coco_dict["images"]
    
    if k > len(all_images):
        print(f"警告: 请求采样 {k} 张图片，但数据集只有 {len(all_images)} 张图片，将使用全部图片")
        sampled_images = all_images
    else:
        sampled_images = _sample(all_images, k)
    
    # 获取采样图片的ID
    sampled_img_ids = {img["id"] for img in sampled_images}
    
    # 筛选对应的标注
    sampled_annotations = []
    for annot in coco_dict["annotations"]:
        if annot["image_id"] in sampled_img_ids:
            sampled_annotations.append(annot)
    
    # 更新COCO字典
    new_coco_dict = {
        "images": sampled_images,
        "annotations": sampled_annotations,
        "categories": coco_dict["categories"],  # 保持类别信息不变
        "info": coco_dict.get("info", {}),  # 保持信息字段
        "licenses": coco_dict.get("licenses", [])  # 保持许可证信息
    }
    
    # 确定输出路径
    if output_path is None:
        output_path = json_path.parent / f"sampled_{k}_{json_path.name}"
    else:
        output_path = Path(output_path)
    
    # 写入新的JSON文件
    write_json(output_path, new_coco_dict)
    
    print(f"采样完成:")
    print(f"  原始图片数量: {len(all_images)}")
    print(f"  采样图片数量: {len(sampled_images)}")
    print(f"  原始标注数量: {len(coco_dict['annotations'])}")
    print(f"  采样标注数量: {len(sampled_annotations)}")
    print(f"  输出文件: {output_path}")
    
    return output_path


def sample_by_class_json_only(json_path, class_ks=10, output_path=None, max_img=None, random_seed=None):
    """
    按类别对JSON文件进行采样，不改变图片位置
    
    Args:
        json_path: 输入JSON文件路径
        class_ks: 每个类别采样的数量，可以是整数（所有类别相同）或列表
        output_path: 输出JSON文件路径
        max_img: 最大图片数量限制
        random_seed: 随机种子
    
    Returns:
        输出JSON文件路径
    """
    json_path = Path(json_path)
    
    if random_seed is not None:
        import random
        random.seed(random_seed)
    
    # 读取COCO格式的JSON文件
    coco_dict = read_json(json_path)
    
    num_cats = len(coco_dict["categories"])
    
    # 处理class_ks参数
    if isinstance(class_ks, int):
        class_k = class_ks
        class_ks = [class_k for _ in range(num_cats + 1)]  # +1 for empty images
    elif len(class_ks) == 1:
        class_k = class_ks[0]
        class_ks = [class_k for _ in range(num_cats + 1)]
    
    assert len(class_ks) == num_cats + 1, f"class_ks长度应为 {num_cats + 1}，但得到 {len(class_ks)}"
    
    # 按类别组织图片ID
    class_to_images = defaultdict(list)
    seen_img_ids = set()
    
    for annot in tqdm(coco_dict["annotations"], desc="组织图片按类别"):
        img_id = annot["image_id"]
        cat_id = annot["category_id"]
        if img_id not in class_to_images[cat_id]:
            class_to_images[cat_id].append(img_id)
        seen_img_ids.add(img_id)
    
    # 处理没有标注的图片
    all_image_ids = {img["id"] for img in coco_dict["images"]}
    empty_image_ids = all_image_ids - seen_img_ids
    class_to_images["empty"] = list(empty_image_ids)
    
    # 采样
    sampled_imgs = []
    for imgs, k in zip(class_to_images.values(), class_ks):
        if k > len(imgs):
            sampled = imgs
        else:
            sampled = _sample(imgs, k)
        sampled_imgs.extend(sampled)
    
    sampled_imgs = list(set(sampled_imgs))  # 去重
    
    # 检查是否超过最大图片数量限制
    if max_img and len(sampled_imgs) > max_img:
        print(f"警告: 采样结果 {len(sampled_imgs)} 张图片超过限制 {max_img} 张")
        # 随机选择max_img张图片
        sampled_imgs = _sample(sampled_imgs, max_img)
    
    # 筛选图片和标注
    sampled_images = [img for img in coco_dict["images"] if img["id"] in sampled_imgs]
    sampled_annotations = [annot for annot in coco_dict["annotations"] if annot["image_id"] in sampled_imgs]
    
    # 更新COCO字典
    new_coco_dict = {
        "images": sampled_images,
        "annotations": sampled_annotations,
        "categories": coco_dict["categories"],
        "info": coco_dict.get("info", {}),
        "licenses": coco_dict.get("licenses", [])
    }
    
    # 确定输出路径
    if output_path is None:
        output_path = json_path.parent / f"sampled_by_class_{json_path.name}"
    else:
        output_path = Path(output_path)
    
    # 写入新的JSON文件
    write_json(output_path, new_coco_dict)
    
    print(f"按类别采样完成:")
    print(f"  原始图片数量: {len(coco_dict['images'])}")
    print(f"  采样图片数量: {len(sampled_images)}")
    print(f"  原始标注数量: {len(coco_dict['annotations'])}")
    print(f"  采样标注数量: {len(sampled_annotations)}")
    print(f"  输出文件: {output_path}")
    
    return output_path
