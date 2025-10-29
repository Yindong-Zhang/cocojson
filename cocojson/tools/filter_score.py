import shutil
from pathlib import Path
import cv2
from cocojson.utils.common import read_coco_json, get_img2annots, path, write_json
from cocojson.utils.draw import draw_annot
import argparse
import os

def filter_and_viz_by_score(
    json_path, img_root, out_dir, min_score=0.5, max_score=None, max_imgs=5000, draw=True
):
    img_root = path(img_root, is_dir=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    coco_dict, _ = read_coco_json(json_path)
    img2annots = get_img2annots(coco_dict["annotations"])
    assert min_score or max_score, "min_score or max_score must be provided"
    # 第一阶段：先遍历统计并收集候选
    candidates = []
    num_images = 0
    num_boxes = 0
    for img_dict in coco_dict["images"]:
        img_id = img_dict["id"]
        annots = img2annots[img_id]
        if min_score is not None:
            valid_annots = [a for a in annots if a.get("score", 1.0) >= min_score]
        else:
            valid_annots = annots

        if max_score is not None:
            valid_annots = [a for a in valid_annots if a.get("score", 1.0) <= max_score]

        if not valid_annots:
            continue

        img_path = img_root / img_dict["file_name"]
        if not img_path.is_file():
            print(f"图片不存在: {img_path}")
            continue

        candidates.append((img_dict, annots, valid_annots))
        num_images += 1
        num_boxes += len(valid_annots)

    print(f"候选图片数: {num_images}，候选框数: {num_boxes}")
    
    # 避免太多的拷贝
    if len(candidates) > max_imgs:
        import random
        candidates = random.sample(candidates, max_imgs)

    # 第二阶段：拷贝/可视化输出
    out_dir.mkdir(exist_ok=True, parents=True)
    saved = 0
    for img_dict, __, valid_annots in candidates:

        img_path = img_root / img_dict["file_name"]
        img_name = os.path.basename(img_dict["file_name"])
        out_img_path = out_dir / img_name

        if draw:
            img = cv2.imread(str(img_path))
            for annot in valid_annots:
                draw_annot(img, annot)
            cv2.imwrite(str(out_img_path), img)
        else:
            shutil.copy(str(img_path), str(out_img_path))
        saved += 1

        img_dict['file_name'] = out_dir.stem + "/" + img_name
        print(f"拷贝: {img_path}")
        

    print(f"共保存了{saved}张图片到{out_dir}")
    
    # 输出对应 images/annotations/categories 的 COCO JSON
    filtered_images = [img_dict for (img_dict, _, _) in candidates]
    filtered_annotations = []
    used_cat_ids = set()
    for _, __, valid_annots in candidates:
        filtered_annotations.extend(valid_annots)
        for a in valid_annots:
            cid = a.get("category_id")
            if cid is not None:
                used_cat_ids.add(cid)

    for i, annot in enumerate(filtered_annotations):
        annot['id'] = i + 1
        
    out_coco = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": coco_dict["categories"],
    }
    if "info" in coco_dict:
        out_coco["info"] = coco_dict["info"]
    if "licenses" in coco_dict:
        out_coco["licenses"] = coco_dict["licenses"]

    out_json_path = out_dir.parent / (out_dir.stem + "_filtered_by_score.json")
    write_json(str(out_json_path), out_coco)
    print(f"已写出标注: {out_json_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据score过滤并可视化COCO图片")
    parser.add_argument('--json_path', type=str, required=True, help='COCO标注json文件路径')
    parser.add_argument('--img_root', type=str, required=True, help='图片根目录')
    parser.add_argument('--out_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--min_score', type=float, default=0.5, help='最小score阈值')
    parser.add_argument('--max_score', type=float, default=None, help='最大score阈值')
    parser.add_argument('--max_imgs', type=int, default=5000, help='最多拷贝图片数')
    parser.add_argument('--draw', action='store_true', help='是否画框')
    args = parser.parse_args()

    filter_and_viz_by_score(
        json_path=args.json_path,
        img_root=args.img_root,
        out_dir=args.out_dir,
        min_score=args.min_score,
        max_score=args.max_score,
        max_imgs=args.max_imgs,
        draw=args.draw,
    )