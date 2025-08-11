import shutil
from pathlib import Path
import cv2
from cocojson.utils.common import read_coco_json, get_img2annots, path
from cocojson.utils.draw import draw_annot
import argparse

def filter_and_viz_by_score(
    json_path, img_root, out_dir, min_score=0.5, max_score=None, draw=True
):
    img_root = path(img_root, is_dir=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    coco_dict, _ = read_coco_json(json_path)
    img2annots = get_img2annots(coco_dict["annotations"])
    assert min_score or max_score, "min_score or max_score must be provided"
    cnt = 0
    for img_dict in coco_dict["images"]:
        img_id = img_dict["id"]
        annots = img2annots[img_id]
        if min_score is not None:
            valid_annots = [a for a in annots if a.get("score", 1.0) >= min_score]
        else:
            valid_annots = annots

        if max_score is not None:
            valid_annots = [a for a in valid_annots if a.get("score", 1.0) <= max_score]
        else:
            valid_annots = valid_annots

        if not valid_annots:
            continue
        
        cnt += 1

        img_path = img_root / img_dict["file_name"]
        if not img_path.is_file():
            print(f"图片不存在: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if draw:
            for annot in annots:
                draw_annot(img, annot)
        out_img_path = out_dir / img_dict["file_name"]
        out_img_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(out_img_path), img)
        print(f"已保存: {out_img_path}")
    print(f"共保存了{cnt}张图片")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据score过滤并可视化COCO图片")
    parser.add_argument('--json_path', type=str, required=True, help='COCO标注json文件路径')
    parser.add_argument('--img_root', type=str, required=True, help='图片根目录')
    parser.add_argument('--out_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--min_score', type=float, default=0.5, help='最小score阈值')
    parser.add_argument('--max_score', type=float, default=None, help='最大score阈值')
    parser.add_argument('--draw', action='store_true', help='是否画框')
    args = parser.parse_args()

    filter_and_viz_by_score(
        json_path=args.json_path,
        img_root=args.img_root,
        out_dir=args.out_dir,
        min_score=args.min_score,
        max_score=args.max_score,
        draw=args.draw,
    )