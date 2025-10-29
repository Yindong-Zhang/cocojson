from typing import Dict, Any, List, Tuple

from cocojson.utils.common import read_coco_json, write_json_in_place


REQUIRED_TOP_FIELDS = ["info", "licenses", "images", "annotations", "categories"]


def _ensure_top_fields(d: Dict[str, Any]) -> List[str]:
    added = []
    for f in REQUIRED_TOP_FIELDS:
        if f not in d:
            added.append(f)
            if f == "info":
                d[f] = {
                    "description": "",
                    "url": "",
                    "version": "",
                    "year": 0,
                    "contributor": "",
                    "date_created": "",
                }
            elif f == "licenses":
                d[f] = []
            elif f == "images":
                d[f] = []
            elif f == "annotations":
                d[f] = []
            elif f == "categories":
                d[f] = []
    return added


def _complete_images(images: List[Dict[str, Any]]) -> Tuple[int, int]:
    fixed = 0
    filled = 0
    for img in images:
        before = img.copy()
        img.setdefault("file_name", "unknown.jpg")
        img.setdefault("height", 0)
        img.setdefault("width", 0)
        img.setdefault("id", 0)
        if before != img:
            if any(k not in before for k in ("file_name", "height", "width", "id")):
                filled += 1
            else:
                fixed += 1
    return fixed, filled


def _rect_polygon_from_bbox(bbox: List[float]) -> List[List[float]]:
    x, y, w, h = bbox
    return [[
        float(x), float(y),
        float(x + w), float(y),
        float(x + w), float(y + h),
        float(x), float(y + h),
    ]]


def _rle_from_bbox(bbox: List[float], img_h: int, img_w: int):
    try:
        import numpy as np
        from pycocotools import mask as maskUtils
    except Exception:
        return None

    x, y, w, h = bbox
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(img_w, int(round(x + w)))
    y1 = min(img_h, int(round(y + h)))
    if x0 >= x1 or y0 >= y1 or img_h <= 0 or img_w <= 0:
        return None
    m = np.zeros((img_h, img_w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    rle = maskUtils.encode(np.asfortranarray(m))
    # pycocotools 返回 bytes 计数，需要转为 utf-8 字符串
    if isinstance(rle.get("counts"), bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _complete_annotations(annots: List[Dict[str, Any]], img_hw_by_id: Dict[int, Tuple[int, int]]) -> Tuple[int, int]:
    fixed = 0
    filled = 0
    for a in annots:
        before = a.copy()
        a.setdefault("id", 0)
        a.setdefault("image_id", 0)
        a.setdefault("category_id", 0)
        a.setdefault("bbox", [0, 0, 0, 0])
        a.setdefault("area", 0)
        a.setdefault("iscrowd", 0)
        # segmentation: polygon 或 RLE
        seg = a.get("segmentation", None)
        bbox = a.get("bbox")

        # 若有 bbox 且 area 缺省或为 0，可按 w*h 估算
        if bbox and isinstance(bbox, list) and len(bbox) == 4:
            w = bbox[2] if isinstance(bbox[2], (int, float)) else 0
            h = bbox[3] if isinstance(bbox[3], (int, float)) else 0
            if (not a.get("area")) and w >= 0 and h >= 0:
                a["area"] = float(w) * float(h)

        # 补全 segmentation
        if seg is None:
            a["segmentation"] = []

        if before != a:
            if any(
                k not in before
                for k in ("id", "image_id", "category_id", "bbox", "area", "iscrowd", "segmentation")
            ):
                filled += 1
            else:
                fixed += 1
    return fixed, filled


def _complete_categories(cats: List[Dict[str, Any]]) -> Tuple[int, int]:
    fixed = 0
    filled = 0
    for c in cats:
        before = c.copy()
        c.setdefault("id", 0)
        c.setdefault("name", "unknown")
        c.setdefault("supercategory", "unknown")
        if before != c:
            if any(k not in before for k in ("id", "name", "supercategory")):
                filled += 1
            else:
                fixed += 1
    return fixed, filled


def _assign_unique_ids_if_needed(images: List[Dict[str, Any]], annots: List[Dict[str, Any]], cats: List[Dict[str, Any]]):
    # 若存在 0 或重复 id，可顺序重排为唯一 id（仅在必要时）
    def ensure_unique(items: List[Dict[str, Any]], key: str):
        seen = set()
        need_reassign = False
        for it in items:
            cur = it.get(key, 0)
            if cur in seen:
                need_reassign = True
                break
            seen.add(cur)
        if 0 in seen:
            need_reassign = True

        if need_reassign:
            for idx, it in enumerate(items, start=1):
                it[key] = idx

    ensure_unique(images, "id")
    ensure_unique(cats, "id")
    ensure_unique(annots, "id")


def check_and_complete_coco_from_file(coco_json: str, out_json: str = None, reassign_unique_ids: bool = True):
    d, _ = read_coco_json(coco_json)

    added_top = _ensure_top_fields(d)
    if added_top:
        print(f"缺少字段: {', '.join(added_top)}，已添加默认值。")

    img_fix, img_fill = _complete_images(d["images"]) if d.get("images") is not None else (0, 0)
    # 准备 image_id -> (h, w)
    img_hw_by_id: Dict[int, Tuple[int, int]] = {}
    for img in d.get("images", []):
        try:
            img_hw_by_id[int(img.get("id"))] = (int(img.get("height", 0)), int(img.get("width", 0)))
        except Exception:
            pass
    ann_fix, ann_fill = _complete_annotations(d["annotations"], img_hw_by_id) if d.get("annotations") is not None else (0, 0)
    cat_fix, cat_fill = _complete_categories(d["categories"]) if d.get("categories") is not None else (0, 0)

    if reassign_unique_ids:
        _assign_unique_ids_if_needed(d["images"], d["annotations"], d["categories"])

    print(
        "完成检查："
        f"images 修复{img_fix}项，补全{img_fill}项；"
        f"annotations 修复{ann_fix}项，补全{ann_fill}项；"
        f"categories 修复{cat_fix}项，补全{cat_fill}项。"
    )

    write_json_in_place(coco_json, d, append_str="completed", out_json=out_json)
    print("检查并补全完成。")


