"""
Remove missing images from COCO JSON, aka images that do not exist in the image directory.
Original image IDs are preserved. 
"""

from cocojson.utils.common import read_coco_json, write_json_in_place
import os

def remove_missing_from_files(coco_json, image_dir, out_json=None):
    coco_dict, _ = read_coco_json(coco_json)
    out_dict = remove_missing(coco_dict, image_dir)
    write_json_in_place(coco_json, out_dict, append_str="check_existence", out_json=out_json)


def remove_missing(coco_dict, image_dir):
    # 获取所有图片和标注
    images = coco_dict["images"]
    annotations = coco_dict["annotations"]
    
    # 检查每个图片是否存在
    valid_image_ids = set()
    for img in images:
        img_path = img["file_name"]
        if os.path.exists(os.path.join(image_dir, img_path)):
            try:
                from PIL import Image
                f_img = Image.open(os.path.join(image_dir, img_path))
                f_img.verify()  # 验证图片是否完整
                f_img.close()
            except Exception as e:
                print(f"无法读取图片 {img_path}: {str(e)}")
                continue
            valid_image_ids.add(img["id"])
    print(f"reserve {len(valid_image_ids)} / {len(images)} images")
    
    # 过滤出有效的图片
    filtered_images = [img for img in images if img["id"] in valid_image_ids]
    
    # 过滤出有效图片对应的标注
    filtered_annotations = [ann for ann in annotations if ann["image_id"] in valid_image_ids]
    
    # 更新coco字典
    coco_dict["images"] = filtered_images
    coco_dict["annotations"] = filtered_annotations
    
    return coco_dict
