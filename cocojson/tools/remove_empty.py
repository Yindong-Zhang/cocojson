"""
Remove empty/negative images from COCO JSON, aka images without associated annotations.

Original image IDs are preserved. 
"""

from cocojson.utils.common import read_coco_json, write_json_in_place


def remove_empty_from_files(coco_json, out_json=None,save_empty=False):
    coco_dict, _ = read_coco_json(coco_json)
    out_dict, empty_coco = remove_empty(coco_dict)
    write_json_in_place(coco_json, out_dict, append_str="noempty", out_json=out_json)
    if save_empty:
        write_json_in_place(coco_json, empty_coco, append_str="empty")
        print(f"save empty json to {coco_json}_empty.json")


def remove_empty(coco_dict):
    wanted_imgs = set([annot["image_id"] for annot in coco_dict["annotations"]])
    print(f"reserve {len(set(wanted_imgs))} / {len(coco_dict['images'])} images")

    empty_images = [img for img in coco_dict["images"] if img["id"] not in wanted_imgs]
    empty_coco = {
        "categories": coco_dict["categories"],
        "images": empty_images,
        "annotations": [],
    }
    new_imgs = [img for img in coco_dict["images"] if img["id"] in wanted_imgs]
    coco_dict["images"] = new_imgs
    return coco_dict, empty_coco
