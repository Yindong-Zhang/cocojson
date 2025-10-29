"""
Exclude images from COCO JSON A that are found in COCO JSON B.
Any images in JSON A that are found in JSON B will be removed (along with associated annotations).
"""

from cocojson.utils.common import read_coco_json, write_json_in_place


def exclude_images_from_files(json_a, json_b, out_json=None):
    """
    Exclude images from COCO JSON A that are found in COCO JSON B.
    
    Args:
        json_a: Path to COCO JSON A (source JSON)
        json_b: Path to COCO JSON B (exclusion JSON)
        out_json: Optional output JSON path
    """
    coco_dict_a, _ = read_coco_json(json_a)
    coco_dict_b, _ = read_coco_json(json_b)
    
    out_dict = exclude_images(coco_dict_a, coco_dict_b)
    write_json_in_place(json_a, out_dict, append_str="excluded", out_json=out_json)


def exclude_images(coco_dict_a, coco_dict_b):
    """
    Exclude images from COCO dict A that are found in COCO dict B.
    
    Args:
        coco_dict_a: COCO dictionary A (source)
        coco_dict_b: COCO dictionary B (exclusion)
        
    Returns:
        Updated COCO dictionary A with excluded images removed
    """
    # Get image file names from JSON B for comparison
    images_b = coco_dict_b["images"]
    excluded_file_names = set(img["file_name"] for img in images_b)
    
    print(f"Found {len(excluded_file_names)} images in JSON B to exclude")
    
    # Filter images from JSON A that are NOT in JSON B
    images_a = coco_dict_a["images"]
    valid_image_ids = set()
    
    for img in images_a:
        if img["file_name"] not in excluded_file_names:
            valid_image_ids.add(img["id"])
    
    print(f"Keeping {len(valid_image_ids)} / {len(images_a)} images from JSON A")
    
    # Filter images
    filtered_images = [img for img in images_a if img["id"] in valid_image_ids]
    
    # Filter annotations for valid images
    annotations_a = coco_dict_a["annotations"]
    filtered_annotations = [ann for ann in annotations_a if ann["image_id"] in valid_image_ids]
    
    print(f"Keeping {len(filtered_annotations)} / {len(annotations_a)} annotations")
    
    # Update COCO dictionary
    coco_dict_a["images"] = filtered_images
    coco_dict_a["annotations"] = filtered_annotations
    
    return coco_dict_a
