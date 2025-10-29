import argparse
import os
import sys
# 添加cocojson父目录到系统路径
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(parent_dir)
sys.path.append(parent_dir)

from cocojson.tools import exclude_images_from_files


def main():
    ap = argparse.ArgumentParser(description="Exclude images from COCO JSON A that are found in COCO JSON B")
    ap.add_argument("json_a", help="Path to COCO JSON A (source JSON)")
    ap.add_argument("json_b", help="Path to COCO JSON B (exclusion JSON)")
    ap.add_argument("--out", help="Output JSON path", type=str)
    args = ap.parse_args()

    exclude_images_from_files(args.json_a, args.json_b, out_json=args.out)


if __name__ == "__main__":
    main()
