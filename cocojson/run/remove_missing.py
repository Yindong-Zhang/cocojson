import argparse
import os
import sys
# 添加cocojson父目录到系统路径
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(parent_dir)
sys.path.append(parent_dir)

from cocojson.tools import remove_missing_from_files




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="Path to coco json")
    ap.add_argument("coco_path", help="Path to coco directory")
    ap.add_argument("--out", help="Output json path", type=str)
    args = ap.parse_args()

    remove_missing_from_files(args.json, args.coco_path, out_json=args.out)


if __name__ == "__main__":
    main()
