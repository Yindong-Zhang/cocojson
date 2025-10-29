import argparse

from cocojson.tools import check_and_complete_coco_from_file


def main():
    ap = argparse.ArgumentParser(description="检查并补全 COCO JSON 缺失字段")
    ap.add_argument("json", help="Path to coco json")
    ap.add_argument("--out", help="Output json path", type=str)
    ap.add_argument(
        "--no_reassign_unique_ids",
        action="store_true",
        help="不对 images/annotations/categories 的 id 做唯一化重排",
    )
    args = ap.parse_args()

    check_and_complete_coco_from_file(
        args.json,
        out_json=args.out,
        reassign_unique_ids=not args.no_reassign_unique_ids,
    )


if __name__ == "__main__":
    main()


