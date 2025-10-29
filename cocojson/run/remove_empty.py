import argparse

from cocojson.tools import remove_empty_from_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="Path to coco json")
    ap.add_argument("--out", help="Output json path", type=str)
    ap.add_argument(
        "--save_empty",
        action="store_true",
        help="Save a separate JSON with only empty images",
    )
    args = ap.parse_args()

    remove_empty_from_files(args.json, out_json=args.out, save_empty=args.save_empty)


if __name__ == "__main__":
    main()
