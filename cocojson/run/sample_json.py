from cocojson.tools.sample_json_only import sample_json_only
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="Path to coco json")
    ap.add_argument("k", help="Number of images to sample", type=int)
    ap.add_argument("--output", help="Path to output json", default=None)
    args = ap.parse_args()

    sample_json_only(args.json, args.k, args.output)

if __name__ == "__main__":
    main()