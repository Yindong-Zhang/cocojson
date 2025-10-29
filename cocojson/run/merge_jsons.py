import argparse

from cocojson.tools import merge_jsons_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsons", nargs="+", help="Path to json files to merge")
    parser.add_argument("--output_json", help="Path to output json file", default="merged.json")
    args = parser.parse_args()
    merge_jsons_files(args.jsons, args.output_json)

if __name__ == "__main__":
    main()
