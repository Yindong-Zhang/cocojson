import argparse

from cocojson.tools.filter_score import filter_and_viz_by_score

def main():
    parser = argparse.ArgumentParser(description="根据score过滤并可视化COCO图片")
    parser.add_argument('json_path', type=str, help='COCO标注json文件路径')
    parser.add_argument('img_root', type=str, help='图片根目录')
    parser.add_argument('out_dir', type=str, help='输出目录')
    parser.add_argument('--min_score', type=float, default=0.5, help='最小score阈值')
    parser.add_argument('--max_score', type=float, default=None, help='最大score阈值')
    parser.add_argument('--draw', action='store_true', help='是否画框')
    args = parser.parse_args()

    filter_and_viz_by_score(
        json_path=args.json_path,
        img_root=args.img_root,
        out_dir=args.out_dir,
        min_score=args.min_score,
        max_score=args.max_score,
        draw=args.draw,
    )

if __name__ == "__main__":
    main()