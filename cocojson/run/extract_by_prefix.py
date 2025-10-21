#!/usr/bin/env python3
"""
从COCO JSON文件中根据文件名前缀提取数据子集。
"""

import argparse
from cocojson.tools.extract_by_prefix import extract_by_prefix

def main():
    parser = argparse.ArgumentParser(
        description='根据文件名前缀从COCO JSON中提取数据子集'
    )
    
    parser.add_argument(
        'json_path',
        help='输入的COCO JSON文件路径'
    )
    
    parser.add_argument(
        '-p', '--prefixes',
        nargs='+',
        required=True,
        help='要匹配的文件名前缀列表，可以提供多个前缀'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='输出文件的后缀名（可选）'
    )
    
    parser.add_argument(
        '--no-remaining',
        action='store_true',
        help='设置此标志则不保存剩余的数据'
    )
    
    args = parser.parse_args()
    
    # 调用extract_by_prefix函数
    extract_by_prefix(
        args.json_path,
        args.prefixes,
        output_name=args.output,
        save_remaining=not args.no_remaining
    )

if __name__ == '__main__':
    main()