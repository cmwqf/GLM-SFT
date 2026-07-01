"""
数据准备：从 JSONL 转换为 verl 需要的 Parquet 格式，并抽取子集用于测试。

用法:
    python scripts/03_prepare_data.py [--num_samples 1000] [--max_length 8192]

输出:
    output/data/train_1000.parquet  — 训练集 (950 条)
    output/data/val_1000.parquet    — 验证集 (50 条)
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd


INPUT_FILE = "/home/ubuntu/datasets/model-data-training/glm_chatml/train.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "data"


def load_jsonl(path: str, num_samples: int | None = None) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    if num_samples and num_samples < len(records):
        random.seed(42)
        records = random.sample(records, num_samples)
    return records


def analyze_records(records: list[dict]):
    msg_counts = [len(r["messages"]) for r in records]
    char_counts = [sum(len(m["content"]) for m in r["messages"]) for r in records]
    print(f"  样本数: {len(records)}")
    print(f"  消息轮数: min={min(msg_counts)}, max={max(msg_counts)}, avg={sum(msg_counts)/len(msg_counts):.0f}")
    print(f"  字符数:   min={min(char_counts)}, max={max(char_counts)}, avg={sum(char_counts)/len(char_counts):.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_FILE, help="输入 JSONL 文件路径")
    parser.add_argument("--num_samples", type=int, default=1000, help="抽取样本数")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"从 {args.input} 加载数据...")
    records = load_jsonl(args.input, args.num_samples)

    print(f"\n--- 数据统计 ---")
    analyze_records(records)

    # 划分训练集和验证集
    random.seed(42)
    random.shuffle(records)
    val_size = max(1, int(len(records) * args.val_ratio))
    val_records = records[:val_size]
    train_records = records[val_size:]

    # 转换为 DataFrame 并保存为 Parquet
    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)

    train_path = output_dir / f"train_{args.num_samples}.parquet"
    val_path = output_dir / f"val_{args.num_samples}.parquet"

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    print(f"\n--- 输出 ---")
    print(f"  训练集: {train_path} ({len(train_df)} 条)")
    print(f"  验证集: {val_path} ({len(val_df)} 条)")
    print(f"\n完成!")


if __name__ == "__main__":
    main()
