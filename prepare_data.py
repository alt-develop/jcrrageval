import argparse
import json
import os
import pandas as pd
import numpy as np

def filter_too_long(context, max_length):
    # df = df[df["context"].apply(lambda x: len(x) <= max_length)]
    if len(context) > max_length:
        context = context.split("\n")
        context = context[:4]
        context = "\n".join(context)
    return context


def prepare_data(df: pd.DataFrame, context_col, question_col, answer_col, max_samples, task):
    # filter too long context
    max_length = 20000
    # df = filter_too_long(df, max_length).reset_index(drop=True)
    df[context_col] = df[context_col].apply(lambda x: filter_too_long(x, max_length))
    
    df = df[:max_samples]
    df = df.reset_index(drop=True)
    
    df["input"] = df.apply(
        lambda x: f"質問: {x[question_col]}\n 段落: {x[context_col]}", axis=1
    )

    df["task"] = task
    if "id" not in df.columns:
        df["id"] = df.index + 1
    df = df.rename(columns={answer_col: "output"})
    
    df = df[["task", "id", "input", "output", 'question', 'context']]
    df = df.drop_duplicates(subset=["id"])
    # df = df[['id', 'instruction', 'input', 'output']]
    
    print(f"Total samples: {len(df)}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare data for RAG")
    parser.add_argument("--input_file", required=True, help="Path to input data file")
    parser.add_argument(
        "--context_col", default="context", help="Column name for context"
    )
    parser.add_argument(
        "--question_col", default="question", help="Column name for question"
    )
    parser.add_argument("--answer_col", default="answer", help="Column name for answer")
    parser.add_argument(
        "--max_samples", type=int, default=10000, help="Max rows to process"
    )
    parser.add_argument(
        "--task", type=str, default="rag", help="Task name"
    )
    args = parser.parse_args()

    if args.input_file.endswith(".json"):
        df = pd.read_json(args.input_file)
    elif args.input_file.endswith(".jsonl"):
        df = pd.read_json(args.input_file, lines=True)
    elif args.input_file.endswith(".csv"):
        df = pd.read_csv(args.input_file)
    elif args.input_file.endswith(".tsv"):
        df = pd.read_csv(args.input_file, sep="\t")
    else:
        raise ValueError("Invalid file format")

    max_samples = args.max_samples
    context_col = args.context_col
    question_col = args.question_col
    answer_col = args.answer_col
    # os.makedirs("instruction_formated", exist_ok=True)
    if max_samples > 0:
        output_df = prepare_data(df, context_col, question_col, answer_col, max_samples, args.task)
        output_file = os.path.join(
            "data", os.path.basename(args.input_file).split(".")[0] + ".json"
        )
        # output_file ="instruction_formated/" +  os.path.basename(args.input_file).split(".")[0] + ".json"
        output_df.to_json(output_file, orient="records", force_ascii=False, indent=2)


if __name__ == "__main__":
    main()
