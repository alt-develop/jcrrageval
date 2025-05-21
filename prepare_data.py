import argparse
import os
import pandas as pd
from datasets import load_dataset


def filter_context_length(context: str, max_length: int = 20000) -> str:
    """Truncate context to first 4 lines if exceeds length limit."""
    if len(context) > max_length:
        return "\n".join(context.split("\n")[:4])
    return context


def process_sheet(df: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    """Process Excel sheet data."""
    # Normalize column names and validate
    df.columns = df.columns.str.lower()
    required_cols = {"context", "question", "answer"}

    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    # Select relevant columns and clean data
    df = df[["context", "question", "answer"]].copy()
    df["context"] = df["context"].astype(str)
    df["context"] = df["context"].apply(filter_context_length)

    df["task"] = "rag"
    df["input"] = df.apply(
        lambda x: f"質問: {x['question']}\n 段落: {x['context']}", axis=1
    )
    df.rename(
        columns={
            "answer": "output",
        },
        inplace=True,
    )

    # Apply sampling and deduplication
    return df.head(max_samples).drop_duplicates().reset_index(drop=True)


def read_dataset(
    dataset_name: str = "ducalt/jcrrag", split: str = "train"
) -> pd.DataFrame:
    """Load dataset from Hugging Face."""
    dataset = load_dataset(dataset_name, split=split)
    df = dataset.to_pandas()
    # rename columns to match expected format
    df.rename(
        columns={
            "ID": "id",
            "Context": "context",
            "Question": "question",
            "GroundtruthAnswer": "output",
        },
        inplace=True,
    )

    df["context"] = df["context"].astype(str)
    df["context"] = df["context"].apply(filter_context_length)
    df["task"] = "rag"
    df["input"] = df.apply(
        lambda x: f"質問: {x['question']}\n 段落: {x['context']}", axis=1
    )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Convert Excel sheets to RAG-ready JSON format"
    )
    parser.add_argument(
        "--use_hf_dataset",
        action="store_true",
        default=True,
        help="Use Hugging Face dataset instead of Excel file",
    )
    parser.add_argument("--input_file", help="Input Excel file path")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000,
        help="Maximum samples per sheet (default: 10000)",
    )

    args = parser.parse_args()
    os.makedirs("data", exist_ok=True)

    if args.use_hf_dataset:
        df = read_dataset()
        output_file = os.path.join("data", "jcrrag.json")
        df.to_json(output_file, orient="records", force_ascii=False, indent=2)
        print(f"Success: {output_file} ({len(df)} entries)")
    elif args.input_file.lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(args.input_file, sheet_name=None)
        for sheet_name, df in sheets.items():
            try:
                processed = process_sheet(df, args.max_samples)
                output_file = os.path.join("data", f"{sheet_name}.json")
                processed.to_json(
                    output_file, orient="records", force_ascii=False, indent=2
                )
                print(
                    f"Success: {sheet_name} → {output_file} ({len(processed)} entries)"
                )
            except ValueError as e:
                print(f"Skipping {sheet_name}: {str(e)}")


if __name__ == "__main__":
    main()