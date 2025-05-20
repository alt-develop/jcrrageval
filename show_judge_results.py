import pandas as pd
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser


def load_and_clean_data(file_path: str, task) -> pd.DataFrame:
    df = pd.read_json(file_path, lines=True)
    df = df[df["task"] == task]
    df = df.drop_duplicates(subset=["model", "task", "id"], keep="last").reset_index(
        drop=True
    )
    df = df.drop(columns=["id", "task"])
    # drop na columns
    df = df.dropna(axis=1, how="all")
    grouped_df = df.groupby(["model"]).mean(numeric_only=True)
    grouped_df["total_score"] = grouped_df.sum(axis=1)
    grouped_df = grouped_df.round(3)
    grouped_df = grouped_df.sort_values(by="total_score", ascending=False)
    return grouped_df


def filter_data(pivot_df: pd.DataFrame, model_list: list) -> pd.DataFrame:
    filtered_df = pivot_df.loc[model_list]
    filtered_df = filtered_df.sort_values(by="total_score", ascending=False)
    return filtered_df


def plot_and_save_results(filtered_df: pd.DataFrame, benchmark_name, task) -> None:
    plt.figure(figsize=(10, 8))
    ax = filtered_df["total_score"].plot(kind="barh", color="skyblue")
    plt.xlabel("Total Score")
    plt.title("Total Score of Models")
    for index, value in enumerate(filtered_df["total_score"]):
        ax.text(value, index, f"{value:.3f}", va="center")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_dir = f"results/{benchmark_name}"
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(f"{output_dir}/model_total_scores_task_{task}.png")
    filtered_df.to_csv(f"{output_dir}/filtered_model_scores_task_{task}.csv")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("tight")
    ax.axis("off")
    filtered_df_reset = filtered_df.reset_index()
    the_table = ax.table(
        cellText=filtered_df_reset.values,
        colLabels=filtered_df_reset.columns,
        cellLoc="center",
        loc="center",
    )
    plt.savefig(
        f"{output_dir}/filtered_model_scores_table_task_{task}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.expand_frame_repr", False)
    print(filtered_df)


def main(benchmark_name: str, model_list, task, judger) -> None:
    file_path = f"judge/model_judgement/{benchmark_name}/{judger}.jsonl"
    pivot_df = load_and_clean_data(file_path, task)
    if model_list != "all":
        filtered_df = filter_data(pivot_df, model_list)
    else:
        filtered_df = pivot_df
    plot_and_save_results(filtered_df, benchmark_name, task)


if __name__ == "__main__":
    # main()
    parser = ArgumentParser()
    parser.add_argument(
        "--benchmark_name", type=str, required=True, help="Name of the benchmark file"
    )
    parser.add_argument(
        "--model_list", nargs="+", default='all', help="List of model IDs to evaluate"
    )
    parser.add_argument(
        "--judger", type=str, required=True, help="Name of the judger", default='gpt-4o'
    )
    parser.add_argument("--task", type=str, default='rag', help="Task to evaluate")
    args = parser.parse_args()

    main(args.benchmark_name, args.model_list, args.task, args.judger)
