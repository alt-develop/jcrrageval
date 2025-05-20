import pandas as pd
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser


def plot_and_save_results(avg_df: pd.DataFrame, benchmark_name, task) -> None:
    plt.figure(figsize=(10, 8))
    print(avg_df.columns)
    ax = avg_df[["f1_score", "em_score"]].plot(kind="barh", color="skyblue")
    plt.xlabel("Scores")
    plt.title("Score of Models")
    # for index, value in enumerate(avg_df["total_score"]):
    #     ax.text(value, index, f"{value:.3f}", va="center")
    for index, row in avg_df.iterrows():
        ax.text(row["f1_score"], index, f"{row['f1_score']:.3f}", va="center")
        ax.text(row["em_score"], index, f"{row['em_score']:.3f}", va="center")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_dir = f"results/{benchmark_name}"
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(f"{output_dir}/model_f1_em_{task}.png")
    avg_df.to_csv(f"{output_dir}/filtered_model_f1_em_{task}.csv")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("tight")
    ax.axis("off")
    avg_df_reset = avg_df.reset_index()
    the_table = ax.table(
        cellText=avg_df_reset.values,
        colLabels=avg_df_reset.columns,
        cellLoc="center",
        loc="center",
    )
    plt.savefig(
        f"{output_dir}/filtered_model_f1_em_table_task_{task}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.expand_frame_repr", False)
    print(avg_df)


def main(benchmark_name: str, model_list: list | str, task) -> None:
    result_dir = f"judge/auto_metrics/{benchmark_name}/"
    dfs = []
    if model_list == 'all':
        model_list = [model.split(".jsonl")[0] for model in os.listdir(result_dir)]
    print(model_list)
    for model in model_list:
        df = pd.read_json(f"{result_dir}/{model}.jsonl", lines=True)
        df["model"] = model
        df = df.drop(columns=["id"])
        dfs.append(df)
    dfs = pd.concat(dfs)
    avg_df = dfs.groupby("model").mean(numeric_only=True).sort_values(by="f1_score", ascending=False)
    avg_df = avg_df.reset_index()
    plot_and_save_results(avg_df, benchmark_name, task)


if __name__ == "__main__":
    # main()
    parser = ArgumentParser()
    parser.add_argument(
        "--benchmark_name", type=str, required=True, help="Name of the benchmark file"
    )
    parser.add_argument(
        "--model_list", nargs="+", default='all', help="List of model IDs to evaluate"
    )
    parser.add_argument("--task", type=str, default='rag', help="Task to evaluate")
    args = parser.parse_args()

    main(args.benchmark_name, args.model_list, args.task)
