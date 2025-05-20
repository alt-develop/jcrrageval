import pandas as pd
import matplotlib.pyplot as plt

def plot_model_comparison(model_list):
    # File path to the JSONL file
    file_path = 'model_judgement/gpt4o.jsonl'

    # Load the JSONL file
    df = pd.read_json(file_path, lines=True)

    # Remove the 'id' column
    df = df.drop(columns=['id'])

    # Group the data by model and task, and calculate the average score for each criteria
    grouped_df = df.groupby(['model', 'task']).mean()

    # Calculate the average score across all non-NaN criteria for each row
    grouped_df['average_score'] = grouped_df.mean(axis=1)

    # Filter the data based on the provided model list
    filtered_df = grouped_df.loc[grouped_df.index.get_level_values('model').isin(model_list)]

    # Pivot the table to have tasks as columns and models as rows
    pivot_df = filtered_df[['average_score']].unstack().reset_index()
    pivot_df.columns = ['Task', 'Model', 'Average Score']

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    for model in model_list:
        subset = pivot_df[pivot_df['Model'] == model]
        plt.bar(subset['Task'], subset['Average Score'], label=model)

    # Add labels and title
    plt.xlabel('Task')
    plt.ylabel('Average Score')
    plt.title('Comparison of Models on Different Tasks')
    plt.legend(title='Model')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig('model_comparison_bar_chart.png')

    # Show the plot
    plt.show()

# Example usage:
model_list = ["checkpoint-72_2609", "checkpoint-360_2609", "checkpoint-504_2609", "Qwen2.5-7B-Instruct"]
plot_model_comparison(model_list)
