from helpers.models import calculate_model_size, modelspecifications
from helpers.trainer import Trainer
import pandas as pd
import os


def load_existing_results(filename, tasks):
    """Load existing results from a CSV or initialize a new DataFrame."""
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        # Ensure that all the required columns are present
        expected_cols = ['Model', 'Size (MB)'] + tasks
        missing_cols = [col for col in expected_cols if col not in df.columns]
        for col in missing_cols:
            df[col] = None  # Add missing columns
        df = df[expected_cols]  # Reorder columns
    else:
        df = pd.DataFrame({task: None for task in tasks}, index=models).T
        df['Model'] = df.index
        df['Size (MB)'] = None  # Initialize the size column
    return df


def train_model_for_task(model, task, trainer_config):
    """Train a model for a specific task and return the result."""
    print(f"Training model {model} on task {task}")
    trainer = Trainer(**trainer_config)
    return trainer.run()


def update_results_and_save(df_results, model, task, result, filename, tasks):
    """Update the results DataFrame and save it to a CSV file."""
    df_results.loc[model, task] = result
    cols = ['Model', 'Size (MB)'] + tasks
    df_results = df_results[cols]  # Ensure the correct order of columns
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Create 'results' directory if it doesn't exist
    df_results.to_csv(filename, index=False)  # Save after every update


def main():
    # Define models and tasks
    models = ['robertatwitter', 'electra', 'bart', 'convbert', 'gpt2', 'roberta', 't5']
    tasks = ['cognitive-bias', 'fake-news', 'gender-bias', 'hate-speech', 'linguistic-bias', 'political-bias', 'racial-bias', 'text-level-bias']
    filename = 'results/metrics.csv'

    # Load or initialize the results DataFrame
    df_results = load_existing_results(filename, tasks)

    # Train each model for each task
    for model in models:
        model_instance, _, _ = modelspecifications(model)
        model_size = calculate_model_size(model_instance, model)
        df_results.loc[model, 'Size (MB)'] = model_size

        for task in tasks:
            # Configuration for the trainer
            trainer_config = {
                "number_of_folds": 5,
                "model": model,
                "batch_size": 32,
                "max_length": 128,
                "task": task,
                "max_epoch": 10,
                "eval_only": False
            }

            # Train the model for the specific task and store the result
            task_result = train_model_for_task(model, task, trainer_config)
            update_results_and_save(df_results, model, task, task_result, filename, tasks)

    print(f"DataFrame updated and saved in {filename}")
    print(df_results)


if __name__ == "__main__":
    main()
