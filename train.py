from helpers.models import calculate_model_size, modelspecifications
from transformers import logging as hf_logging
from helpers.trainer import Trainer
from datetime import datetime
import pandas as pd
import argparse
import warnings
import torch
import os


def load_existing_results(filename, tasks, models):
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
    # Suppress Hugging Face transformers warnings
    hf_logging.set_verbosity_error()

    # Suppress specific warnings globally if needed
    warnings.filterwarnings("ignore")

    # Parse command-line arguments with default values
    parser = argparse.ArgumentParser(description="Train or evaluate models on specific tasks.")

    # Accept a list of models and tasks
    parser.add_argument('--models', nargs='+',
                        default=['robertatwitter', 'electra', 'bart', 'convbert', 'gpt2', 'roberta', 't5'],
                        help='List of models to be trained (default: [robertatwitter, electra, bart, convbert, gpt2, roberta, t5])')
    parser.add_argument('--tasks', nargs='+',
                        default=['cognitive-bias', 'fake-news', 'gender-bias', 'hate-speech', 'linguistic-bias', 'political-bias', 'racial-bias', 'text-level-bias'],
                        help='List of tasks for bias detection (default: [cognitive-bias, fake-news, gender-bias, etc.])')

    # Default values for other parameters
    parser.add_argument('--number_of_folds', type=int, default=5, help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length for tokenization (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training (default: 10)')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model without training')

    args = parser.parse_args()

    # Print all the parameters at the beginning
    print("Starting the training/evaluation with the following parameters:")
    print(f"Models: {args.models}")
    print(f"Tasks: {args.tasks}")
    print(f"Number of folds: {args.number_of_folds}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max sequence length: {args.max_length}")
    print(f"Epochs: {args.epochs}")
    print(f"Evaluation only: {args.eval}")

    # Check if PyTorch is using a GPU and print the device info
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Generate the current timestamp in the format YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the filename with the current timestamp
    filename = f'results/metrics_{timestamp}.csv'

    # Ensure the 'results' directory exists
    if not os.path.exists('results'):
        os.makedirs('results')

    # Load or initialize the results DataFrame
    df_results = load_existing_results(filename, args.tasks, args.models)

    # Iterate over the list of models and tasks
    for model in args.models:
        model_instance, _, _ = modelspecifications(model)
        model_size = calculate_model_size(model_instance, model)
        df_results.loc[model, 'Size (MB)'] = model_size

        for task in args.tasks:
            # Configuration for the trainer
            trainer_config = {
                "number_of_folds": args.number_of_folds,
                "model": model,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "task": task,
                "max_epoch": args.epochs,
                "eval_only": args.eval
            }

            # Train or evaluate the model for the specific task
            if args.eval:
                print(f"Evaluating model {model} on task {task}")
                task_result = train_model_for_task(model, task, trainer_config)
            else:
                print(f"Training model {model} on task {task} for {args.epochs} epochs")
                task_result = train_model_for_task(model, task, trainer_config)

            # Save the results
            update_results_and_save(df_results, model, task, task_result, filename, args.tasks)

    print(f"DataFrame updated and saved in {filename}")
    print(df_results)


if __name__ == "__main__":
    main()