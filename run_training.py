from trainer import Trainer
from models import calculate_model_size, modelspecifications

import pandas as pd

models = ['roberta', 'bart', 'gpt2', 'convbert']
tasks = ['cognitive-bias', 'fake-news', 'gender-bias', 'hate-speech', 'political-bias','racial-bias', 'text-level-bias']
results = {model: {} for model in models}

for model in models:
    for task in tasks:
        print(f'Addestramento del modello {model} sul task {task}')
        config = {
            "number_of_folds": 5,
            "model": model,
            "batch_size": 32,
            "max_length": 128,
            "task": task,
            "max_epoch": 10,
            "eval_only": False
        }
        trainer = Trainer(**config)
        results[model][task] = trainer.run()

    df_results = pd.DataFrame(results).T  # Traspose to have models as rows
    for model in models:
        model_instance, _, _ = modelspecifications(model)
        df_results.loc[model, 'Size (MB)'] = calculate_model_size(model_instance, model)
    cols = ['Size (MB)'] + tasks
    df_results = df_results[cols]
    print(df_results)