from trainer import Trainer
from models import calculate_model_size, modelspecifications
import pandas as pd

models = ['roberta', 't5','bart', 'gpt2', 'convbert']
tasks = ['cognitive-bias', 'fake-news','gender-bias', 'hate-speech',  'political-bias','racial-bias', 'text-level-bias']
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

    df_results = pd.DataFrame(results).T  # il dataframe viene trasposto per avere i modelli nelle righe
    df_results['Model'] = df_results.index  # colonna con i nomi dei modelli

for model in models:
    model_instance, _, _ = modelspecifications(model)
    df_results.loc[model, 'Size (MB)'] = calculate_model_size(model_instance, model)

cols = ['Model', 'Size (MB)'] + tasks  
df_results = df_results[cols]  #riordinamento delle colonne

df_results.to_csv('training_results.csv', index=False)  # Salvataggio del dataframe ottenuto 

print("Dataframe salvato in training_results.csv")
print(df_results)