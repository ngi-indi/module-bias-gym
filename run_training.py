from trainer import Trainer
from models import calculate_model_size, modelspecifications
import pandas as pd
import os


# Definizione dei modelli e dei task
models = ['robertatwitter', 'electra']#['llama3'] #['llama']#['bart', 'convbert', 'gpt2', 'roberta', 't5' ]
tasks = [ 'cognitive-bias', 'fake-news','gender-bias', 'hate-speech', 'linguistic-bias','political-bias', 'racial-bias', 'text-level-bias']
filename = 'training_results_llama.csv'

# Definizione di un dizionario per mantenere i risultati
results = {model: {task: None for task in tasks} for model in models}

# Verifica dell'esistenza del file per caricare i risultati esistenti o creare un nuovo DataFrame
if os.path.exists(filename):
    df_results = pd.read_csv(filename)
    # Verifica della presenza di tutte le combinazioni modello-task e riordino delle colonne se necessario
    expected_cols = ['Model', 'Size (MB)'] + tasks
    missing_cols = [col for col in expected_cols if col not in df_results.columns]
    for col in missing_cols:
        df_results[col] = None  # Aggiunta delle colonne mancanti
    df_results = df_results[expected_cols]  # Riordino delle colonne
else:
    df_results = pd.DataFrame(results).T
    df_results['Model'] = df_results.index
    df_results['Size (MB)'] = None  # Inizializzazione della colonna per la grandezza del modello

# Addestramento dei modelli
for model in models:
    model_instance, _, _ = modelspecifications(model)
    df_results.loc[model, 'Size (MB)'] = calculate_model_size(model_instance, model)
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

        # Aggiornamento e salvataggio dei risultati nel DataFrame
        df_results.loc[model, task] = results[model][task]
        cols = ['Model', 'Size (MB)'] + tasks
        df_results = df_results[cols]   # Riordino delle colonne
        df_results.to_csv(filename, index=False)  # Salvataggio del DataFrame dopo ogni aggiornamento

print("Dataframe aggiornato e salvato in training_results.csv")
print(df_results)