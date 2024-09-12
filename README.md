This folder contains code for train and evaluate some transformers language models on multiple bias types. 
## Contents
### trainer.py 
This script contains the Trainer class which allows the training, evaluation, and saving of language models. 
It includes functionality for:
- Setting up the training environment.
- Loading and preprocessing data.
- Conducting training sessions with cross-validation.
- Evaluating models.
- Saving the best-performing models.

### run_training.py
This script utilizes the Trainer class to train models defined in the models.py script on specified tasks. 
It handles:
- Tasks and models definition.
- Loading of model configurations and specifications.
- Recording and updating training results in a CSV file.

### models.py 
this script contains the initialization of the models and their tokenizers.

## Usage 
Download the pre-trained model weights from [https://drive.google.com/drive/folders/1aOTVMTdLcDhOHuj-bcJbO5SPM7Zdh-_O?usp=drive_link](this Google Drive directory) and place them in a directory named 'training_models' within the same directory as the scripts.
Download the training datasets from [https://drive.google.com/drive/folders/1VSXZcAmDQj7Gk1_AEA1HI_dVVUF-sFmW?usp=drive_link](this Google Drive directory) and place them in a directory named 'training_datasets' within the same directory as the scripts.

To start the training process execute 'run_training.py'. After training, performance metrics will be store in the 'training_results.csv' file. 

