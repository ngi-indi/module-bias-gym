import copy
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import trange
from tqdm.auto import tqdm
from transformers import get_scheduler
from models import modelspecifications


class Trainer:
    def __init__(self, number_of_folds: int, model: str, batch_size: int, task: str, max_epoch: int, eval_only: bool, max_length: int):
        # Initialization of class properties with configurable parameters and initial device setup
        self.seed = 42
        self.set_random_seed()
        self.max_length = max_length
        self.model_name = model
        self.eval_only = eval_only
        self.number_of_folds = number_of_folds
        self.batch_size = batch_size
        self.task = task
        self.max_epoch = max_epoch
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def set_random_seed(self):
        # Setting the seed to ensure reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_data(self):
        # Loading and cleaning data
        data_path = os.getcwd() + "/training_datasets/" + self.task + ".csv"
        df = pd.read_csv(data_path)
        df = df.dropna()  # removal of NaN attributes
        df = df[df['label'].isin([0, 1])]  # Filtering rows with labels 0 or 1
        df = df.reset_index(drop=True)
        return df

    def tokenize_data(self, df, tokenizer):
        # Tokenizing the dataset using the specified tokenizer
        tokenized = []
        print("Tokenizing...")
        for i in tqdm(range(len(df))):
            # processing and transforming tokens
            tok = tokenizer(df.iloc[i]['text'], max_length=self.max_length, padding="max_length", truncation=True)
            tok['input_ids'] = torch.tensor(tok['input_ids'])
            tok['attention_mask'] = torch.tensor(tok['attention_mask'])
            tok['labels'] = torch.tensor([df.iloc[i]['label']])
            if 'token_type_ids' in tok.keys():
                tok['token_type_ids'] = torch.tensor(tok['token_type_ids'])
            tokenized.append(tok)
        return tokenized

    def evaluate(self, model, dl):
        # Evaluating the model on a provided DataLoader, calculating average loss and generating a classification report
        num_steps = len(dl)
        progress_bar = tqdm(range(num_steps))
        loss = 0
        predictions = []
        truth = []
        model.eval()

        for batch in dl:
            labels = list(batch['labels'].detach().cpu().numpy())
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            loss += outputs.loss
            predictions.extend(torch.argmax(logits, dim=-1))
            truth.extend(labels)
            progress_bar.update(1)

        loss = loss / num_steps
        predictions = torch.stack(predictions).cpu()
        report = classification_report(truth, predictions, labels=[0, 1], target_names=['non-biased', 'biased'], output_dict=True)
        return loss, report
    
    def fit(self, model, optimizer, lr_scheduler, train_dl, dev_dl):
        last_loss = 100
        patience = 1
        trigger = 0

        for epoch in trange(self.max_epoch, desc='Epoch'):
            progress_bar = tqdm(range(len(train_dl)))
            model.train()

            for batch in train_dl:
                batch['input_ids'] = batch['input_ids'].squeeze(1)
                optimizer.zero_grad()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
            dev_loss, _ = self.evaluate(model, dev_dl)

            if dev_loss >= last_loss:
                trigger += 1
                if trigger >= patience:
                    print('Early stopping...')
                    break
            else:
                trigger = 0
            last_loss = dev_loss
            
        return model

    #saving the best model obtained 
    def save_model(self, model, model_name, task_name):
        model_save_path = os.path.join('Models')
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        model_filename = f"{model_name}_best_{task_name}.pt"
        torch.save(model.state_dict(), os.path.join(model_save_path, model_filename))
        print(f"Modello migliore {model_name} per il task {task_name} salvato in {model_filename}")
 

    def run(self):
        #model extraction, tokenizer and learning rate
        model, tokenizer, lr = modelspecifications(name=self.model_name, model_length=self.max_length)

        tokenizer.pad_token = tokenizer.eos_token

        # loading the dataset, tokenization and division of the data to perform cross validation
        df = self.load_data()
        tokenized_df = self.tokenize_data(df,tokenizer)
        skfold = StratifiedKFold(n_splits=self.number_of_folds, shuffle=True, random_state=self.seed)
        scores = []

        best_model = None
        best_score = float('-inf')

        # division of the dataset based on the 'dataset_id' column, as the complete dataset includes multiple datasets
        for train_idx, dev_idx in skfold.split(np.arange(len(df)),df['dataset_id'].to_list()):
            dev_idx, test_idx = train_test_split(dev_idx, test_size=0.75, train_size=0.25, random_state=42, shuffle=True)  #

            # random sampling of elements from each data set
            train_sampler = SubsetRandomSampler(train_idx)
            dev_sampler = SubsetRandomSampler(dev_idx)
            test_sampler = SubsetRandomSampler(test_idx)

            # creation of data loaders to mix data and collect them in batches
            train_dl = DataLoader(tokenized_df, batch_size=self.batch_size, sampler=train_sampler)
            dev_dl = DataLoader(tokenized_df, batch_size=self.batch_size, sampler=dev_sampler)
            test_dl = DataLoader(tokenized_df, batch_size=self.batch_size, sampler=test_sampler)

            # creation of a clone of the original model, so that the clone can be retrained
            fold_model = copy.deepcopy(model)
            fold_model.to(self.device)

            # training parameters
            optimizer = torch.optim.AdamW(fold_model.parameters(), lr=lr)
            lr_scheduler = get_scheduler(
                "cosine",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=self.max_epoch * len(train_dl)
            )

            # if you want to use the pre-trained model, the previously created copy of the model is used, otherwise the latter is trained
            if self.eval_only:
                trained_model = fold_model
            else:
                trained_model = self.fit(fold_model,optimizer,lr_scheduler,train_dl,dev_dl)

            # model evaluation with the extraction of the macro avg f1-score
            eval_loss, report = self.evaluate(trained_model,test_dl)
            scores.append(report['macro avg']['f1-score'])

            if report['macro avg']['f1-score'] > best_score:
                best_score = report['macro avg']['f1-score']
                best_model = trained_model

        if best_model is not None:
            self.save_model(best_model, self.model_name, self.task)
            
        return sum(scores) / self.number_of_folds
