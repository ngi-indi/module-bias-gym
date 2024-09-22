from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report
from helpers.models import modelspecifications
from transformers import get_scheduler
from tqdm.auto import tqdm, trange
import pandas as pd
import numpy as np
import random
import torch
import copy
import os


class Trainer:

    def __init__(self, number_of_folds: int, model: str, batch_size: int, task: str, max_epoch: int, eval_only: bool, max_length: int):
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
        """Set random seed for reproducibility across experiments."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_data(self):
        """Load and preprocess the dataset."""
        data_path = os.path.join(os.getcwd(), "datasets", f"{self.task}.csv")
        df = pd.read_csv(data_path).dropna()
        df = df[df['label'].isin([0, 1])].reset_index(drop=True)
        return df

    def tokenize_data(self, df, tokenizer):
        """Tokenize the dataset using the specified tokenizer."""
        tokenized = []
        for i in tqdm(range(len(df))):
            tok = tokenizer(df.iloc[i]['text'], max_length=self.max_length, padding="max_length", truncation=True)
            tok = {key: torch.tensor(value) for key, value in tok.items()}
            tok['labels'] = torch.tensor([df.iloc[i]['label']])
            tokenized.append(tok)
        return tokenized

    def create_data_loader(self, tokenized_data, indices):
        """Create a DataLoader with random sampling."""
        sampler = SubsetRandomSampler(indices)
        return DataLoader(tokenized_data, batch_size=self.batch_size, sampler=sampler)

    def evaluate(self, model, data_loader):
        """Evaluate the model on a provided DataLoader and generate a classification report."""
        model.eval()
        total_loss, predictions, truth = 0, [], []

        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            batch = {key: value.to(self.device) for key, value in batch.items()}
            labels = batch['labels'].cpu().numpy()

            with torch.no_grad():
                outputs = model(**batch)
                total_loss += outputs.loss.item()
                logits = outputs.logits
                predictions.extend(torch.argmax(logits, dim=-1).cpu())
                truth.extend(labels)

        avg_loss = total_loss / len(data_loader)
        report = classification_report(truth, predictions, labels=[0, 1], target_names=['non-biased', 'biased'], output_dict=True)
        return avg_loss, report

    def fit(self, model, optimizer, lr_scheduler, train_loader, dev_loader):
        """Train the model with early stopping based on validation loss."""
        best_loss, patience, trigger = float('inf'), 1, 0

        # Single progress bar for all epochs
        with tqdm(total=len(train_loader) * self.max_epoch, desc='Training', ncols=100) as progress_bar:
            for epoch in range(self.max_epoch):
                model.train()

                # Loop over batches
                for batch in train_loader:
                    batch = {key: value.to(self.device) for key, value in batch.items()}
                    optimizer.zero_grad()
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()

                    # Update the progress bar with the current loss (overwrite in the same line)
                    progress_bar.set_postfix({"Epoch": epoch + 1, "Training Loss": f"{loss.item():.4f}"})
                    progress_bar.update(1)

                # After each epoch, evaluate the model on the validation set
                dev_loss, _ = self.evaluate(model, dev_loader)

                # Early stopping logic based on validation loss
                if dev_loss >= best_loss:
                    trigger += 1
                    if trigger >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        break
                else:
                    trigger, best_loss = 0, dev_loss

        return model

    def save_model(self, model, model_name, task_name):
        """Save the trained model to disk."""
        save_dir = os.path.join('models')
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{model_name}_best_{task_name}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Best model saved as: {model_path}")

    def run(self):
        """Execute the complete training and evaluation process with cross-validation."""
        model, tokenizer, lr = modelspecifications(name=self.model_name, model_length=self.max_length)
        tokenizer.pad_token = tokenizer.eos_token

        df = self.load_data()
        tokenized_data = self.tokenize_data(df, tokenizer)
        skfold = StratifiedKFold(n_splits=self.number_of_folds, shuffle=True, random_state=self.seed)

        best_model, best_score, scores = None, float('-inf'), []

        for train_idx, dev_idx in skfold.split(df, df['dataset_id']):
            dev_idx, test_idx = train_test_split(dev_idx, test_size=0.75, random_state=self.seed, shuffle=True)

            train_loader = self.create_data_loader(tokenized_data, train_idx)
            dev_loader = self.create_data_loader(tokenized_data, dev_idx)
            test_loader = self.create_data_loader(tokenized_data, test_idx)

            model_copy = copy.deepcopy(model).to(self.device)
            optimizer = torch.optim.AdamW(model_copy.parameters(), lr=lr)
            lr_scheduler = get_scheduler(
                "cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=self.max_epoch * len(train_loader)
            )

            trained_model = model_copy if self.eval_only else self.fit(model_copy, optimizer, lr_scheduler, train_loader, dev_loader)
            eval_loss, report = self.evaluate(trained_model, test_loader)
            scores.append(report['macro avg']['f1-score'])

            if report['macro avg']['f1-score'] > best_score:
                best_score, best_model = report['macro avg']['f1-score'], trained_model

        if best_model:
            self.save_model(best_model, self.model_name, self.task)

        return sum(scores) / self.number_of_folds
