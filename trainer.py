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
    def __init__(self, number_of_folds: int, model: str, batch_size: int, task: str,max_epoch:int, eval_only: bool, max_length: int):
        self.seed = 42
        self.set_random_seed()
        self.max_length = max_length
        self.model_name = model
        self.eval_only = eval_only
        self.max_length = max_length
        self.number_of_folds = number_of_folds
        self.batch_size = batch_size
        self.task = task
        self.max_epoch = max_epoch
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") #assegnazione del dispositivo di esecuzione all'attributo 'device' dell'istanza

        #imposta il seed per la generazione di numeri casuali nelle librerie utilizzate per la riproducibilità dei risultati durante l'addestramento
    def set_random_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_data(self):
        data_path = os.getcwd() + "/datasets/mbib-full/" + self.task + ".csv"
        df = pd.read_csv(data_path)
        return df

    #tokenizzazione dei dati del dataset
    def tokenize_data(self, df, tokenizer):
        tokenized = []
        print("Tokenizing...")

        for i in tqdm(range(len(df))):
            tok = tokenizer(df.iloc[i]['text'], padding="max_length",truncation=True) #tutte le sequenze di token hanno la stessa lunghezza massima
            #elaborazione e trasformazione dei token
            tok['input_ids'] = torch.tensor(tok['input_ids'])
            tok['attention_mask'] = torch.tensor(tok['attention_mask'])
            tok['labels'] = torch.tensor(df.iloc[i]['label'])
            if 'token_type_ids' in tok.keys():
              tok['token_type_ids'] = torch.tensor(tok['token_type_ids'])
            tokenized.append(tok)

        return tokenized
    

    def evaluate(self,model,dl):
        num_steps = len(dl) #numero di batch presenti nel dataloader
        progress_bar = tqdm(range(num_steps))

        loss = 0
        predictions = []
        truth = []


        model.eval()
        #loop sui batch del dataloader che memorizza sia le predizioni del modello sia le etichette di verità
        for batch in dl:
            labels = list(batch['labels'].detach().cpu().numpy()) #ottiene le etichette vere del batch e le converte in una lista di numpy. le etichette sono spostate sulla cpu

            batch = {k: v.to(self.device) for k, v in batch.items()} #muove i tensori del batch sulla GPU

            #disabilita il calcolo del gradiente durante la valutazione per risparmiare memoria
            with torch.no_grad():
                outputs = model(**batch) #esegue il modello passando il batch e ottiene gli output del modello
            logits = outputs.logits #ottiene la predizione prima che venga applicata la funzione di attivazione
            loss += outputs.loss #aggiunge la loss ottenuta per il batch alla loss finale
            predictions.extend(torch.argmax(logits, dim=-1)) #aggiunge le predizioni del modello per ottenere l'indice della classe con la probabilità massima
            truth.extend(labels) #aggiunge le etichette di verità al valore truth

            progress_bar.update(1) #aggiorna la barra di avanzamento

        loss = loss/num_steps  #calcolo della loss media
        predictions = torch.stack(predictions).cpu() #conversione delle previsioni del modello in un tensore pytorch
        report = classification_report(truth, predictions, target_names=['non-biased', 'biased'],
                                               output_dict=True) #calcola il report delle metriche di classificazione tra le etichette di verità e le predizioni del modello
        return loss, report
    
    def fit(self,model,optimizer,lr_scheduler,train_dl,dev_dl):
        num_training_steps = self.max_epoch * len(train_dl)
        last_loss = 100
        patience = 1
        trigger = 0 #contatore di epoche senza addestramenti

        for epoch in trange(self.max_epoch, desc='Epoch'): #trange serve per ottenere una barra di avanzamento con un'indicazione del progresso
            progress_bar = tqdm(range(len(train_dl)))

            model.train()
            #addestramento del modello
            for batch in train_dl:
                optimizer.zero_grad()   #azzera i gradienti prima di effettuare il passo backward
                batch = {k: v.to(self.device) for k, v in batch.items()}   #spostamento dei dati sulla cpu/gpu
                outputs = model(**batch)   #esecuzione del modello con il batch corrente
                loss = outputs.loss    #estrazione della loss dall'output
                loss.backward()   #calcolo dei gradienti dei pesi rispetto alla loss, utilizzando la tecnica dell'Automatic Differentiation di PyTorch
                optimizer.step()    #in base ai gradienti calcolati ottimizza i pesi del modello
                lr_scheduler.step()   #eventuali aggiornamenti al learning rate
                progress_bar.update(1)   #aggiorna la barra di progresso dell'addestramento


            dev_loss,_ = self.evaluate(model,dev_dl)   #valutazione del modello sul validation set e viene estratta la loss media

            # early stopping
            if dev_loss >= last_loss:
                trigger += 1
                if trigger >= patience:
                    print('Early stopping...')
                    break
            else:
                trigger = 0
            last_loss = dev_loss

        return model

    def run(self):
        #estrazione del modello, tokenizer e learning rate
        model, tokenizer, lr = modelspecifications(name=self.model_name, model_length=self.max_length)

        # carica il dataset, lo tokenizza e divide i dati per effettuare la cross validation
        df = self.load_data()
        tokenized_df = self.tokenize_data(df,tokenizer)
        skfold = StratifiedKFold(n_splits=self.number_of_folds, shuffle=True, random_state=self.seed)
        scores = []

        #divide il dataset in base alla colonna 'dataset_id', in quanto il dataset completo comprende più dataset
        for train_idx, dev_idx in skfold.split(np.arange(len(df)),df['dataset_id'].to_list()):
            dev_idx, test_idx = train_test_split(dev_idx, test_size=0.75, train_size=0.25, random_state=42, shuffle=True)  #

            #campionamento casuale degli elementi di ogni set di dati
            train_sampler = SubsetRandomSampler(train_idx)
            dev_sampler = SubsetRandomSampler(dev_idx)
            test_sampler = SubsetRandomSampler(test_idx)

            #creazione di dataloader per mescolare i dati e raccoglierli in dei batch
            train_dl = DataLoader(tokenized_df, batch_size=self.batch_size, sampler=train_sampler)
            dev_dl = DataLoader(tokenized_df, batch_size=self.batch_size, sampler=dev_sampler)
            test_dl = DataLoader(tokenized_df, batch_size=self.batch_size, sampler=test_sampler)

            #viene creato un clone del modello originale, per far sì che il clone possa essere riaddestrato
            fold_model = copy.deepcopy(model)
            fold_model.to(self.device)

            #setup dei parametri del training
            optimizer = torch.optim.AdamW(fold_model.parameters(), lr=lr)
            lr_scheduler = get_scheduler(
                "cosine",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=self.max_epoch * len(train_dl)
            )

            #se si vuole utilizzare il modello pre-addestrato viene utilizzata la copia del modello creata in precedenza, altrimenti si addestra quest'ultima
            if self.eval_only:
                trained_model = fold_model
            else:
                trained_model = self.fit(fold_model,optimizer,lr_scheduler,train_dl,dev_dl)

            #valutazione del modello, da cui vengono estratte le metriche di interesse
            eval_loss, report = self.evaluate(trained_model,test_dl)
            scores.append(report['macro avg']['f1-score'])
        return sum(scores)/self.number_of_folds