from transformers import (RobertaTokenizer, RobertaForSequenceClassification, BartTokenizer, 
                          BartForSequenceClassification, GPT2Tokenizer, GPT2ForSequenceClassification, 
                          ConvBertTokenizer, ConvBertForSequenceClassification, T5Tokenizer, 
                          T5ForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM)
import torch
import os

def modelspecifications(name, model_length=128):
    if name == "roberta":
        roberta_tokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base", model_max_length=model_length, use_fast=False)
        roberta_model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base', num_labels=2)
        learning_rate = 5e-5
        return roberta_model, roberta_tokenizer, learning_rate

    elif name == "bart":
        bart_tokenizer = BartTokenizer.from_pretrained(
            "facebook/bart-base", model_max_length=model_length)
        bart_model = BartForSequenceClassification.from_pretrained(
            "facebook/bart-base", num_labels=2)
        learning_rate = 5e-5
        return bart_model, bart_tokenizer, learning_rate

    elif name == "convbert":
        convbert_tokenizer = ConvBertTokenizer.from_pretrained(
            'YituTech/conv-bert-base', model_max_length=model_length)
        convbert_model = ConvBertForSequenceClassification.from_pretrained(
            'YituTech/conv-bert-base', num_labels=2)
        learning_rate = 5e-5
        return convbert_model, convbert_tokenizer, learning_rate

    elif name == "gpt2":
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
            "gpt2", model_max_length=model_length)
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model = GPT2ForSequenceClassification.from_pretrained(
            'gpt2', num_labels=2)
        gpt2_model.config.pad_token_id = gpt2_tokenizer.pad_token_id
        learning_rate = 5e-5
        return gpt2_model, gpt2_tokenizer, learning_rate

    elif name == "t5":
        t5_tokenizer = T5Tokenizer.from_pretrained(
            "t5-small", model_max_length=model_length)
        t5_model = T5ForConditionalGeneration.from_pretrained(
            "t5-small", num_labels=2)
        learning_rate = 5e-5
        return t5_model, t5_tokenizer, learning_rate

    elif name == "llama":
        llama_tokenizer = LlamaTokenizer.from_pretrained(
            "enoch/llama-7b-hf", model_max_length=model_length)
        llama_model = LlamaForCausalLM.from_pretrained(
            "enoch/llama-7b-hf", num_labels = 2)
        learning_rate = 5e-5
        return llama_model, llama_tokenizer, learning_rate

    else:
        print('Model not found')
        raise ValueError
    

def calculate_model_size(model, model_name):
    temp_path = f"temp_{model_name}.bin"
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024*1024)
    os.remove(temp_path)
    size_mb = round(size_mb, 3)
    return size_mb