from transformers import ( AutoModelForSequenceClassification, AutoTokenizer, BartTokenizer, BartForSequenceClassification, ConvBertTokenizer, ConvBertForSequenceClassification, ElectraForSequenceClassification, ElectraTokenizer, GPT2Tokenizer, GPT2ForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration )
import torch
import os


DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_MODEL_LENGTH = 128


# Helper function to load a model and tokenizer
def load_model_and_tokenizer(model_class, tokenizer_class, model_name, model_length=DEFAULT_MODEL_LENGTH, num_labels=2):
    tokenizer = tokenizer_class.from_pretrained(model_name, model_max_length=model_length)
    model = model_class.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer


# Main function to specify models based on the provided name
def modelspecifications(name, model_length=DEFAULT_MODEL_LENGTH):
    model_name_map = {
        "roberta": ("roberta-base", RobertaForSequenceClassification, RobertaTokenizer),
        "bart": ("facebook/bart-base", BartForSequenceClassification, BartTokenizer),
        "electra": ("google/electra-base-discriminator", ElectraForSequenceClassification, ElectraTokenizer),
        "convbert": ("YituTech/conv-bert-base", ConvBertForSequenceClassification, ConvBertTokenizer),
        "robertatwitter": ("cardiffnlp/twitter-roberta-base", AutoModelForSequenceClassification, AutoTokenizer),
        "gpt2": ("gpt2", GPT2ForSequenceClassification, GPT2Tokenizer),
        "t5": ("t5-small", T5ForConditionalGeneration, T5Tokenizer)
    }

    if name in model_name_map:
        model_name, model_class, tokenizer_class = model_name_map[name]
        model, tokenizer = load_model_and_tokenizer(model_class, tokenizer_class, model_name, model_length)

        # Special case for GPT2
        if name == "gpt2":
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer, DEFAULT_LEARNING_RATE
    else:
        raise ValueError(f"Model '{name}' not found")


# Function to calculate the size of the model in MB
def calculate_model_size(model, model_name):
    temp_path = f"temp_{model_name}.bin"
    torch.save(model.state_dict(), temp_path)
    size_mb = round(os.path.getsize(temp_path) / (1024 * 1024), 3)
    os.remove(temp_path)
    return size_mb
