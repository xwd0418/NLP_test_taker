from transformers import AutoTokenizer, AutoModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import tqdm
import pickle, random
import torch, os, pytorch_lightning as pl, glob
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import torch.nn.functional as F
from datasets import load_dataset

### downloading dataset:
dataset_together = load_dataset("openlifescienceai/medmcqa")

### downloading model:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = "healx/gpt-2-pubmed-medium"
max_length = 100
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.padding_side='left'
tokenizer.pad_token = tokenizer.eos_token
print(f'[Main] using GPU : {torch.cuda.get_device_name()}')


def generate_datasets_with_prompt_and_save(split = "val", batchsize=8):   
    assert(split in ["val", "train", "test"])
    
    split_for_dataset = split if split != "val" else "validation"
    dataset = dataset_together[split_for_dataset]
    questions = dataset['question']
    input_prompts_hint = list(map(lambda q: f"{q} Hint:",questions))
    
    # generating hints 
    questions_with_hints = []
    for i in tqdm.tqdm(range(len(input_prompts_hint)//batchsize + 1)):
        print(i," out of ", len(input_prompts_hint)//batchsize + 1)
        inputs = tokenizer(input_prompts_hint[batchsize*i : min(batchsize*(i+1), len(input_prompts_hint))], 
                           return_tensors="pt",  padding=True, truncation=True).to(device)

        # Generate text by feeding the encoded input through the model and sampling output tokens
        outputs = model.generate(input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"], 
                                max_length=max_length+inputs["input_ids"].shape[1], 
                                num_beams=10, early_stopping=True,
                                repetition_penalty=4.2,
                                pad_token_id=50256
                                )
        generated_text = tokenizer.batch_decode(outputs, 
                                        skip_special_tokens=True
                                        )
        # break


        questions_with_hints+=generated_text
    
            
    
    # tokenize everything
    A,B,C,D = dataset['opa'], dataset['opb'],dataset['opc'],dataset['opd']
    subject_names = dataset['subject_name']
    topic_names = dataset['topic_name']
    
    labels = dataset['cop']
    for downstream_model_name in ['general', "pubmed"]:
        if downstream_model_name == "general":
            finetuning_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            finetuning_tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
            
        input_prompts = list(map(lambda a,b,c,d,q,subject,topic: f"Subject: {subject}, Topic: {topic}\nQuestion with hint: {q}\nA: {a}\nB: {b}\nC: {c}\nD: {d}\n",
                                A,B,C,D,questions_with_hints, subject_names, topic_names))
        
        tokens = finetuning_tokenizer(input_prompts, padding=True, truncation=True, return_tensors="pt")
        torch.save(torch.utils.data.TensorDataset(tokens.input_ids, tokens.attention_mask, torch.tensor(labels)), 
                   f"/root/pubmedQA_291/dataset_pickles/medmcqa/hints-added-tokenized-by-{downstream_model_name}-bert/classification_style/{split}.pkl")
    
            
    # torch.utils.data.TensorDataset(tokens.input_ids, tokens.attention_mask, torch.tensor(labels))


import sys
generate_datasets_with_prompt_and_save(sys.argv[1])
    