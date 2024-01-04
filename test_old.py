User
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer,AutoTokenizer,DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
import evaluate
import numpy as np
import pandas as pd
from datasets import load_metric
import json
import torch.nn as nn
from tqdm import tqdm
import torch
from utils import *
from datasets import load_dataset


def run_train():
    
    with open('banking77_info.json') as f:
        info = json.load(f)

    label_name = info['label']['names']
    idx2label = {idx:label for idx,label in enumerate(label_name)}
    label2idx = {label : idx for idx,label in enumerate(label_name)}
    
    test_name = 'original-con-base'



    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    
    train_dataset = load_dataset('json', data_files="/data/2_data_server/nlp-04/lost_technology/data/banking/5shot/CEAcombined.json")
    test_dataset = load_dataset('csv', data_files="/data/2_data_server/nlp-04/lost_technology/original_data/banking/test.csv")
    test_dataset = test_dataset.filter(lambda example: example['text'] is not None)
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    

    def compute_metrics(eval_pred):

        predictions, labels = eval_pred

        predictions = np.argmax(predictions, axis=1)
        
        # 정확도 계산
        accuracy = load_metric("accuracy")
        accuracy_value = accuracy.compute(predictions=predictions, references=labels)
        
        # F1 점수 계산
        f1 = load_metric("f1")
        f1_value = f1.compute(predictions=predictions, references=labels,average= 'weighted')

        # Macro F1 점수 계산
        macro_f1_value = f1.compute(predictions=predictions, references=labels, average="macro")

        # Micro F1 점수 계산
        micro_f1_value = f1.compute(predictions=predictions, references=labels, average="micro")
        macro_stats = {key: {'count':0,'correct':0} for key in label2idx.keys()}

        for prediction, label in zip(predictions, labels):
        
            macro_stats[idx2label[label]]['count'] += 1
            if prediction == label:
                macro_stats[idx2label[label]]['correct'] += 1
        
 
        #save macro stats as json
        with open('macro_stats.json', 'w') as f:
            f.write(json.dumps(macro_stats))
        
        return {
            "accuracy": accuracy_value,
            "f1": f1_value,
            "macro_f1": macro_f1_value,
            "micro_f1": micro_f1_value
        }

    #labels = list(set(tokenized_train['train']['label']))

    """
    # idx2label 매핑 생성
    print(labels)
    idx2label = {idx: label for idx, label in enumerate(labels)}
    label2idx = {label: idx for idx, label in idx2label.items()}
    """
    print(idx2label)


    #model = nn.DataParallel(model)

    #for train_data_file in ['Test1_original_add_shot_why_verify_identity','Test1_original_add_shot_verify_my_identity','Test1_original_add_shot_both']:
    for train_data_file in ['1']:
        #path = '/data/1_data_server/kkm/TAGSv2/temp/pair_with_controlled_1_with_original.json'
        path = f'temp_data.json'
        #path = f'/data/1_data_server/kkm/TAGSv2/temp_data/{train_data_file}.json'
        train_dataset = load_dataset('json',data_files=path)
        tokenized_train = train_dataset.map(preprocess_function, batched=True)
        train = tokenized_train['train'].to_pandas()  # 예시로 훈련 데이터셋을 사용합니다.
        test = tokenized_test['test'].to_pandas()
        model = AutoModelForSequenceClassification.from_pretrained(
            'roberta-large', num_labels=77, id2label=idx2label, label2id=label2idx
        ).cuda()

        train['label'] = train['label'].map(label2idx)
    
        train = Dataset.from_pandas(train)
        test = Dataset.from_pandas(test)

        training_args = TrainingArguments(
            output_dir="my_awesome_model",
            learning_rate=1e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=250,
            weight_decay=0.01,
            evaluation_strategy="steps",
            eval_steps=0.1,
            do_train = False,
            run_name = test_name,

        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train,
            eval_dataset=test,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,

        )
        trainer.train()

    with open('temp_data.json', 'w') as f:
        json.dump(list_data, f)
        
    return model
def main():
    ct = 0
    while flag == 0:
        run_train()
        ct+=1
        print('try : ')
        print(ct)
if __name__ == "__main__":
    main()