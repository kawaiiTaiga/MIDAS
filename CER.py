from transformers import AutoModelForCausalLM, LlamaTokenizer
import torch
import utils
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import pandas as pd
import json
import itertools
import numpy as np
from tqdm import tqdm
import time
from threading import Thread
import os
from utils import * 

with open('config.json', 'r') as f:
    configs = json.load(f)




class GenerationModel:
    def __init__(self, model_name, gpu):
 
        self.device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
        self.model = model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-chat-hf",device_map=self.device,  torch_dtype='auto')
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        df = pd.read_csv(f'/data/2_data_server/nlp-04/lost_technology/original_data/{configs["datasets"]}/train_{configs["shot"]}.csv')
        self.original_data = df.groupby('category')['text'].apply(list).to_dict()
    

    def generate(self,data):

        
        class_a = data[0]
        class_b = data[1]
        data_a = self.original_data[class_a]
        data_b = self.original_data[class_b]

        instruction = f"[INST]This is part of dataset for intent classfication for banking data.\
            {class_a} : {data_a}\
            {class_b} : {data_b}\
            Tell me the main difference between {class_a} and {class_b} in one sentence.[/INST] The main difference between {class_a} and {class_b} is"

        input_ids = self.tokenizer(instruction, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(input_ids, )
        generated_text = self.tokenizer.decode(outputs[0], repetition_penalty = 1.15)
        COT = generated_text.split('[/INST]')[1]
        return COT
def main():
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model = GenerationModel("meta-llama/Llama-2-13b-chat-hf", rank)
    df = pd.read_csv(f'/data/2_data_server/nlp-04/lost_technology/original_data/{configs["datasets"]}/train_{configs["shot"]}.csv')
    result = df.groupby('category')['text'].apply(list).to_dict()

    keys = list(result.keys())
    combinations = list(itertools.combinations(keys, 2))

    parts = np.array_split(combinations, world_size)

    results = []
    
    if rank == 0:
        initialize_progress_data(world_size)
        # 주 프로세스에서 모니터링 스레드 시작
        monitor_progress_thread(world_size, parts)



    for idx, datas in enumerate(parts):
        if idx == rank:
            for i, data in enumerate(datas):
                cot_result = model.generate(data)
                results.append({str(data): cot_result})
                update_progress(rank, i, len(datas))
                 
    with open(f'cot/{configs["datasets"]}/{configs["shot"]}shot/data_{rank}.json', 'w') as f:
        json.dump(results, f,indent=4)
    

    dist.barrier()



    combined_data = []
    if rank == 0:
        for i in range(world_size):
            with open(f'cot/{configs["datasets"]}/{configs["shot"]}shot/data_{i}.json', 'r') as f:
                data = json.load(f)
                combined_data.extend(data)
        with open(f'cot/{configs["datasets"]}/{configs["shot"]}shot/combined.json', 'w') as f:
            json.dump(combined_data, f,indent=4)
if __name__ == '__main__':
    main()