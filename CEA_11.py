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
import re
import random
import datetime


with open('config.json', 'r') as f:
    configs = json.load(f)


class GenerationModel:
    def __init__(self, model_name, gpu):
 
        self.device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
        self.model = model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-chat-hf",device_map=self.device,  torch_dtype='auto')
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        df = pd.read_csv(f'original_data/{configs["datasets"]}/train_{configs["shot"]}.csv')
        self.original_data = df.groupby('category')['text'].apply(list).to_dict()
        with open(F'cot/{configs["datasets"]}/{configs["shot"]}shot/combined.json', 'r') as file:
            self.CER = json_tuple_change(json.load(file))
        self.SimilarityChecker = SimilarityChecker(self.original_data,self.device)
        self.similarity_ranking = self.SimilarityChecker.find_top_k_similar_classes(k=configs['CEA']['pair_count'])
        
    def generation(self,target_class,ambigious_class,total_length,idx,subprogress_bars,i,t):
        subprogress_postfix = f'generation:{idx}/{total_length}||{i}/{t}'
        subprogress_bars.set_postfix_str(subprogress_postfix)
        combined_result = []
        for i in range(configs['CEA']['generation_count']):
            
            sampled_data = random.sample(self.original_data[target_class], configs['CEA']['sample_count'])
            data_target = "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(sampled_data))
            
            sampled_ambigious = random.sample(self.original_data[ambigious_class], configs['CEA']['sample_count'])
            data_ambigious = "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(sampled_ambigious))

            Key = tuple(sorted([target_class,ambigious_class]))
            CER = self.CER[Key]

            instruction = f"""[INST]{target_class} : {data_target}
{ambigious_class} : {data_ambigious}
Distinctive Text : {CER}
This is classification dataset about question type. Based on the provided texts for each classes and the distinctive text highlighting their differences, generate five new texts that emphasize the unique characteristics of class {target_class}. 
Generate texts for fit in {target_class} class. Number each text generation, and after completing all, append "[END]".[/INST] Generated texts : 1."""

            input_ids = self.tokenizer(instruction, return_tensors="pt").input_ids.to(self.device)
            outputs = self.model.generate(input_ids, repetition_penalty = configs['CEA']['generation_repetition_penalty'] )
            generated_text = self.tokenizer.decode(outputs[0])
            result = generated_text.split(f'Generated texts : ')[1]
            if '[END]' in result:
                result = result.split('[END]')[0]
            candidiate_texts = re.split(r'\d+\.', result)
            candidiate_texts = [item.strip() for item in candidiate_texts if item.strip()]
            combined_result.extend(candidiate_texts)
            
            
            subprogress_bars.total = configs['CEA']['generation_count']
            subprogress_bars.n = i
            subprogress_bars.refresh()

        return combined_result
    
    def verification(self,candidiate_texts,total_length,idx,subprogress_bars,i,t):
        combined_result = []
        
        subprogress_postfix = f'verification:{idx}/{total_length}||{i}/{t}'
        subprogress_bars.set_postfix_str(subprogress_postfix)
        
        for i,text in enumerate(candidiate_texts):
            
            similar_texts = self.SimilarityChecker.find_top_k_similar_texts(text,k=configs['CEA']['verification_count'])
     
            instruction = ""
            for similar_text in similar_texts:
                instruction += f"Text : {similar_text[0]} Class : {similar_text[1]}\n"
            
            instruction += f"Text : {text} Class :"
            input_ids = self.tokenizer(instruction, return_tensors="pt").input_ids.to(self.device)
            
            outputs = self.model.generate(input_ids,max_new_tokens=15)
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text.replace(instruction, "")
            generated_text = generated_text.split('\n')[0].strip()
            verification_result = self.SimilarityChecker.find_most_similar_class_by_name(generated_text)
            
            combined_result.append(verification_result)
            
            
            subprogress_bars.total = len(candidiate_texts)
            subprogress_bars.n = i 
            subprogress_bars.refresh()
            
        return combined_result
    
    def mutation(self,candidate_texts,verification_result,target_class,total_length,idx,subprogress_bars,i,t):
        combined_result = []
        ct = 0
        
        subprogress_postfix = f'mutation:{idx}/{total_length}||{i}/{t}'
        subprogress_bars.set_postfix_str(subprogress_postfix)
        
        for text,pred in zip(candidate_texts,verification_result):
            ct +=1
            if pred == target_class:
                combined_result.append(text)
            else:
                Key = tuple(sorted([target_class,pred]))
                CER = self.CER[Key]

                data_target = "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(self.original_data[target_class]))

                instruction = f'''[INST]{target_class} :{self.original_data[target_class]}
Distinctive Text : {CER}.   
This is query text which is belong to class {pred}. 
Query text : '{text}'
Modify this query text to be suitable for {target_class}.
[/INST]Modified text :"'''
                input_ids = self.tokenizer(instruction, return_tensors="pt").input_ids.to(self.device)
                outputs = self.model.generate(input_ids, max_new_tokens=100, repetition_penalty = configs['CEA']['mutation_repetition_penalty'])
                generated_text = self.tokenizer.decode(outputs[0])
                generated_text = generated_text.split(f'Modified text :')[1]
                generated_text = generated_text.split('"')[1].strip()
                combined_result.append(generated_text)
            
            subprogress_bars.total = len(candidate_texts)
            subprogress_bars.n = ct
            subprogress_bars.refresh()

        return combined_result



    def CEA(self,data,i,t):
        ambigious_classes = self.similarity_ranking[data]
        combined_data = []
        
        
        # 서브프로세스 진행 막대 생성 (초기 total 값은 나중에 설정)
        sub_bar = tqdm(total=0,
                   desc=f"gpu {rank}",
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{postfix}]",
                   position=rank)
        subprogress_bars = sub_bar
        
        
        for idx,ambigious_class in enumerate(ambigious_classes):

            total_length = len(ambigious_classes)

            
            candidate_texts = self.generation(data,ambigious_class[0],total_length,idx,subprogress_bars,i,t)
            
            
            verification_result = self.verification(candidate_texts,total_length,idx,subprogress_bars,i,t)
            
            final_result = self.mutation(candidate_texts,verification_result,data,total_length,idx,subprogress_bars,i,t)
            
            final_result = [{'class' : data, 'text' : text, 'ambig' : ambigious_class[0], 'pred' : pred } for text,pred in zip(final_result,verification_result)]
            combined_data.extend(final_result)
            
        #save as json file
        with open(f'data/{configs["datasets"]}/{configs["shot"]}shot/CEA/2/{data}_{configs["test_name"]}.json', 'w') as f:
            json.dump(combined_data, f,indent=4)
        return final_result
def main():
    
   
    
    model = GenerationModel("meta-llama/Llama-2-13b-chat-hf", rank)
    df = pd.read_csv(f'/data/2_data_server/nlp-04/lost_technology/original_data/{configs["datasets"]}/train_{configs["shot"]}.csv')
    result = df.groupby('category')['text'].apply(list).to_dict()

    classes = list(result.keys())
    
    #classes = np.array_split(classes,3)
    parts = np.array_split(classes, world_size)

    results = []
    




    for idx, datas in enumerate(parts):
        if idx == rank:
            for i, data in enumerate(datas):
                CEA_RESULT = model.CEA(data,i,len(datas))
                #results.extend({str(data): cot_result})

                
                 
    with open(f'cot/{configs["datasets"]}/{configs["shot"]}shot/data_{rank}.json', 'w') as f:
        json.dump(results, f,indent=4)
    

    dist.barrier()



    
if __name__ == '__main__':
    dist.init_process_group(backend='nccl',timeout=datetime.timedelta(hours=5))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
     # Manager 객체 생성
    
    main()