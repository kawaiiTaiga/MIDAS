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
        
    def generation(self,target_class,ideas,description,sub_bar,i,t):
        
        subprogress_postfix = f'generation:||{i}/{t}'
        sub_bar.set_postfix_str(subprogress_postfix)
        sub_bar.total = len(self.original_data[target_class])
        combined_result = []
        datas = self.original_data[target_class]
        
        for idx,d in enumerate(datas):
            ideas_r = random.sample(ideas,10)
            for i in ideas_r:
                
                instruction = f"""[INST]This is sentence from dataset about question type classification.
Reference the given sentence and generate new data for {target_class} to fit the idea presented. Give me five new texts.
Class name : {target_class}
sentence : {d}
idea : {i} 
[/INST] Generated texts :1."""

                input_ids = self.tokenizer(instruction, return_tensors="pt").input_ids.to(self.device)
                outputs = self.model.generate(input_ids, max_new_tokens=100,repetition_penalty = configs['CEA']['generation_repetition_penalty'])
                generated_text = self.tokenizer.decode(outputs[0])
                result = generated_text.split(' Generated texts :')[1]
                results = re.split(r'\d+\.', result)
                results = [item.strip() for item in results if item.strip()]
                combined_result.extend(results)
            #print(result)
                combined_result.extend(results)
                
                sub_bar.n = idx
                sub_bar.refresh()
        return combined_result
    
    def verification(self,candidiate_texts,sub_bar,i,t):
        
        subprogress_postfix = f'verification:||{i}/{t}'
        sub_bar.set_postfix_str(subprogress_postfix)
        sub_bar.total = len(candidiate_texts)
        
        combined_result = []
        for idx,text in enumerate(candidiate_texts):
            #print(candidiate_texts)
            similar_texts = self.SimilarityChecker.find_top_k_similar_texts(text,k=configs['CEA']['verification_count'])
     
            instruction = ""
            for similar_text in similar_texts:
                instruction += f"text : {similar_text[0]} class : {similar_text[1]}\n"
            
            instruction += f"text : {text} class :"
            input_ids = self.tokenizer(instruction, return_tensors="pt").input_ids.to(self.device)
            
            outputs = self.model.generate(input_ids,max_new_tokens=15)
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text.replace(instruction, "")
            generated_text = generated_text.split('\n')[0].strip()
            verification_result = self.SimilarityChecker.find_most_similar_class_by_name(generated_text)
            
            combined_result.append(verification_result)
            
            sub_bar.n = idx
            sub_bar.refresh()
            
        return combined_result
    
    def mutation(self,candidate_texts,verification_result,target_class,sub_bar,i,t):
        combined_result = []
        ct = 0

        subprogress_postfix = f'mutation:||{i}/{t}'
        sub_bar.set_postfix_str(subprogress_postfix)
        sub_bar.total = len(candidate_texts)
        
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
Modify this query text to be suitable for {target_class}.[/INST]Modified text :"'''
                input_ids = self.tokenizer(instruction, return_tensors="pt").input_ids.to(self.device)
                outputs = self.model.generate(input_ids, max_new_tokens=100, repetition_penalty = configs['CEA']['mutation_repetition_penalty'])
                generated_text = self.tokenizer.decode(outputs[0])
                generated_text = generated_text.split(f'Modified text :')[1]
                generated_text = generated_text.split('"')[1].strip()
                combined_result.append(generated_text)

                sub_bar.n = ct
                sub_bar.refresh()

        return combined_result

    def description_generation(self,target_class, sub_bar,i,t):
        
        subprogress_postfix = f'desc_gen:||{i}/{t}'
        sub_bar.set_postfix_str(subprogress_postfix)
        
        sub_bar.total = 1
        sub_bar.n = 0
        sub_bar.refresh()
        
        data_target = self.original_data[target_class]
        
        instruction = f"[INST]This is one of the classes in a classification dataset about question type.\
{target_class}: {data_target}\
Describe this class in one sentence.[/INST] Class description :"

        input_ids = self.tokenizer(instruction, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(input_ids, )
        generated_text = self.tokenizer.decode(outputs[0])
        description = generated_text.split('Class description :')[1]
        
        return description
        

    def idea_generation(self,data,description, sub_bar,i,t):
        
        subprogress_postfix = f'idea_gen:||{i}/{t}'
        sub_bar.set_postfix_str(subprogress_postfix)
        
        
        
        data_target = self.original_data[data]
        sub_bar.total = len(data_target)
        combined_result = []
        for idx,text in enumerate(data_target):
            

            instruction = f"""This is sentence from dataset for question type classification. Considering the class, suggest five ideas that can make the given class more diverse.\
The output format should summarize each idea in one sentence; example sentences are not required.
class name : {data}
class_description : {description} 
Sentence : {text} 
[/INTS] Ideas : 1."""

            input_ids = self.tokenizer(instruction, return_tensors="pt").input_ids.to(self.device)
            outputs = self.model.generate(input_ids, max_new_tokens=500)
            generated_text = self.tokenizer.decode(outputs[0])
            generated_text = generated_text.split(' Ideas :')[1].strip()
            ideas = re.split(r'\d+\.', generated_text)
            ideas = [item.strip() for item in ideas if item.strip()]
            combined_result.extend(ideas)
            
            sub_bar.n = idx
            sub_bar.refresh()
        return combined_result


    def CEA(self,target_classs,i,t):
       
        
        sub_bar = tqdm(total=0,
                   desc=f"gpu {rank}",
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{postfix}]",
                   position=rank)
 
  
        description = self.description_generation(target_classs,sub_bar,i,t)

        ideas = self.idea_generation(target_classs,description,sub_bar,i,t)
        
        ideas_save = [{'class' : target_classs, 'description' : description, 'ideas' : ideas}]
        with open(f'cot/{configs["datasets"]}/{configs["shot"]}shot/ideas/{target_classs}_{configs["test_name"]}.json', 'w') as f:
            json.dump(ideas_save, f,indent=4)

        
        candidate_texts = self.generation(target_classs,ideas,description,sub_bar,i,t)
            
        
        verification_result = self.verification(candidate_texts,sub_bar,i,t)
            
        
        final_result = self.mutation(candidate_texts,verification_result,target_classs,sub_bar,i,t)
            
        final_result = [{'class' : target_classs, 'text' : text, 'pred' : pred } for text,pred in zip(final_result,verification_result)]
        

        with open(f'data/{configs["datasets"]}/{configs["shot"]}shot/SEA/2/{target_classs}_{configs["test_name"]}.json', 'w') as f:
            json.dump(final_result, f,indent=4)
        return final_result
def main():
    

    
    model = GenerationModel("meta-llama/Llama-2-13b-chat-hf", rank)
    df = pd.read_csv(f'/data/2_data_server/nlp-04/lost_technology/original_data/{configs["datasets"]}/train_{configs["shot"]}.csv')
    result = df.groupby('category')['text'].apply(list).to_dict()

    classes = list(result.keys())
    
    #classes = np.array_split(classes, 2)
    parts = np.array_split(classes, world_size)

    results = []



    for idx, datas in enumerate(parts):
        if idx == rank:
            for i, data in enumerate(datas):
                CEA_RESULT = model.CEA(data,i,len(datas))
                #results.extend({str(data): cot_result})

                
      

    dist.barrier()


if __name__ == '__main__':
    dist.init_process_group(backend='nccl',timeout=datetime.timedelta(hours=5))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    main()