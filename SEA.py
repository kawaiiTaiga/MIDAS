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
        
    def generation(self,target_class,ideas,description):
        
        combined_result = []
        datas = self.original_data[target_class]
        for idx,d in enumerate(datas):
            for i in ideas:
                
                instruction = f""""[INST]{target_class} description: {description}\
a sample data from this class: {d} \
ideas for mutating this class: {i} 
Generate three mutated version of the sample data. The enrichment should incorporate the specific ideas provided, adding more depth and diversity to the data, \
while staying true to the original class description.[/INST] Mutated texts :1."""

                input_ids = self.tokenizer(instruction, return_tensors="pt").input_ids.to(self.device)
                outputs = self.model.generate(input_ids, max_new_tokens=100,repetition_penalty = configs['CEA']['generation_repetition_penalty'])
                generated_text = self.tokenizer.decode(outputs[0])
                result = generated_text.split(' Mutated texts :')[1]
                results = re.split(r'\d+\.', result)
                results = [item.strip() for item in results if item.strip()]
                combined_result.extend(results)
            #print(result)
                combined_result.extend(results)
            update_sub_progress(rank, idx, len(datas),'generation',1,1)
        return combined_result
    
    def verification(self,candidiate_texts):
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
            update_sub_progress(rank, idx, len(candidiate_texts),'verification',1,1)
        return combined_result
    
    def mutation(self,candidate_texts,verification_result,target_class):
        combined_result = []
        ct = 0

        for text,pred in zip(candidate_texts,verification_result):
            ct +=1 
            if pred == target_class:
                combined_result.append(text)
            else:
                Key = tuple(sorted([target_class,pred]))
                CER = self.CER[Key]

                data_target = "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(self.original_data[target_class]))

                instruction = f'''[INST]Input text: {text}\
This sentence currently belongs to the '{pred}'. Transform this sentence into one that would belong to the '{target_class}', taking into consideration the provided key differentiator and sample texts of {target_class}.\
Key differentiator : {CER}\
After generating the transformed text, stop generating new text.
Sample texts of {target_class}:{data_target}[/INST] Transformed text for class {target_class}:"'''
                input_ids = self.tokenizer(instruction, return_tensors="pt").input_ids.to(self.device)
                outputs = self.model.generate(input_ids, max_new_tokens=100, repetition_penalty = configs['CEA']['mutation_repetition_penalty'])
                generated_text = self.tokenizer.decode(outputs[0])
                generated_text = generated_text.split(f'Transformed text for class {target_class}:')[1]
                generated_text = generated_text.split('"')[1].strip()
                combined_result.append(generated_text)
            update_sub_progress(rank, ct, len(candidate_texts),'mutation',1,1)

        return combined_result

    def description_generation(self,target_class):
        
        
        data_target = self.original_data[target_class]
        
        instruction = f"[INST]This is one of the classes in a classification dataset related to banking tasks.\
{target_class}: {data_target}\
Describe this class in one sentence.[/INST] Class description :"

        input_ids = self.tokenizer(instruction, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(input_ids, )
        generated_text = self.tokenizer.decode(outputs[0])
        description = generated_text.split('Class description :')[1]

        return description
        

    def idea_generation(self,data,description):
        data_target = self.original_data[data]
        combined_result = []
        for idx,text in enumerate(data_target):
            instruction = f"This is a text from a classification dataset related to banking tasks.\
{data}: {data_target}\
class_description:{description}\
Provide 5 ideas to enrich this class further. Numbering the generated text\
[/INTS] Generated ideas :1."
            input_ids = self.tokenizer(instruction, return_tensors="pt").input_ids.to(self.device)
            outputs = self.model.generate(input_ids, max_new_tokens=500)
            generated_text = self.tokenizer.decode(outputs[0])
            generated_text = generated_text.split('Generated ideas :')[1].strip()
            ideas = re.split(r'\d+\.', generated_text)
            ideas = [item.strip() for item in ideas if item.strip()]
            combined_result.extend(ideas)
            update_sub_progress(rank, idx, len(data_target),'idea_gen',1,1)
        return combined_result


    def CEA(self,target_classs):
       
        sub_bar = tqdm(total=0,
               desc=f"gpu {rank}",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{postfix}]",
               position=rank)

        update_sub_progress(rank, 0, 0,'desc_gen',1,1)
        description = self.description_generation(target_classs)

        ideas = self.idea_generation(target_classs,description)
        ideas_save = [{'class' : target_classs, 'description' : description, 'ideas' : ideas}]
        with open(f'cot/{configs["datasets"]}/{configs["shot"]}shot/ideas/{target_classs}_{configs["test_name"]}.json', 'w') as f:
            json.dump(ideas_save, f,indent=4)

        
        candidate_texts = self.generation(target_classs,ideas,description)
            
        
        verification_result = self.verification(candidate_texts)
            
        
        final_result = self.mutation(candidate_texts,verification_result,target_classs)
            
        final_result = [{'class' : target_classs, 'text' : text, 'pred' : pred } for text,pred in zip(final_result,verification_result)]
        
            
        #save as json file
        with open(f'data/{configs["datasets"]}/{configs["shot"]}shot/SEA/{target_classs}_{configs["test_name"]}.json', 'w') as f:
            json.dump(final_result, f,indent=4)
        return final_result
def main():
    

        
    
    model = GenerationModel("meta-llama/Llama-2-13b-chat-hf", rank)
    df = pd.read_csv(f'/data/2_data_server/nlp-04/lost_technology/original_data/{configs["datasets"]}/train_{configs["shot"]}.csv')
    result = df.groupby('category')['text'].apply(list).to_dict()

    classes = list(result.keys())

    parts = np.array_split(classes, world_size)

    results = []
    
    if rank == 0:
        initialize_progress_data(world_size)
        # 주 프로세스에서 모니터링 스레드 시작
        monitor_progress_thread(world_size, parts,True)



    for idx, datas in enumerate(parts):
        if idx == rank:
            for i, data in enumerate(datas):
                update_progress(rank, i, len(datas))
                CEA_RESULT = model.CEA(data)
                #results.extend({str(data): cot_result})

                
                 
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
    dist.init_process_group(backend='nccl',timeout=datetime.timedelta(minutes=30))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    main()