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
        
    def generation(self,target_class,ambigious_class):
        
        combined_result = []
        for i in range(configs['CEA']['generation_count']):
   
            
            data_target = '","'.join(random.sample(self.original_data[target_class],configs['CEA']['sample_count']))
            data_ambigious = '","'.join(random.sample(self.original_data[ambigious_class],configs['CEA']['sample_count']))

            Key = tuple(sorted([target_class,ambigious_class]))
            CER = self.CER[Key]

            instruction = f"""[INST]I have an intent classification dataset for banking tasks with two classes, {target_class} and {ambigious_class}.\
{target_class} : "{data_target}"\
{ambigious_class} : "{data_ambigious}"\
key differentiator : {CER}\
key differentiator outlining the primary distinctions between {target_class} and {ambigious_class}.
I now require 10 new text data examples specifically for class {target_class}.\
These examples should be deliberately crafted to ensure that, when trained in a classification model, class {target_class} is distinctly identifiable from class {ambigious_class}.\
Could you generate these examples for me? [\INST] Examples : 1."""

            input_ids = self.tokenizer(instruction, return_tensors="pt").input_ids.to(self.device)
            outputs = self.model.generate(input_ids, )
            generated_text = self.tokenizer.decode(outputs[0], repetition_penalty = configs['CEA']['generation_repetition_penalty'])
            result = generated_text.split('Examples : ')[1]
            candidiate_texts = re.split(r'\d+\.', result)
            candidiate_texts = [item.strip() for item in candidiate_texts if item.strip()]
            combined_result.extend(candidiate_texts)
            
        return combined_result
    
    def verification(self,candidiate_texts):
        combined_result = []
        for text in candidiate_texts:

            similar_texts = self.SimilarityChecker.find_top_k_similar_texts(text,k=configs['CEA']['verification_count'])
     
            instruction = ""
            for similar_text in similar_texts:
                instruction += f"text : {similar_text[0]} class : {similar_text[1]}\n"
            
            instruction += f"text : {text} class :"
            input_ids = self.tokenizer(instruction, return_tensors="pt").input_ids.to(self.device)
            outputs = self.model.generate(input_ids,max_new_tokens=30)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text.replace(instruction, "")
            generated_text = generated_text.split('\n')[0].strip()
            verification_result = self.SimilarityChecker.find_most_similar_class_by_name(generated_text)
            combined_result.append(verification_result)
            
        return combined_result
    
    def mutation(self,candidate_texts,verification_result,target_class):
        combined_result = []
        for text,pred in zip(candidate_texts,verification_result):
            if pred == target_class:
                combined_result.append(text)
            else:
                Key = tuple(sorted([target_class,pred]))
                CER = self.CER[Key]

                data_target = '","'.join(self.original_data[target_class])

                instruction = f"""[INST]Input text: {text}\
This sentence currently belongs to the '{pred}'. Transform this sentence into one that would belong to the '{target_class}', taking into consideration the provided key differentiator and sample texts of {target_class}.\
Key differentiator : {CER}\
Sample texts of {target_class}:{data_target}[\INTS] Transformed text:"""
                input_ids = self.tokenizer(instruction, return_tensors="pt").input_ids.to(self.device)
                outputs = self.model.generate(input_ids, )
                generated_text = self.tokenizer.decode(outputs[0], repetition_penalty = configs['CEA']['mutation_repetition_penalty'])
                generated_text = generated_text.split('Transformed text:')[1]
                combined_result.append(generated_text)
                

        return combined_result



    def CEA(self,data):
       
        ambigious_classes = self.similarity_ranking[data]
        combined_data = []
        for idx,ambigious_class in enumerate(ambigious_classes):

            

            update_sub_progress(rank, idx, len(ambigious_classes),'generation')
            candidate_texts = self.generation(data,ambigious_class[0])
            
            update_sub_progress(rank, idx, len(ambigious_classes),'verification')
            verification_result = self.verification(candidate_texts)
            
            final_result = self.mutation(candidate_texts,verification_result,data)
            update_sub_progress(rank, idx, len(ambigious_classes),'verification')
            final_result = [{'class' : data, 'text' : text, 'ambig' : ambigious_class, 'pred' : pred } for text,pred in zip(final_result,verification_result)]
            combined_data.extend(final_result)
            
        #save as json file
        with open(f'data/{configs["datasets"]}/{configs["shot"]}shot/{data}_{configs["test_name"]}.json', 'w') as f:
            json.dump(combined_data, f,indent=4)
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
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    main()