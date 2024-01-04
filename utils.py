

from threading import Thread, Event
import os
from tqdm import tqdm
import time
import torch
import torch.distributed as dist
import itertools
from transformers import AutoModel, AutoTokenizer
import numpy as np
import socket
import multiprocessing as mp


# 현재 서버의 호스트 이름을 가져옴
hostname = socket.gethostname()

"""
 # Manager 객체 생성
progress_manager = mp.Manager()
subprogress_manager = mp.Manager()

# 공유 메모리로 사용할 딕셔너리 생성
progress_data = progress_manager.dict()
subprogress_data = subprogress_manager.dict()
def initialize_progress_data(world_size):
    for i in range(world_size):
        progress_data[i] = {"progress": 0, "total": 0, "status": '', "upper": 0, "upper_idx": 0}
def read_sub_progress_data(rank):
    return subprogress_data.get(rank)

def read_progress_data(rank):
    return progress_data.get(rank)

def update_progress(rank, current, total, server_name=''):
    progress_data[rank] = {"progress": current, "total": total}

def update_sub_progress(rank, current, total, status, upper_level, upper_level_idx):
    subprogress_data[rank] = {"progress": current, "total": total, "status": status, "upper": upper_level, "upper_idx": upper_level_idx}

def colorize(text, color):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "endc": "\033[0m"
    }
    return colors.get(color, "") + text + colors["endc"]

def sub_monitor_progress(world_size, parts):
    print('d')
    bar_colors = ["red", "green", "yellow", "blue", "magenta", "cyan", "white"]
    progress_bars = []
    subprogress_bars = []
    # 메인 프로세스 진행 막대 생성
    for i, part in enumerate(parts):
        main_bar = tqdm(total=len(part),
                        desc=colorize(f"Process {i}", bar_colors[i % len(bar_colors)]),
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        progress_bars.append(main_bar)
        # 서브프로세스 진행 막대 생성 (초기 total 값은 나중에 설정)
        sub_bar = tqdm(total=0,
                       desc=colorize(f"Subprocess of {i}", "white"),
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                       position=len(progress_bars) + i)
        subprogress_bars.append(sub_bar)
    completed = [False] * world_size
    while not all(completed):
        for i in range(world_size):
            progress_data = read_progress_data(i)
            if progress_data is not None:
                progress_bars[i].n = progress_data["progress"]
                progress_bars[i].refresh()
                if progress_data["progress"] + 2 >= len(parts[i]):
                    completed[i] = True
            # 서브프로세스 상태 및 상태 문자열 업데이트
            
            subprogress_data = read_sub_progress_data(i)
            if subprogress_data is not None:

                subprogress_postfix = f'{subprogress_data["status"]} : {subprogress_data["upper_idx"]} / {subprogress_data["upper"]}'
                subprogress_bars[i].total = subprogress_data["total"]
                subprogress_bars[i].n = subprogress_data["progress"]
                subprogress_bars[i].set_postfix_str(subprogress_postfix)
                subprogress_bars[i].refresh()
            else:
                completed[i] = False
        time.sleep(0.5)
    for bar in progress_bars:
        bar.close()
    for bar in subprogress_bars:
        bar.close()

def monitor_progress(world_size, parts):
    try:
        bar_colors = ["red", "green", "yellow", "blue", "magenta", "cyan", "white"]
        progress_bars = [
            tqdm(total=len(part), 
                 desc=colorize(f"Process {i}", bar_colors[i % len(bar_colors)]),
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
            for i, part in enumerate(parts)
        ]
        completed = [False] * world_size
        while not all(completed):
            for i in range(world_size):
                progress_data = read_progress_data(i)
                if progress_data is not None:
                    progress_bars[i].n = progress_data["progress"]
                    progress_bars[i].refresh()
                    if progress_data["progress"] + 2 >= len(parts[i]):
                        completed[i] = True
                else:
                    completed[i] = False
            time.sleep(0.5)  # 업데이트 주기
        for bar in progress_bars:
            bar.close()
    except Exception as e:
        print(f"Error in monitor thread: {e}")
    
def monitor_progress_thread(world_size, parts,sub,server=''):
    if sub==False:
        monitor_thread = Thread(target=monitor_progress, args=(world_size, parts))
    else:
        monitor_thread = Thread(target=sub_monitor_progress, args=(world_size, parts))
    monitor_thread.start()
"""
def json_tuple_change(data):
    new_data = {}
    for item in data:
        for key, value in item.items():
            # 키를 세트로 변환
            key = key.replace("\n", "")
            key_tuple = sorted(key.strip("[]'").split("' '"))
            key_tuple = tuple(key_tuple)
            new_data[key_tuple] = value
    return new_data


class SimilarityChecker:
    def __init__(self, data, device,model_name="sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device =device
        self.model = AutoModel.from_pretrained(model_name,device_map=device,torch_dtype="auto")
        self.embeddings = []
        self.texts = []
        self.classes = []
        self.class_embeddings = {}  # 클래스 이름의 임베딩을 저장할 딕셔너리
        self.compute_embeddings(data)


    def compute_embeddings(self, data):
        """입력된 모든 텍스트와 클래스 이름에 대한 임베딩을 계산하고 저장합니다."""
        for cls, texts in data.items():
            # 클래스 이름의 임베딩을 계산하고 저장
            class_embedding = self.get_embedding(cls)
            self.class_embeddings[cls] = class_embedding

            for text in texts:
                self.embeddings.append(self.get_embedding(text))
                self.texts.append(text)
                self.classes.append(cls)
    def get_embedding(self, text):
        """텍스트의 임베딩을 계산합니다."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    @staticmethod
    def cosine_similarity(a, b):
        """코사인 유사도를 계산합니다."""
        return (a @ b.T) / (a.norm() * b.norm())

    def find_most_similar_class(self, input_text):
        """주어진 텍스트와 가장 유사한 클래스를 찾습니다."""
        input_embedding = self.get_embedding(input_text)
        class_similarities = {cls: [] for cls in set(self.classes)}

        for cls, emb in zip(self.classes, self.embeddings):
            similarity = self.cosine_similarity(input_embedding, emb).item()
            class_similarities[cls].append(similarity)

        average_similarities = {cls: np.mean(similarities) for cls, similarities in class_similarities.items()}
        most_similar_class = max(average_similarities, key=average_similarities.get)

        return most_similar_class, average_similarities[most_similar_class]

    def calculate_class_similarities(self):
        """클래스 간 유사도를 계산합니다."""
        unique_classes = set(self.classes)
        class_similarities = {cls: {other_cls: [] for other_cls in unique_classes if other_cls != cls} for cls in unique_classes}

        for i, (class1, emb1) in enumerate(zip(self.classes, self.embeddings)):
            for j, (class2, emb2) in enumerate(zip(self.classes, self.embeddings)):
                if i != j and class1 != class2:
                    similarity = self.cosine_similarity(emb1, emb2).item()
                    class_similarities[class1][class2].append(similarity)

        # 평균 유사도 계산
        average_similarities = {cls: {other_cls: np.mean(similarities) for other_cls, similarities in cls_sim.items()} for cls, cls_sim in class_similarities.items()}

        return average_similarities
    def find_top_k_similar_classes(self, k):
        """각 클래스별 상위 k개의 가장 유사한 클래스를 찾습니다."""
        class_similarities = self.calculate_class_similarities()
        top_k_classes = {}
        for cls, similarities in class_similarities.items():
            # 유사도에 따라 정렬하고 상위 k개 선택
            sorted_classes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
            top_k_classes[cls] = sorted_classes

        return top_k_classes
    def find_most_similar_class_by_name(self, new_class_name):
        """새로운 형식의 클래스 이름에 대해 가장 유사한 클래스를 찾습니다."""
        # 새로운 클래스 이름이 이미 존재하는 경우, 기존의 임베딩 사용
        new_class_name = new_class_name.strip()
        if new_class_name in self.class_embeddings:
            new_class_embedding = self.class_embeddings[new_class_name]
            return new_class_name
        else:
            # 새로운 클래스 이름에 대한 임베딩 계산
            new_class_embedding = self.get_embedding(new_class_name)

        max_similarity = -1
        most_similar_class = None

        # 기존 클래스 임베딩과 비교
        for existing_class, emb in self.class_embeddings.items():
            if new_class_name != existing_class:
                similarity = self.cosine_similarity(new_class_embedding, emb).item()
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_class = existing_class

        return most_similar_class

    def find_top_k_similar_texts(self, input_text, k):
        """주어진 텍스트와 가장 유사한 텍스트 k개와 그 텍스트들의 클래스를 찾습니다."""

        input_embedding = self.get_embedding(input_text)
        similarities = []

        for text, emb, cls in zip(self.texts, self.embeddings, self.classes):
            similarity = self.cosine_similarity(input_embedding, emb).item()
            similarities.append((text, similarity, cls))



        # 유사도에 따라 정렬하고 상위 k개 선택
        top_k_texts = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

        return [(text, cls, similarity) for text, similarity, cls in top_k_texts]