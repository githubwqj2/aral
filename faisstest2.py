import numpy as np
import faiss
import torch
from transformers import AutoTokenizer,AutoModel, AutoModelForMaskedLM

simberttokenizer = AutoTokenizer.from_pretrained("WangZeJun/simbert-base-chinese")
simbertencoder=AutoModel.from_pretrained("WangZeJun/simbert-base-chinese")




class MYDataSet:
    def __int__(self,data):
        self.data=data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

def collate(batch):
    return torch.FloatTensor(batch)
