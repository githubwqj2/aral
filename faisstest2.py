import os
from functools import partial
from typing import List

import faiss
import numpy as np
import pandas
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel

simberttokenizer = AutoTokenizer.from_pretrained("./simbert")
simbertencoder = AutoModel.from_pretrained("./simbert")
simbertencoder.eval()
dim, index_param = 768, 'Flat'
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
simbertencoder.to(device)


class MyDataSet(Dataset):
    def __init__(self, path):
        self.data = pandas.read_json(path, lines=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data['question'][idx], self.data['question'].index[idx])


def collateFun(batch_idx_tuple, tokenizer):
    batch, idx = zip(*batch_idx_tuple)
    batch_encode = tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch, padding='longest',
                                               add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                                               max_length=512,  # 填充 & 截断长度
                                               truncation=True, )

    return torch.tensor(batch_encode['input_ids']) \
        , torch.tensor(batch_encode['token_type_ids']) \
        , torch.tensor(batch_encode['attention_mask']) \
        , idx, batch


partial_collateFun = partial(collateFun, tokenizer=simberttokenizer)

train_data = MyDataSet('./input/medicine.jsonl')

train_dataloader = Data.DataLoader(
    train_data, shuffle=True, collate_fn=partial_collateFun, batch_size=batch_size
)


#一方面是 L2 Normalize之后， 欧式距离与余弦相似度 可以互相转化
#此外：可以消除输入向量的尺度差异，使得模型在计算相似度时不受到向量长度的影响。
#   归一化后的向量可以被视为单位向量，避免某些特征在计算相似度时占据主导地位，从而保持更好的特征平衡。


# 1. 双塔模型通常用于计算相似度或距离度量，例如在推荐系统中用于计算用户和物品之间的相似度。模型的最后一层是通过计算内积（或点积）来度量两个输入向量之间的相似度。
# 2. 在计算内积之前，通过对输入向量进行L2 Norm操作，可以将向量的范数（长度）归一化为1。这样做的好处是，可以消除输入向量的尺度差异，使得模型在计算相似度时不受到向量长度的影响。
# 3. 归一化后的向量可以被视为单位向量，其方向（角度）对于相似度计算更加重要。这样做有助于模型更加关注特征向量之间的相对关系，而不受向量的绝对尺度影响。
# 4. 此外，L2 Norm操作还可以对输入向量进行标准化，使得每个维度的取值范围都接近相同。这有助于避免某些特征在计算相似度时占据主导地位，从而保持更好的特征平衡。


def l2_norm(vecs):
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def settingIndex(dim, index_param):
    """
      设置faiss的index,到目前还没有添加数据的
      """
    if index_param[0:4] == 'HNSW' and ',' not in index_param:
        hnsw_num = int(index_param.split('HNSW')[-1])
        print(f'Index维度为{dim}，HNSW参数为{hnsw_num}')
        index = faiss.IndexHNSWFlat(dim, hnsw_num, faiss.METRIC_INNER_PRODUCT)
    else:
        quantizer = faiss.IndexFlatL2(dim)  # 欧式距离 判断落入那个分区
        # quantizer = faiss.IndexFlatIP(d)    # 点乘
        nlist = 2  # 将数据集向量分为10个维诺空间
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        # index = faiss.index_factory(dim, index_param, faiss.METRIC_INNER_PRODUCT)
    index.verbose = True
    index.do_polysemous_training = False
    return index


#保存索引
def dumpIndex(index, index_save_path,row_data, row_data_path):
    """
    保存index索引
    """

    faiss.write_index(index, index_save_path)
    row_data.to_json(row_data_path,orient='records',lines='orient', force_ascii=False)


#添加并保存索引
def addIndex(tag_data_loader, encoder, myfaiss):
    if  os.path.exists('./faiss_index/faiss_index_saved'):
        return

    """
    根据文本数据得到768维向量
    """

    rowdata, rowindex = [], []
    for step, batch_data in enumerate(tag_data_loader):
        input_ids, token_type_ids, attention_mask, sentense_ids, rowd = batch_data[0], batch_data[1], batch_data[
            2], batch_data[3], batch_data[4]
        #将原始数据累加进行保存
        rowdata.extend(rowd)
        rowindex.extend(sentense_ids)
        #进行预测
        output = encoder(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                         token_type_ids=token_type_ids.to(device))
        outdata=output['pooler_output'].detach().cpu().numpy()
        myfaiss.train(l2_norm(outdata))
        myfaiss.add_with_ids(l2_norm(outdata), np.array(sentense_ids))
    #组装pandas的df
    df = {"sentense": rowdata, "id": rowindex}
    df = pandas.core.frame.DataFrame(df)

    dumpIndex(myfaiss, './faiss_index/faiss_index_saved',df, './faiss_index/rawdata_index_saved')


def get_simple_vec(batch_data, tokenizer, encoder):
    """
    根据文本数据得到768维向量
    """
    input_ids, token_type_ids, attention_mask, _, _ = collateFun([(batch_data, [0])], tokenizer)
    print(input_ids)
    print(attention_mask)
    print(token_type_ids)
    output = encoder(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                     token_type_ids=token_type_ids.to(device))
    return l2_norm(output['pooler_output'].detach().cpu().numpy())



# ids = settingIndex(dim, index_param)
# addIndex(train_dataloader, simbertencoder, ids)
#进行组装，便于取数
index_sentense = {}
dataf_index = pandas.read_json('./faiss_index/rawdata_index_saved', lines=True)
for index, row in dataf_index.iterrows():
    index_sentense[row[1]] = row[0]
ids=faiss.read_index('./faiss_index/faiss_index_saved')

def search_vec(key_word, ids, tokenizer, topK=10):
    target_vecs = get_simple_vec(key_word, tokenizer, simbertencoder)
    print(target_vecs)
    C, I = ids.search(target_vecs, topK)  # C 分数 I index
    index_sentense.get(I[0][0])
    return index_sentense.get(I[0][0])



if __name__ == '__main__':

    search_word = '我这是前列腺还是肾有问题 婚前有过性交,经常,婚后房事时一次也就1分钟'
    recall_doc=search_vec(search_word, ids, simberttokenizer)
