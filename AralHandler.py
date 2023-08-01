import os

import faiss
import pandas
import torch
from transformers import AutoTokenizer, AutoModel
from ts.torch_handler.base_handler import BaseHandler



""" 如果是自定义的模型 需要加上 --model-file 参数（这里面的类继承了nn.moudle）  如果添加了其他类 --include_modules
torch-model-archiver --model-name aral --version 1.0 --export-path ./ --serialized-file ./simbert/pytorch_model.bin --handler ./AralHandler.py --extra-files "./index_to_name.json,./simbert/vocab.txt,./simbert/config.json,./faiss_index/rawdata_index_saved,./faiss_index/faiss_index_saved"
"""

#torchserve --start --ncs --model-store ./ --models aral.mar
#torchserve --stop

# curl http://127.0.0.1:8081/models

# curl http://127.0.0.1:8080/predictions/aral -T ./testdata
class MyHandler(BaseHandler):

    def __init__(self):
        super(MyHandler, self).__init__()
        self.initialized = False

    def initialize(self, context):
        self.manifest = context.manifest
        """ 这是manifest内容
        {'createdOn': '01/08/2023 14:30:46', 'runtime': 'python', 
        'model': {'modelName': 'aral', 'serializedFile': 'pytorch_model.bin', 'handler': 'AralHandler.py', 'modelVersion': '1.0'}, 
        'archiverVersion': '0.8.1'}
        """
        properties = context.system_properties
        """  这是properties内容
        {'createdOn': '01/08/2023 14:30:46', 'runtime': 'python', 
        'model': {'modelName': 'aral', 'serializedFile': 'pytorch_model.bin', 'handler': 'AralHandler.py', 'modelVersion': '1.0'}, 
        'archiverVersion': '0.8.1'}
        """
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        #所有的文件，不管你直接引入的时候有多少层，打包之后 都到了model_dir的文件夹下面了,也就是没有多余的父文件夹了
        rawdata_index_saved = os.path.join(model_dir, 'rawdata_index_saved')
        faiss_index_saved = os.path.join(model_dir, 'faiss_index_saved')
        print("-----------------------------------")
        print(self.manifest)
        print("-----------------------------------")
        print(properties)
        print("-----------------------------------")

        print("-----------------------------------")
        print(rawdata_index_saved)
        print("-----------------------------------")
        print(faiss_index_saved)
        print("-----------------------------------")

        # 加载相关模型
        self.index_sentense={}
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir) #tokenizer需要读取词汇表
        self.model = AutoModel.from_pretrained(model_dir) #也需要读取词汇表
        dataf_index = pandas.read_json(rawdata_index_saved, lines=True)
        for index, row in dataf_index.iterrows():
            self.index_sentense[row[1]] = row[0]
        print("-----------------------------------")
        print(len(self.index_sentense))
        print(os.path.exists(faiss_index_saved))
        print("-----------------------------------")
        self.faissmodel = faiss.read_index(faiss_index_saved)

        self.model.to(self.device)
        self.model.eval()
        self.initialized = True

    def preprocess(self, requests):
        print("*******************************************************************")
        print(requests)
        print("*******************************************************************")
        traindata=[]
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode('utf-8')
            if input_text is not None:
                traindata.append(input_text)

        batch_encode = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=traindata, padding='longest',
                                                        add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                                                        max_length=512,  # 填充 & 截断长度
                                                        truncation=True, )

        return torch.tensor(batch_encode['input_ids']) \
            , torch.tensor(batch_encode['token_type_ids']) \
            , torch.tensor(batch_encode['attention_mask'])

    def inference(self, data, *args, **kwargs):
        input_ids, token_type_ids, attention_mask = data
        output = self.model(input_ids.to(self.device), token_type_ids.to(self.device), attention_mask.to(self.device))
        #进行l2正则化
        l2_pooler=self.l2_norm(output['pooler_output'].detach().cpu().numpy())
        return l2_pooler

    def postprocess(self, data):

        C, I = self.faissmodel.search(data, 10)
        print(C)
        print(I)
        dreturndata = self.index_sentense.get(I[0][0])
        return [dreturndata]

    def l2_norm(self,vecs):
        return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5