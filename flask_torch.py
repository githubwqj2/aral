import faiss
import pandas
import torch
from flask import Flask, request


# 创建Flask应用程序
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)

dim, index_param = 768, 'Flat'
index_sentense = {}
tokenizer = AutoTokenizer.from_pretrained("./simbert")  # tokenizer需要读取词汇表
model = AutoModel.from_pretrained("./simbert")  # 也需要读取词汇表
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# id_sentense的建立
dataf_index = pandas.read_json('./faiss_index/rawdata_index_saved', lines=True)
for index, row in dataf_index.iterrows():
    index_sentense[row[1]] = row[0]
# faiss的索引建立
faissmodel = faiss.read_index("./faiss_index/faiss_index_saved")
# 模型转为cuda
model.to(device)
model.eval()


def l2_norm( vecs):
    return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5

# 定义路由和处理函数
@app.route('/predictions', methods=['GET'])
def handle_predict():
    # 获取请求中的图像数据
    input_text = request.args.get("text")

    batch_encode = tokenizer.batch_encode_plus(batch_text_or_text_pairs=[input_text], padding='longest',
                                                    add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                                                    max_length=512,  # 填充 & 截断长度
                                                    truncation=True, )

    input_ids, token_type_ids, attention_mask = batch_encode['input_ids'],batch_encode['token_type_ids'],batch_encode['attention_mask']
    print("**************************token*****************************************")
    print(input_ids)
    print("***************************token****************************************")
    output = model(input_ids=torch.tensor(input_ids).to(device), token_type_ids=torch.tensor(token_type_ids).to(device), attention_mask=torch.tensor(attention_mask).to(device))
    # 进行l2正则化
    l2_pooler = l2_norm(output['pooler_output'].detach().cpu().numpy())
    C, I = faissmodel.search(l2_pooler, 10)


    dreturndata =index_sentense.get(I[0][0])

    # 返回预测结果
    return [dreturndata]


# 运行Flask应用程序
if __name__ == '__main__':
    app.run()
