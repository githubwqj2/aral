import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

simberttokenizer = AutoTokenizer.from_pretrained("./simbert")
simbertencoder = AutoModel.from_pretrained("./simbert")
simbertencoder.eval()
dim, index_param = 768, 'Flat'
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
simbertencoder.to(device)





def predicts(question1, matchquestion2):
    batchdata=[question1]+matchquestion2
    batch_encode = simberttokenizer.batch_encode_plus(batch_text_or_text_pairs=batchdata, padding='longest',
                                               add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                                               max_length=512,  # 填充 & 截断长度
                                               truncation=True, )



    input_ids=torch.tensor(batch_encode['input_ids'])
    token_type_ids=torch.tensor(batch_encode['token_type_ids'])
    attention_mask= torch.tensor(batch_encode['attention_mask'])
    output = simbertencoder(input_ids=input_ids.to(device), token_type_ids=token_type_ids.to(device),
                                attention_mask=attention_mask.to(device))
    pooler_out=output['pooler_output'].detach().cpu().numpy()
    similar=cosine_similarity(np.array(pooler_out[1:]),np.array(pooler_out[0]).reshape(1, -1))
    return similar

if __name__ == '__main__':
    question="同房要注意哪些卫生 男朋友来看我，晚上开了房间，就迫不及待的要吃我，叫他去洗澡都不肯，没办法我们就那样做了一次"
    matchquestion2=["小孩哮喘有什么症状？ 无",
                    "急性甲亢与甲亢的区别 我同事的嫂子患有急性甲亢，20天左右就治好了，我想问一下，我患有甲亢，也能在这么短的时间里治愈吗？另外急性甲亢与甲亢有什么区别啊",
                    "脑瘤这个病是怎么长出来的？ 我二十年前的时候出过车祸，然后得了脑震荡，不过治好了，但是这几天我自己觉着不对劲，就是头特别疼，而且是那种没任何理由的疼，我查了资料，恐怕是脑子里长了脑瘤。在乎怎样的帮助：脑瘤这个病是怎么长出的的？",
                    "同房要注意哪些卫生 男朋友来看我，晚上开了房间，就迫不及待的要吃我，叫他去洗澡都不肯，没办法我们就那样做了一次，之后我就感觉阴道疼的很，每天下体有黄色的东西排出来，气味很难闻。想得到怎样的帮助:同房要注意哪些卫生？",
                    "吃完避孕药在同房会不会怀孕？ 我昨晚刚吃了避孕药之后就和老公同房了，您好医生，吃完避孕药在同房会不会怀孕？",
                    "男性精囊炎有什么症状？ 吃头一个月的时候我过一次房事，还是会偶尔的尿频尿急的，有时候还会出冷汗，血精还是有点的。"
                    ]
    result=predicts(question,matchquestion2)
    print(result)