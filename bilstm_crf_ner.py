import json

import requests


def intent_classifier(text):
    url = 'http://127.0.0.1:60061/service/api/medical_ner'
    data = {"text_list":text}
    headers = {'Content-Type':'application/json;charset=utf8'}
    reponse = requests.post(url,data=json.dumps(data),headers=headers)
    if reponse.status_code == 200:
        reponse = json.loads(reponse.text)
        rawdata=reponse['data']
        returndict={}
        for rwa in rawdata:
            dieasestr=rwa.get("string")
            entities=set([_.get("word") for _ in rwa.get("entities",[])])
            returndict[dieasestr]=entities
        returnarray= [(_, returndict.get(_)) for _ in text]
        return returnarray

    else:
        return -1



if __name__ == '__main__':

    question="同房要注意哪些卫生 男朋友来看我，晚上开了房间，就迫不及待的要吃我，叫他去洗澡都不肯，没办法我们就那样做了一次"
    matchquestion2=["小孩哮喘有什么症状？ 无",
                    "急性甲亢与甲亢的区别 我同事的嫂子患有急性甲亢，20天左右就治好了，我想问一下，我患有甲亢，也能在这么短的时间里治愈吗？另外急性甲亢与甲亢有什么区别啊",
                    "脑瘤这个病是怎么长出来的？ 我二十年前的时候出过车祸，然后得了脑震荡，不过治好了，但是这几天我自己觉着不对劲，就是头特别疼，而且是那种没任何理由的疼，我查了资料，恐怕是脑子里长了脑瘤。在乎怎样的帮助：脑瘤这个病是怎么长出的的？",
                    "同房要注意哪些卫生 男朋友来看我，晚上开了房间，就迫不及待的要吃我，叫他去洗澡都不肯，没办法我们就那样做了一次，之后我就感觉阴道疼的很，每天下体有黄色的东西排出来，气味很难闻。想得到怎样的帮助:同房要注意哪些卫生？",
                    "吃完避孕药在同房会不会怀孕？ 我昨晚刚吃了避孕药之后就和老公同房了，您好医生，吃完避孕药在同房会不会怀孕？",
                    "男性精囊炎有什么症状？ 吃头一个月的时候我过一次房事，还是会偶尔的尿频尿急的，有时候还会出冷汗，血精还是有点的。"
                    ]
    print(intent_classifier([question]+matchquestion2))