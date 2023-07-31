import os

import torch
from transformers import AutoTokenizer, AutoModel
from ts.torch_handler.base_handler import BaseHandler


class MyHandler(BaseHandler):

    def __init__(self):
        super(MyHandler, self).__init__()
        self.initialized = False

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        serialized_file = self.manifest["sentiment-dl-common-api-model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        self.tokenizer = AutoTokenizer.from_pretrained(model_pt_path)
        self.model = AutoModel.from_pretrained(model_pt_path)
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        batch_encode = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=data, padding='longest',
                                                        add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                                                        max_length=512,  # 填充 & 截断长度
                                                        truncation=True, )

        return torch.tensor(batch_encode['input_ids']) \
            , torch.tensor(batch_encode['token_type_ids']) \
            , torch.tensor(batch_encode['attention_mask'])

    def inference(self, data, *args, **kwargs):

        input_ids, token_type_ids, attention_mask = data
        output = self.model(input_ids.to(self.device), token_type_ids.to(self.device), attention_mask.to(self.device))

        return output['pooler_output'].detach().cpu().numpy()

    def postprocess(self, data):
        return super().postprocess(data)
