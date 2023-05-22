from transformers import AutoModel
import torch.nn as nn
import torch


class MyModel(nn.Module):
    def __init__(self, Config):
        super(MyModel, self).__init__()
        self.bert = AutoModel.from_pretrained(Config.plm_path)
        self.fc = nn.Linear(768, Config.num_classes)

    def forward(self, batch_inputs):
        cls = self.bert(input_ids=batch_inputs).pooler_output
        out = self.fc(cls)
        return out