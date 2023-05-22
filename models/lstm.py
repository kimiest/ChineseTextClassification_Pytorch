from transformers import AutoModel
import torch.nn as nn
import torch

class MyModel(nn.Module):
    def __init__(self, Config):
        super(MyModel, self).__init__()
        self.plm = AutoModel.from_pretrained(Config.plm_path)
        for param in self.plm.parameters():
            param.requires_grad = True  # PLM参数是否一起训练
        self.lstm = nn.LSTM(768, Config.rnn_hidden, Config.num_layers,
                            bidirectional=True, batch_first=True, dropout=Config.dropout)
        self.dropout = nn.Dropout(Config.dropout)
        self.fc = nn.Linear(Config.rnn_hidden * 2, Config.num_classes)  # 双向lstm输出维度乘2

    def forward(self, batch_inputs):
        encoder_out = self.plm(input_ids=batch_inputs).last_hidden_state
        out, _ = self.lstm(encoder_out)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out