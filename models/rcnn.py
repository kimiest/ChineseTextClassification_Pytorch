from transformers import AutoModel
import torch.nn as nn
import torch
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, Config):
        super(MyModel, self).__init__()
        self.plm = AutoModel.from_pretrained(Config.plm_path)
        for param in self.plm.parameters():
            param.requires_grad = False  # PLM参数是否一起训练
            self.lstm = nn.LSTM(768, Config.rnn_hidden, Config.num_layers,
                                bidirectional=True, batch_first=True, dropout=Config.dropout)
            self.maxpool = nn.MaxPool1d(Config.max_length)
            self.fc = nn.Linear(Config.rnn_hidden * 2 + 768, Config.num_classes)

    def forward(self, batch_inputs):
        encoder_out = self.plm(input_ids=batch_inputs).last_hidden_state
        out, _ = self.lstm(encoder_out)
        out = torch.cat((encoder_out, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze(-1)
        out = self.fc(out)
        return out