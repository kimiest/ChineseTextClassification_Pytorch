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
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, Config.num_filters, (k, 768)) for k in Config.filter_sizes])
        self.dropout = nn.Dropout(Config.dropout)

        self.fc = nn.Linear(Config.num_filters * len(Config.filter_sizes), Config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, batch_inputs):
        encoder_out = self.plm(input_ids=batch_inputs).last_hidden_state
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out