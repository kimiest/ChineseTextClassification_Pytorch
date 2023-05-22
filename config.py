from transformers import AutoTokenizer

class Config():
    # 路径配置
    train_path = r'E:\Code\FinancialReportingTyping\data\train_data.json'
    eval_path = r''
    num_classes = 10

    # 通用配置
    plm_path = 'bert-base-chinese'
    tokenizer_path = 'bert-base-chinese'
    max_length = 150
    batch_size = 4
    epoch = 10
    learning_rate = 2e-5
    weight_decay = 2e-6
    schedule='CosineAnnealingLR'

    # CNN模型配置
    filter_sizes = (2, 3, 4)
    num_filters = 256
    dropout = 0.3

    # LSTM模型配置
    rnn_hidden = 100
    num_layers = 2
    dropout = 0.1



