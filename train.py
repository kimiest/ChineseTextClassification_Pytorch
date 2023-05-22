from tqdm import tqdm

def train_one_epoch(model, optimizer, scheduler, criterion, train_dl, device, epoch):
    model.train()
    num_examples = 0
    total_loss = 0.0
    total_correct = 0

    bar = tqdm(enumerate(train_dl), total=len(train_dl))
    for i, batch in bar:
        batch_inputs = batch['inputs'].to(device)
        batch_labels = batch['labels'].to(device)

        '''获取模型输出并且计算损失'''
        out = model(batch_inputs)
        loss = criterion(out, batch_labels)

        '''1.清空梯度 2.反向传播求梯度 3.优化参数'''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''计算当前epoch中：1.预测正确的样本总数 2.总损失'''
        num_examples += len(batch_labels)
        batch_preds = out.argmax(dim=-1)
        correct = (batch_preds == batch_labels).sum().item()
        total_correct += correct
        total_loss += loss.item()

        '''计算准确率和平均损失，显示在tadm中'''
        accuracy  = total_correct / num_examples
        avg_loss = total_loss / num_examples
        bar.set_postfix(epoch=epoch, train_loss=avg_loss, train_accuracy=accuracy)

        '''每隔500个batch，调整一次学习率'''
        if i % 300 == 0:
            if scheduler is not None:
                scheduler.step()

    return avg_loss, accuracy

