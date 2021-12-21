import os
import time
import random
import argparse
import numpy as np
import logging
from distutils.util import strtobool
from sklearn.metrics import accuracy_score, f1_score
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast  # 只有torch 1.6以上才有
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertModel, BertConfig, BertTokenizer, DataCollatorWithPadding, AdamW, \
    get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, help='pair文本最大长度', default=40)
parser.add_argument('--batch_size', type=int, help='batch_size', default=256)
parser.add_argument('--pretrained_model_dir', type=str, help='预训练模型地址', default='./pretrained_model/bert-base-uncased')
parser.add_argument('--num_train_epochs', type=int, help='训练epoch数', default=10)
parser.add_argument('--max_steps', type=int, help='最大训练步数，如果设置了，则覆盖num_train_epochs', default=-1)
parser.add_argument('--warmup_ratio', type=float, help='warmup比例', default=0.05)
parser.add_argument('--learning_rate', type=float, help='学习率', default=2e-5)
parser.add_argument('--adam_epsilon', type=float, help='adam_epsilon', default=1e-6)
parser.add_argument('--max_grad_norm', type=int, help='max_grad_norm', default=1)
parser.add_argument('--early_stopping_patience', type=int, help='early_stopping_patience', default=2)
parser.add_argument('--save_steps', type=int, help='多少步保存模型', default=100)
parser.add_argument('--save_total_limit', type=int, help='checkpoint数量', default=1)
parser.add_argument('--logging_steps', type=int, help='多少步日志打印', default=100)
parser.add_argument('--output_model_dir', type=str, help='模型保存地址', default='./bert-rdrop')
parser.add_argument('--seed', type=int, help='随机种子', default=2021)
args = parser.parse_args()


class ModelParamConfig:
    def __init__(self):
        self.num_classes = 2
        self.dropout_prob = 0.1


class CustomModel(nn.Module):
    def __init__(self, pretrain_model_path, model_param_config):
        super(CustomModel, self).__init__()
        self.config = BertConfig.from_pretrained(pretrain_model_path, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(pretrain_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(model_param_config.dropout_prob)
        self.fc = nn.Linear(self.config.hidden_size, model_param_config.num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, loss_fn=None):
        # bz: [bz, 128] [bz, 128] [bz, 128] [bz]
        input_ids, attention_mask, token_type_ids = input_ids.repeat(2, 1), attention_mask.repeat(2, 1), token_type_ids.repeat(2, 1)
        sequence_out, cls_out = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                                          attention_mask=attention_mask, return_dict=False)
        cls_out = self.dropout(cls_out)
        logits = self.fc(cls_out)
        if loss_fn is not None:
            loss = self.compute_loss(logits, labels, loss_fn)
            return logits, loss
        else:
            return logits

    def compute_loss(self, logits, labels, loss_fn, alpha=5):
        # keep dropout and forward twice
        # logits = model(x)
        #
        # logits2 = model(x)

        # cross entropy loss for classifier
        # ce_loss = 0.5 * (nn.cross_entropy_loss(logits, label) + cross_entropy_loss(logits2, label))
        ce_loss1 = loss_fn(logits[:list(labels.shape)[0]], labels)
        ce_loss2 = loss_fn(logits[list(labels.shape)[0]:], labels)
        ce_loss = 0.5 * (ce_loss1 + ce_loss2)

        kl_loss = self.compute_kl_loss(logits[:list(labels.shape)[0]], logits[list(labels.shape)[0]:])

        # carefully choose hyper-parameters
        loss = ce_loss + 0.5 * kl_loss

        return loss

    @staticmethod
    def compute_kl_loss(p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss


def seed_everything(seed):
    """
    固定随机种子
    :param random_seed: 随机种子数目
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(model):
    output_dir = os.path.join(args.output_model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f'Saving model to {output_dir}')
    torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))


# 读取数据
data_files = {"train": "NLP_Final_Project/QuoraQuestionPairs/data/train_clean.txt",
              "valid": "NLP_Final_Project/QuoraQuestionPairs/data/valid.txt",
              "test": "NLP_Final_Project/QuoraQuestionPairs/data/test.txt"}
dataset = load_dataset('csv', data_files=data_files, delimiter='\t', quoting=3)

# 加载词表
vocab_file_dir = './pretrained_model/bert-base-uncased/vocab.txt'
tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=args.max_len)


def tokenize_function(example):
    return tokenizer(example["text_a"], example["text_b"], truncation=True, max_length=args.max_len)


tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=8)
tokenized_datasets = tokenized_datasets.remove_columns(["text_a", "text_b"])
tokenized_datasets.rename_column("label", "labels")

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["valid"], batch_size=args.batch_size, collate_fn=data_collator
)
test_dataloader = DataLoader(
    tokenized_datasets["test"], batch_size=args.batch_size, collate_fn=data_collator
)

# 模型训练
model_param_config = ModelParamConfig()
# 模型指定GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CustomModel(args.pretrained_model_dir, model_param_config).to(device)
# 优化器定义
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  # 规定哪些参数不进行衰减
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
total_steps = args.num_train_epochs * len(train_dataloader)

num_warmup_steps = int(total_steps * args.warmup_ratio)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

# 损失函数定义
loss_nll = nn.CrossEntropyLoss()

# 模型训练
global_step = 0
save_steps = total_steps // args.num_train_epochs
eval_steps = save_steps
log_loss_steps = args.logging_steps
avg_loss = 0.

best_f1 = 0.
early_stop = 0
for epoch in range(args.num_train_epochs):
    train_loss = 0.0
    logger.info('\n------------epoch:{}------------'.format(epoch))
    last = time.time()
    for step, batch_data in enumerate(tqdm.tqdm(train_dataloader)):
        model.train()
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        loss = model(**batch_data, loss_fn=loss_nll)[1]
        train_loss += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()
        model.zero_grad()

        global_step += 1
        if global_step % log_loss_steps == 0:
            avg_loss /= log_loss_steps
            logger.info('Step: %d / %d ----> total loss: %.5f' % (global_step, total_steps, avg_loss))
            avg_loss = 0.
        else:
            avg_loss += loss.item()

    logger.info(f"微调第{epoch}轮耗时：{time.time() - last}")

    eval_loss = 0
    eval_acc = 0
    y_true = []
    y_predict = []
    y_predict_target = []
    model.eval()
    with torch.no_grad():
        for step, batch_data in enumerate(tqdm.tqdm(eval_dataloader)):
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)
            labels = batch_data['labels']
            y_true.extend(labels.cpu().numpy())

            logits, loss = model(**batch_data, loss_fn=loss_nll)
            logits = logits[:len(labels)]  # 训练时，输入被double了
            predict_scores = F.softmax(logits)
            y_predict_target.extend(predict_scores.argmax(dim=1).detach().to("cpu").numpy())
            predict_scores = predict_scores[:, 1]
            y_predict.extend(predict_scores.detach().to("cpu").numpy())

            acc = ((logits.argmax(dim=-1) == labels).sum()).item()
            eval_acc += acc / logits.shape[0]
            eval_loss += loss

    eval_loss = eval_loss / len(eval_dataloader)
    eval_acc = eval_acc / len(eval_dataloader)
    eval_f1 = f1_score(y_true, y_predict_target, average='macro')

    if best_f1 < eval_f1:
        early_stop = 0
        best_f1 = eval_f1
        save_model(model)
    else:
        early_stop += 1
    logger.info(
        'epoch: %d, train loss: %.8f, eval loss: %.8f, eval acc: %.8f, eval f1: %.8f, best_f1: %.8f\n' %
        (epoch, train_loss, eval_loss, eval_acc, eval_f1, best_f1))
    # 检测早停
    if early_stop >= args.early_stopping_patience:
        break

    test_loss, test_acc, y_true, y_predict, y_predict_target = 0, 0, [], [], []
    with torch.no_grad():
        for step, batch_data in enumerate(tqdm.tqdm(test_dataloader)):
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)
            labels = batch_data['labels']
            y_true.extend(labels.cpu().numpy())

            logits, loss = model(**batch_data, loss_fn=loss_nll)
            logits = logits[:len(labels)]
            predict_scores = F.softmax(logits)
            y_predict_target.extend(predict_scores.argmax(dim=1).detach().to("cpu").numpy())
            predict_scores = predict_scores[:, 1]
            y_predict.extend(predict_scores.detach().to("cpu").numpy())

            acc = ((logits.argmax(dim=-1) == labels).sum()).item()
            test_acc += acc / logits.shape[0]
            test_loss += loss

    test_loss = test_loss / len(test_dataloader)
    test_acc = test_acc / len(test_dataloader)
    test_f1 = f1_score(y_true, y_predict_target, average='macro')
    logger.info(
        'epoch: %d, train loss: %.8f, test loss: %.8f, test acc: %.8f, test f1: %.8f, best_f1: %.8f\n' %
        (epoch, train_loss, test_loss, test_acc, test_f1, best_f1))

    torch.cuda.empty_cache()  # 每个epoch结束之后清空显存，防止显存不足

# 保存词表
os.system("cp %s %s" % (vocab_file_dir, args.output_model_dir))
# 保存config
os.system("cp %s %s" % (os.path.join(args.pretrained_model_dir, 'config.json'), args.output_model_dir))
