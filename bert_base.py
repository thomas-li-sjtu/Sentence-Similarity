import os
import json
import argparse
import numpy as np
from datasets import load_dataset, load_metric
from sklearn.metrics import accuracy_score, f1_score
from transformers.trainer_utils import SchedulerType
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, DataCollatorForLanguageModeling, \
    TrainingArguments, \
    Trainer, DataCollatorWithPadding, EarlyStoppingCallback

# disable wandb
os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, help='pair文本最大长度', default=40)
parser.add_argument('--batch_size', type=int, help='batch_size', default=256)
parser.add_argument('--pretrained_model_dir', type=str, help='预训练模型地址', default='./pretrained_model/bert-base-uncased')
parser.add_argument('--num_train_epochs', type=int, help='训练epoch数', default=3)
parser.add_argument('--max_steps', type=int, help='最大训练步数，如果设置了，则覆盖num_train_epochs', default=-1)
parser.add_argument('--warmup_ratio', type=float, help='warmup比例', default=0.05)
parser.add_argument('--learning_rate', type=float, help='学习率', default=2e-5)
parser.add_argument('--evaluation_strategy', type=str, help='evaluation策略：steps or epoch', default='epoch')
parser.add_argument('--metric_for_best_model', type=str, help='f1 or accuracy', default='f1')
parser.add_argument('--early_stopping_patience', type=int, help='early_stopping_patience', default=2)
parser.add_argument('--save_strategy', type=str, help='保存模型策略', default='epoch')
parser.add_argument('--save_total_limit', type=int, help='checkpoint数量', default=1)
parser.add_argument('--logging_steps', type=int, help='多少步日志打印', default=100)
parser.add_argument('--output_model_dir', type=str, help='模型保存地址', default='./bert-finetune')
parser.add_argument('--seed', type=int, help='随机种子', default=2021)
args = parser.parse_args()

# 读取数据
data_files = {"train": "NLP_Final_Project/QuoraQuestionPairs/data/train_clean.txt",
              "valid": "NLP_Final_Project/QuoraQuestionPairs/data/valid.txt",
              "test": "NLP_Final_Project/QuoraQuestionPairs/data/test.txt"}
dataset = load_dataset('csv', data_files=data_files, delimiter='\t', quoting=3)

# 加载词表
vocab_file_dir = './pretrained_model/bert-base-uncased/vocab.txt'
tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=args.max_len)


# 定义处理函数
def tokenize_function(example):
    return tokenizer(example["text_a"], example["text_b"], truncation=True, max_length=args.max_len)


# tokenize数据
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.rename_column("label", "labels")
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["valid"]
full_test_dataset = tokenized_datasets["test"]

# 加载模型（Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output)）
model = BertForSequenceClassification.from_pretrained(args.pretrained_model_dir, num_labels=2)

# 修改embedding大小
model.resize_token_embeddings(len(tokenizer))

# 配置训练参数
training_args = TrainingArguments(
    num_train_epochs=args.num_train_epochs,
    max_steps=args.max_steps,
    evaluation_strategy=args.evaluation_strategy,
    load_best_model_at_end=True,
    metric_for_best_model=args.metric_for_best_model,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    eval_accumulation_steps=1,
    learning_rate=args.learning_rate,
    lr_scheduler_type=SchedulerType.LINEAR,
    warmup_ratio=args.warmup_ratio,
    output_dir=args.output_model_dir,
    overwrite_output_dir=True,
    save_strategy=args.save_strategy,
    save_total_limit=args.save_total_limit,
    logging_steps=args.logging_steps,
    logging_first_step=True,
    seed=args.seed,
)


def compute_metrics_v1(eval_pred):
    # 这里使用的是本地metric，主要是防止网络不通，实际上可以直接metric1 = load_metric("accuracy")，程序会从远端加载metric计算方法
    metric1 = load_metric("./metrics/accuracy.py")
    metric2 = load_metric("./metrics/f1.py")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # 这里要注意metric计算结果是一个dict，需要用key获取对应的指标值
    accuracy = metric1.compute(predictions=predictions, references=labels)['accuracy']
    f1 = metric2.compute(predictions=predictions, references=labels, average='macro')['f1']
    return {"accuracy": accuracy, "f1": f1}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=full_train_dataset,
    eval_dataset=full_eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics_v1,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
)

# 训练模型
trainer.train()
trainer.save_model(args.output_model_dir)
trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

# 预测数据
test_predictions, test_label_ids, test_metrics = trainer.predict(full_test_dataset)
np.savetxt(os.path.join(args.output_model_dir, 'test_predictions'), test_predictions)
np.savetxt(os.path.join(args.output_model_dir, 'test_label_ids'), test_label_ids)
with open(os.path.join(args.output_model_dir, "test_metrics"), 'w') as fout:
    json_dumps_str = json.dumps(test_metrics, indent=4)
    fout.write(json_dumps_str)

# 保存词表
os.system("cp %s %s" % (vocab_file_dir, training_args.output_dir))