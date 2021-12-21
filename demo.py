import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from transformers import BertModel, BertConfig, BertTokenizer, DataCollatorWithPadding, AdamW, \
    get_linear_schedule_with_warmup
import gradio as gr


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
        sequence_out, cls_out = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                                          attention_mask=attention_mask, return_dict=False)
        logits = self.fc(self.dropout(cls_out))
        if loss_fn is not None:
            loss = self.compute_loss(sequence_out[:, 0], logits, labels, loss_fn)
            return logits, loss
        else:
            return logits

    def compute_loss(self, y_pred, logits, labels, loss_fn, tao=0.05, device="cuda:0", alpha=0.5):
        idxs = torch.arange(0, int(y_pred.shape[0]), device=device)
        y_true = idxs + 1 - idxs % 2 * 2
        similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
        similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
        similarities = similarities / tao
        loss = torch.mean(F.cross_entropy(similarities, y_true))

        nll = loss_fn(logits, labels)
        return nll + alpha * loss


class CSECollator(object):
    def __init__(self,
                 tokenizer,
                 features=("input_ids", "attention_mask", "token_type_ids", "label"),
                 max_len=100):
        self.tokenizer = tokenizer
        self.features = features
        self.max_len = max_len

    def collate(self, batch):
        new_batch = []
        for example in batch:
            for i in range(2):
                # 每个句子重复两次
                new_batch.append({fea: example[fea] for fea in self.features})
        new_batch = self.tokenizer.pad(
            new_batch,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return new_batch


vocab_file_dir = './pretrained_model/bert-base-uncased/vocab.txt'
tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)
data_collator = CSECollator(tokenizer, max_len=40)
model_param_config = ModelParamConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CustomModel("./pretrained_model/bert-base-uncased", model_param_config)
ckpt = torch.load("bert-simcse(best)/pytorch_model.bin")
model.load_state_dict(ckpt)
model.eval()
model.to(device)


def simcse(text1, text2):
    # Tokenize input texts
    inputs = tokenizer(text1, text2, truncation=True, max_length=40, return_tensors="pt")
    if "label" in inputs:
        inputs["labels"] = inputs["label"]
        del inputs["label"]
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get the embeddings
    with torch.no_grad():
        logits = model(**inputs)
        predict_scores = F.softmax(logits)

    predict_scores = predict_scores.detach().to("cpu").numpy()
    if predict_scores[0][0] > predict_scores[0][1]:
        return 0, predict_scores[0][0]
    else:
        return 1, predict_scores[0][1]
    # # Calculate cosine similarities
    # # Cosine similarities are in [-1, 1]. Higher means more similar
    # cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
    # cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])
    # return {"cosine similarity": cosine_sim_0_1}, {"cosine similarity":cosine_sim_0_2}


inputs = [
    gr.inputs.Textbox(lines=5, label="Input Text One"),
    gr.inputs.Textbox(lines=5, label="Input Text Two"),
]

# outputs = [
#             gr.outputs.Label(type="confidences", label="Similarity between text one and two"),
# ]
outputs = [
    gr.outputs.Textbox(label="Similarity Label"),
    gr.outputs.Textbox(label="Predict Scores"),
]

title = "Demo for Quora Question Pair Matching"
description = "Demo for bert-simcse. To use it, simply add your texts, or click the example to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://github.com/thomas-li-sjtu/Sentence-Similarity'>Github Repo</a></p>"
# article = "<p style='text-align: center'>
examples = [
    ["What are the books to improve English?",
     "What are some good books or resources to improve English?"]
]

gr.Interface(simcse, inputs, outputs,
             title=title,
             description=description,
             article=article,
             examples=examples,
             theme="huggingface",
             interpretation="default").launch(share=True, auth=("admin", "pass1234"))
