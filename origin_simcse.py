import torch
import tqdm
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

# Import our models. The package will take care of downloading the models automatically
# tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased", cache_dir='./pretrained_model/sup-simcse-bert')
# model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased", cache_dir='./pretrained_model/sup-simcse-bert')

# tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-roberta-base", cache_dir='./pretrained_model/unsup-simcse-roberta-base')
# model = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-roberta-base", cache_dir='./pretrained_model/unsup-simcse-roberta-base')

with open("NLP_Final_Project/QuoraQuestionPairs/data/test.txt", "r") as file:
    data = file.readlines()
    data.pop(0)
labels = []
for i in range(len(data)):
    data[i] = data[i].strip().split("\t")
    assert len(data[i]) == 3
    labels.append(int(data[i][-1]))
    data[i] = data[i][:2]
print(len(labels))
print(data[:3])

# result = []
#
# num_batch = len(data) // 16
# for i in tqdm.tqdm(range(num_batch)):
#     texts = []
#     for j in range(i * 16, (i + 1) * 16):
#         texts.extend(data[j])
#
#     inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
#
#     # Get the embeddings
#     with torch.no_grad():
#         embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
#
#     # Calculate cosine similarities
#     # Cosine similarities are in [-1, 1]. Higher means more similar
#     for j in range(0, 32, 2):
#         cos = 1 - cosine(embeddings[j], embeddings[j + 1])
#         result.append(cos)
#
# import pickle
#
# pickle.dump(result, open("test_simcse.pickle", "wb"))

import pickle
from sklearn.metrics import accuracy_score, f1_score

result = pickle.load(open("test_simcse.pickle", "rb"))
result = [0 if i < 0.74 else 1 for i in result]
eval_f1 = f1_score(labels[:len(result)], result, average='macro')
print(eval_f1)
