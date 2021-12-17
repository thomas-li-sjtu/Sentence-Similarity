import tqdm
data = []
with open("NLP_Final_Project/QuoraQuestionPairs/data/train_clean.txt", "r") as file:
    data = file.readlines()
data.pop(0)
print(len(data))

sent_dict = {}
label_1_no_chain = []  #
for line in data:
    line = line.strip().split("\t")
    assert len(line) == 3
    if line[-1] == "1":
        label_1_no_chain.append("\t".join(line)+"\n")
        label_1_no_chain.append("\t".join([line[1], line[0], line[2]])+"\n")

        if sent_dict.get(line[0]):
            sent_dict[line[0]].append(line[1])
        else:
            sent_dict[line[0]] = [line[0], line[1]]

values = [set(value) for key, value in sent_dict.items()]
print(len(values))
print(len(label_1_no_chain))
print(label_1_no_chain[0], label_1_no_chain[1])
combine = []
for i in tqdm.tqdm(range(len(values))):
    joined = 0
    for j in range(len(values) - 1):
        if values[j] & values[-1]:  # 存在交集
            values[j] |= values[-1]  # 取并集
            joined = 1
            break
    if not joined:
        combine.append(values[-1].copy())
    values.pop()
for i in range(5):
    print(combine[i])
combine = [list(i) for i in combine]
import pickle
pickle.dump(combine, open("combine.pickle", "wb"))

import itertools
out = []
for i in combine:
    tmp = itertools.combinations(i, 2)
    for pair in tmp:
        out.append("{}\t{}\t{}\n".format(pair[0], pair[1], 1))
print(len(out))
unique = list(set(out) - (set(out) & set(label_1_no_chain)))
print(len(unique))
print(unique[0])

pickle.dump(unique, open("unique.pickle", "wb"))
