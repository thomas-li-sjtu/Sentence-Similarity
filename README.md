自然语言处理前言技术大作业：判断给定的两个句子含义是否相同（1为相同，0为不同）

## 数据清理
* max_len：40
* 句子中存在一些脏数据：
    * `How can I develop android app?          0`——train.txt
    * `How can I create an Android app?          0`——valid.txt
* 数据增强：（仅链式）增加9万多条正例

## 模型
1. bert-base-uncased
2. bert-base-enhance
3. bert-rdrop
4. bert-simcse
5. sentence-bert
6. albert
7. roberta

   |       Model        |  acc  | precision | recall |  F1   |
   | :----------------: | :---: | :-------: | :----: | :---: |
   | bert-base-uncased  | 89.39 |   83.45   | 88.61  | 88.72 |
   | bert-base-enhance  | 90.60 |   85.01   | 90.23  | 90.00 |
   |     bert-rdrop     | 87.56 |   81.73   | 85.04  | 86.71 |
   |     bert-simcse    | 89.95 |   85.80   | 86.96  | 89.21 |
   |    sentence-bert   | 89.03 |   84.42   | 85.90  | 88.23 |
   |       albert       | 89.46 |   85.48   | 85.79  | 88.66 |
   |       roberta      | 90.10 |   86.03   | 90.04  | 90.39 |

 