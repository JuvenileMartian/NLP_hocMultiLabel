import csv
from transformers import BertTokenizer

with open('./bert_data/hoc/train.tsv', "r") as f:
    reader = csv.reader(f, delimiter="\t")
    lines = []
    for line in reader:
        lines.append(line)

with open('./bert_data/hoc/train_unlabeled_positive.txt', 'w', encoding='utf-8') as file:
    for sentence in lines:
        if '_1' in sentence[0]:
            file.write(sentence[1].lower() + '\n')  # 每个句子后加上换行符

# python -m sentencepiece.train --input='./bert_data/hoc/train_unlabeled.txt' --model_prefix=bpe_model --vocab_size=30522 --model_type=bpe --character_coverage=1.0

import sentencepiece as spm

# 定义训练参数
input_file = './bert_data/hoc/train_unlabeled.txt'
model_prefix = "bpe_model"
vocab_size = 30517
model_type = "bpe"  # 可以选择 "bpe", "unigram", "char", "word"
character_coverage = 1.0  # 对于英文可以用1.0，多语言推荐0.9995

# 训练 SentencePiece 模型
spm.SentencePieceTrainer.train(
    input=input_file,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type=model_type,
    character_coverage=character_coverage
)

sp = spm.SentencePieceProcessor(model_file='bpe_model.model')

# 查看词汇表大小
print("Vocabulary size:", sp.get_piece_size())

# 打印前 10 个词汇
for i in range(10):
    print(sp.id_to_piece(i), sp.get_score(i))

vocab_list = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

# 添加 BERT 必需的特殊 token
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
vocab_list = special_tokens + vocab_list

# 保存为 vocab.txt
with open("vocab_bpe.txt", "w", encoding="utf-8") as f:
    for token in vocab_list:
        f.write(token + "\n")

print("✅ vocab_bpe.txt 已成功生成！")

text = "The present study investigated whether ALA disrupts the Warburg effect , which represents a shift in ATP generation from oxidative phosphorylation to glycolysis , protecting tumor cells against oxidative stress-mediated apoptosis ."
tokenizer = BertTokenizer(vocab_file="vocab_bpe.txt")
tokenizer.tokenize(text)