from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, BertTokenizerFast
from collections import Counter, defaultdict
import re

# 原始 BERT 的词表路径
original_vocab_file = 'vocab.txt'  # 假如这是原始的vocab.txt文件

# 读取原始词表并加载
with open(original_vocab_file, 'r', encoding='utf-8') as f:
    original_tokens = [line.strip() for line in f]

# 初始化一个 BertWordPieceTokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")



# 训练语料文件
corpus_file = './bert_data/hoc/train_unlabeled_positive.txt'

counter = defaultdict(float)

with open(corpus_file, 'r', encoding='utf-8') as f:
        counter = Counter(f.read().split())

# 从大到小排序并提取元素名
sorted_elements = [(item, count) for item, count in counter.most_common()]

# 打印排序后的元素列表
print(sorted_elements)
#new_words = [word for word in sorted_elements if word not in tokenizer.get_vocab()]

# 训练新的词表，并在原始词表的基础上进行扩展
#tokenizer.add_tokens(new_words)


# 保存新的词表和 Tokenizer
#save_dir = 'extended_tokenizer'
#tokenizer.save_model(save_dir)

def is_all_english_or_symbols(string: str) -> bool:
    # 匹配字符串是否只包含英文字符（大小写）、数字、空格、以及常见符号
    pattern = r'^[a-zA-Z0-9\s\.\,\!\?\:\;\-\_\(\)\[\]\{\}\"\'\/\\@#\$%\^&\*\+=<>]*$'
    return bool(re.fullmatch(pattern, string))

vocab = tokenizer.get_vocab()

lv = sorted(vocab.items(), key=lambda x: x[1])
lv = [x[0] for x in lv]

new_words = [word for word in sorted_elements if word[0] not in lv]
nn = iter(new_words)

n_replace = 20
n = 0
for i in range(len(lv)):     
     if '[unused' in lv[i]:# or not is_all_english_or_symbols(lv[i]):
          lv[i] = next(nn)[0]
          n += 1
          if n >= n_replace:
               break

# 将词表写入文件
with open("vocab_extip20.txt", "w", encoding="utf-8") as f:
    i = 0
    g = 0
    for token in lv:  # 按照索引排序
        #if not '[unused' in token and is_all_english_or_symbols(token):
            f.write(token + "\n")
            i += 1
            if i == 30522:
                break
        #else:
             #g +=1
    #print(g)