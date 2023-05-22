# 导入所需的库
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import gensim
import jieba
from util import Read_file_list,combine2gram
import re
from collections import Counter

#文本预处理
path_list = Read_file_list(r".\txt")
corpus = []
for path in path_list:
    with open(path, "r", encoding="ANSI") as file:
        text = [line.strip("\n").replace("\u3000", "").replace("\t", "") for line in file][3:]
        corpus += text

#去掉停词
with open("cn_stopwords.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
lines = [line.strip('\n') for line in lines]
for j in range(len(corpus)):
    for line in lines:
        corpus[j]=corpus[j].replace(line, "")
        corpus[j] = corpus[j].replace(" ", "")
regex_str = ".*?([^\u4E00-\u9FA5]).*?"
english = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;「<=>?@，。?★、…【】〖〗《》？“”‘’！[\\]^_`{|}~]+'
symbol = []
for j in range(len(corpus)):
    corpus[j] = re.sub(english, "", corpus[j])
    symbol += re.findall(regex_str, corpus[j])
'''
count_ = Counter(symbol)
count_symbol = count_.most_common()
noise_symbol = []
for eve_tuple in count_symbol:
    if eve_tuple[1] < 200:
        noise_symbol.append(eve_tuple[0])
noise_number = 0
for line in corpus:
    for noise in noise_symbol:
        line=line.replace(noise, "")
        noise_number += 1
print("完成的噪声数据替换点：", noise_number)
'''

token = []
for para in corpus:
    token += [i for i in para]

#print(token)

gensim_model = gensim.models.Word2Vec(token,vector_size=256, window=20,min_count=1,workers=4)
gensim_model.wv.save_word2vec_format('word_vec_2.txt')
