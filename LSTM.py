# 导入所需的库
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import gensim
import jieba
from util import Read_file_list, combine2gram
import re
from collections import Counter


# 定义生成文本的函数，给定一个初始文本和生成长度，返回生成的文本字符串
def generate_text(model, start_text, gen_length, seq_length):
    model.eval()  # 将模型设为评估模式，不使用dropout等技巧
    with torch.no_grad():  # 不计算梯度，节省内存和时间
        text_generated = []  # 存储生成的字符索引
        for i in range(gen_length):
            print(start_text)
            input_eval = [gensim_model.wv[c] for c in start_text]  # 将初始文本转换为整数列表，并作为输入张量
            input_eval_tensor = torch.tensor(input_eval)
            input_eval_tensor = input_eval_tensor.view(-1, batch_size, 256)
            output = model(input_eval_tensor.to(device))  # hidden_state
            output = output.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
            rand = np.random.randint(10)
            predicted = gensim_model.wv.most_similar(output, topn=10)[rand][0]
            start_text = start_text + predicted
            start_text = start_text[1:]
            text_generated += predicted
    return text_generated


# 文本预处理
path_list = Read_file_list(r".\txt")
corpus = []
for path in path_list:
    with open(path, "r", encoding="ANSI") as file:
        text = [line.strip("\n").replace("\u3000", "").replace("\t", "") for line in file][3:]
        corpus += text

# 去掉停词
'''
with open("cn_stopwords.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
lines = [line.strip('\n') for line in lines]
'''
for j in range(len(corpus)):
    #for line in lines:
        #corpus[j] = corpus[j].replace(line, "")
    corpus[j] = corpus[j].replace(" ", "")

regex_str = ".*?([^\u4E00-\u9FA5]).*?"
english = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;「<=>?@，。?★、…【】〖〗《》？“”‘’！[\\]^_`{|}~]+'
symbol = []
for j in range(len(corpus)):
    corpus[j] = re.sub(english, "", corpus[j])
    corpus[j] = re.sub(regex_str, "", corpus[j])
    #symbol += re.findall(regex_str, corpus[j])

token = []
for para in corpus:
    token += [i for i in para]

gensim_model = gensim.models.Word2Vec(token,vector_size=256, window=20,min_count=1,workers=4)
#gensim_model = gensim.models.KeyedVectors.load_word2vec_format('word_vec_2.txt', binary=False)

vocab_size = len(gensim_model.wv.key_to_index)  # 字符集大小

# 定义序列长度和批量大小
seq_length = 256  # 每个输入序列包含10个字符
batch_size = 20  # 每个批次包含2个序列

# 将文本数据转换为整数序列
dataX = []  # 输入序列列表
dataY = []  # 目标序列列表
for i in range(0, len(token) - batch_size, batch_size):
    seq_in = token[i:i + batch_size]  # 取seq_length个字符作为输入序列
    seq_out = token[i + batch_size]  # 取下一个字符作为目标序列
    dataX.append([gensim_model.wv[c] for c in seq_in])  # 将输入序列转换为整数列表并添加到dataX中
    dataY.append(gensim_model.wv[seq_out])  # 将目标序列转换为整数并添加到dataY中

# 将dataX和dataY转换为PyTorch张量，并调整形状为(batch_size, seq_length)
X = torch.tensor(dataX)
y = torch.tensor(dataY)
X = X.view(-1, batch_size, seq_length)  # 调整形状，使得第一个维度是批次大小
y = y.view(-1, seq_length)  # 调整形状，使得第一个维度是批次大小

print("X.shape : " + str(X.shape))

# 定义LSTM模型的参数
#emb_dim = 16  # 嵌入层的维度
hidden_dim = 128  # LSTM层的隐藏状态维度
n_layers = 1  # LSTM层的层数
dropout = 0  # LSTM层的dropout概率


# 定义LSTM模型类
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, dropout):  # , emb_dim
        super(LSTMModel, self).__init__()
        self.vocab_size = vocab_size  # .to(device)
        #self.emb_dim = emb_dim  # .to(device)
        self.hidden_dim = hidden_dim  # .to(device)
        self.n_layers = n_layers  # .to(device)
        self.dropout = dropout  # .to(device)

        # 定义嵌入层，将整数索引转换为向量表示
        # self.embedding = nn.Embedding(vocab_size, emb_dim)

        # 定义LSTM层，接收嵌入向量作为输入，输出隐藏状态和单元状态
        self.lstm = nn.LSTM(vocab_size, hidden_dim, n_layers, dropout=dropout)

        # 定义线性层，接收最后一个时间步的隐藏状态作为输入，输出分类概率分布
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x的形状是(batch_size, seq_length)
        batch_size = x.size(0)

        # 通过嵌入层得到嵌入向量，形状是(batch_size, seq_length, emb_dim)
        # x = self.embedding(x)
        # x = x.view(-1,1,20)

        # 调整嵌入向量的形状，使得第一个维度是序列长度，第二个维度是批次大小，第三个维度是嵌入维度，形状是(seq_length, batch_size, emb_dim)
        x = x.transpose(0, 1)

        # 初始化LSTM层的隐藏状态和单元状态，形状都是(n_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)

        # 通过LSTM层得到所有时间步的输出和最后一个时间步的隐藏状态和单元状态，输出的形状是(seq_length, batch_size, hidden_dim)，隐藏状态和单元状态的形状都是(n_layers, batch_size, hidden_dim)
        output, (hn, cn) = self.lstm(x, (h0, c0))  # .to(device)

        # 取最后一个时间步的输出作为线性层的输入，形状是(batch_size, hidden_dim)
        output = output[-1]

        # 通过线性层得到分类概率分布，形状是(batch_size, vocab_size)
        output = self.linear(output)

        return output


# 创建LSTM模型实例，并移动到GPU上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(seq_length, hidden_dim, n_layers, dropout).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss(reduction='mean')  # CrossEntropyLoss()  # 使用交叉熵损失函数，适合分类问题
optimizer = torch.optim.Adam(model.parameters())  # 使用Adam优化器

# 定义训练轮数和打印间隔
n_epochs = 100  # 训练100轮
print_every = 1  # 每10轮打印一次损失值

data = range(1, n_epochs + 1)

# 开始训练模型
with open("generated_text.txt", "w") as file:
    for epoch in tqdm(data):
        model.train()
        print("epoch : " + str(epoch))
        # 初始化总损失为0
        total_loss = 0

        count = 0

        # data = [zip(X.to(device), y.to(device))]

        # 遍历每个批次的数据
        for x_batch, y_batch in zip(X.to(device), y.to(device)):  # tqdm(data):#.to(device)  .to(device)
            # 清空梯度缓存
            optimizer.zero_grad()

            count = count + 1
            if count % 2000 == 0:
                print("count : " + str(count))

            x_batch = x_batch.view(-1, batch_size, seq_length)

            # 前向传播得到预测结果，形状是(batch_size, vocab_size)
            output = model(x_batch)

            output = output.transpose(0, 1)
            # print("type output :" + str(type(output)))
            # print("y_batch :" + str(type(y_batch)))

            # 计算损失值，并累加到总损失中
            loss = criterion(output, y_batch)
            total_loss += loss.item()

            # 反向传播计算梯度，并更新参数
            loss.backward()
            optimizer.step()

        # 如果达到打印间隔，则打印当前轮数和平均损失值
        if epoch % print_every == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(X)}")
            text = X[np.random.randint(800), :, :]
            input_text = []
            for i in range(batch_size):
                a = text[i, :]
                input_text.append(gensim_model.wv.most_similar([a.numpy().astype(np.float32)], topn=1)[0][0])
            input_text = ''.join(input_text)
            print(input_text)
            file.write("start_txt : " + input_text + "\n")

            text_generated = generate_text(model, input_text, 50, seq_length)
            print("Epoch : " + str(epoch) + " total_loss : " + str(total_loss / len(X)) + ' ' + ''.join(text_generated))
            file.write("Epoch : " + str(epoch) + " total_loss : " + str(total_loss / len(X)) + ' ' + ''.join(
                text_generated) + '\n')
            torch.save(model, "E:/Anaconda/envs/yolov5-again/NLP_HW4/代码整理/exps/LSTM_" + str(epoch) + ".pth")

# start_text = "那人仰天长叹说道"#随手在琴弦上弹了几下短音
# token = [i for i in start_text]
