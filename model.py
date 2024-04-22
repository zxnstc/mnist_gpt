import torch
import torch.nn as nn
from torch.nn import functional as F
import struct
import numpy as np
import torchvision
import time
from transformers import optimization
from torch.utils.tensorboard import SummaryWriter

starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
writer = SummaryWriter(log_dir="./runs/"+starttime[:13],comment=starttime[:13],flush_secs=60)


# 超参数设置
# hyperparameters
batch_size = 256 # how many independent sequences will we process in parallel?
block_size = 28*28 # what is the maximum context length for predictions?
max_iters = 20000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 256
n_head = 4
n_layer = 10
dropout = 0.0
vocab_size=256
# ------------

torch.manual_seed(1337)

# 1. 数据处理
def load_mnist_image(file_name):
    with open(file_name, 'rb') as file:
        struct.unpack('>4i', file.read(16)) # 去除文件头16字节
        images = np.fromfile(file, dtype=np.uint8) # 将剩下的字节读入np数组
    return images.reshape(-1, 28*28)

# images (60000, 784) 已经把每张图片拉平成一个列表
images = load_mnist_image('/data/notebook/mnist_gpt/MNIST/raw/train-images-idx3-ubyte')
# print(images.shape)


# 2. 数据转化
# images (60000, 784)-->tensor (60000, 784)
# train_data:torch.Size([54000, 784])
data = torch.tensor(images, dtype=torch.long)
data=torch.reshape(data,(-1,))
# print(data.shape)
# torch.Size([47040000])
n = int(0.9*len(data)) # first 90% will be train, rest val 
# n=5400
train_data = data[:n]
val_data = data[n:-100]
# print(train_data.shape)
# torch.Size([42336000])



# 3. 数据分批
# data loading
def get_batch(split:str)-> torch.Tensor([batch_size,block_size]):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # 生成一个长度为batch_size的一维张量，张量中的每个元素都是一个随机整数，这个整数的范围是0到len(data) - block_size。这些随机整数将被用作从数据中提取序列的起始索引。
    # ix = torch.randint(len(data) - block_size, (batch_size,))
    ix = torch.randint(len(data) // 784-1, (batch_size,)) * 784
    # print("ix:",ix)
    x = torch.stack([data[i:i+block_size] for i in ix])          
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# x,y=get_batch('train')
# print(x.shape,y.shape)
# torch.Size([16, 784]) torch.Size([16, 784])


# 4. 模型评估
# 评估模型在训练集和验证集上的平均损失。这个函数的实现非常简单：它首先将模型设置为评估模式，然后对每个数据集进行eval_iters次迭代。
# 在每次迭代中，它获取一个小批量数据，然后计算模型的输出和损失。最后，它返回每个数据集的平均损失。
@torch.no_grad()
def estimate_loss(model):
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        # print(losses.shape)
        # torch.Size([200])
        # torch.Size([200])
        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits,loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out

# 5. 模型head定义
class Head(nn.Module):
    """ one head of self-attention """


    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # 创建一个下三角矩阵，并将其注册为模型的一个缓冲区。
        # 这个下三角矩阵将被用作self-attention的权重矩阵，它将确保模型只能在当前时间步之前的时间步上进行自注意力操作。
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

        self.dropout=nn.Dropout(dropout)


    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x) #(B,T,head_size)
        q=self.query(x) #(B,T,head_size)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) # (B, T, T)
        wei=F.softmax(wei,dim=-1) # (B, T, T)
        wei=self.dropout(wei)
        # perform the weighted aggregation of the values
        v=self.value(x)
        out=wei@v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


# 6. 模型MultiHeadAttention定义
# 实现多头自注意力机制。在这个机制中，我们并行地进行多次自注意力计算，然后将结果拼接起来，通过一个线性层和一个dropout层进行处理，得到最终的输出。
# 这种方法可以让模型在不同的表示子空间中学习输入的不同特征。
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    # __init__方法是类的构造函数，它接收两个参数：num_heads和head_size。num_heads是注意力头的数量，head_size是每个注意力头的大小。
    # 创建了一个nn.ModuleList，它包含了num_heads个Head对象。我们还定义了一个线性层self.proj和一个dropout层self.dropout。
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj=nn.Linear(n_embd,n_embd)
        self.dropout=nn.Dropout(dropout)

    # forward方法定义了前向传播的计算过程。首先，我们对每个注意力头h进行计算，然后将结果在最后一个维度上拼接起来，得到out。
    # 然后，我们将out输入到线性层和dropout层，得到最终的输出。
    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.dropout(self.proj(out))
        return out


# 7. 模型FeedFoward定义
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    # 在__init__方法中，我们首先调用了父类的构造函数，然后定义了一个神经网络self.net。这个神经网络是一个nn.Sequential对象，
    # 它包含了两个线性层和一个ReLU激活函数，以及一个dropout层。
    # 第一个线性层将输入的维度扩大到4 * n_embd，然后通过ReLU激活函数进行非线性变换，然后第二个线性层将维度缩小回n_embd，最后通过dropout层进行正则化。
    def __init__(self,n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self,x):
        return self.net(x)


# 8. 模型Block定义，layerNorm, multiheadattention, layerNorm, feedforward
# Block类的作用是实现一个Transformer模型中的一个块。这个块包含了一个多头自注意力模块和一个前馈神经网络模块，以及两个层归一化操作。
# 这种结构可以让模型在处理序列数据时，能够同时考虑到每个位置的信息和全局的信息。
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self,n_embd,n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        head_size=n_embd//n_head
        self.sa=MultiHeadAttention(n_head,head_size)
        self.ffwd=FeedFoward(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x=x+self.sa(self.ln1(x))
        x=x+self.ffwd(self.ln2(x))
        return x


# 9. 模型BigramLanguageModel定义
# super simple bigram model，训练一个二元语言模型，并生成新的文本。
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks=nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)

    # forward方法定义了前向传播的计算过程。首先，我们从词嵌入表和位置嵌入表中获取嵌入，将它们相加得到x。
    # 然后，我们将x输入到self.blocks中，进行层归一化，然后输入到self.lm_head中，得到logits。如果提供了目标，我们会计算交叉熵损失。
    def forward(self,idx,targets=None):
        # idx ：torch.Size([16, 32, 784]) 
        # targets：torch.Size([16, 32, 784])
        B,T=idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)

        x=self.blocks(x) # (B,T,C)
        x=self.ln_f(x) # (B,T,C)
        logits=self.lm_head(x)

        if targets is None:
                loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # 将当前的索引裁剪到最后的block_size个令牌，获取预测的logits，
    # 只关注最后一个时间步，应用softmax得到概率，从分布中采样，将采样的索引添加到运行的序列中。
    def generate(self,idx,max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond=idx[:,-block_size:]
            logits,loss=self(idx_cond)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx






if __name__ == "__main__":
    # 10. 模型实例化
    model = BigramLanguageModel()

    m = model.to(device)
    
    writer.add_graph(model,input_to_model=get_batch('train'))
    # print the number of parameters in the model
    # 将其移动到设备device上。然后，我们打印出模型中的参数数量。
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = optimization.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=max_iters,
    )

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        # eval_interval = 100
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.4g}")

            # write to tensorboard
            writer.add_scalar('Loss/train', losses['train'], iter)
            writer.add_scalar('Loss/val', losses['val'], iter)
            writer.add_scalars(main_tag='Loss/train and val',tag_scalar_dict={"train":losses['train'],"loss":losses['val']},global_step=iter)
            writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step=iter)

            

        # 从训练集中采样一个批次的数据xb和yb，将它们输入到模型中，得到logits和loss。然后将优化器的梯度清零，计算损失的反向传播，更新优化器的参数。
        # 采样一个批次的数据，计算损失，清零梯度，计算反向传播，然后更新参数。这是训练神经网络模型的基本步骤。
        # sample a batch of data
        xb, yb = get_batch('train')
        # evaluate the loss
        logits, loss = model(xb, yb)
        # 清零优化器的梯度。这是因为PyTorch的优化器在每次更新参数时都会累积梯度，所以在每次更新参数之前，我们需要清零梯度。
        optimizer.zero_grad(set_to_none=True)
        # 计算损失的反向传播。这会计算出每个参数的梯度。
        loss.backward()
        # 更新优化器的参数。这会根据每个参数的梯度和学习率来更新参数的值。
        optimizer.step()
        lr_scheduler.step()

    torch.save(model,'./model.pth')
    writer.close()