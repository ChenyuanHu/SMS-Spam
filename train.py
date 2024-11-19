import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import tiktoken


class SMSSpamDataset(Dataset):

    def __init__(self, file_path, enc):
        self.data = pd.read_csv(file_path, encoding='latin1')
        self.enc = enc
        self.block_size = 512
        self.vocab_size = 50257

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data.iloc[index]["v2"]
        x = self.enc.encode(x)
        if len(x) < self.block_size:
            x = x + [self.enc.eot_token] * (self.block_size - len(x))
        else:
            print("max len:", len(x))
            x = x[:self.block_size]

        x = torch.tensor(x, dtype=torch.long)

        y = self.data.iloc[index]["v1"]
        y = torch.tensor(1, dtype=torch.long) if y == "spam" else torch.tensor(0, dtype=torch.long)
        return x, y


enc = tiktoken.get_encoding("gpt2")
dataset = SMSSpamDataset("spam.csv", enc)

print(len(dataset))

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
dataset = None


def count_samples(dataset):
    spam_count = 0
    ham_count = 0
    for _, label in dataset:
        if label == 1:
            spam_count += 1
        else:
            ham_count += 1
    return spam_count, ham_count

# 统计训练集和验证集中的正负样本数
spam_count, ham_count = count_samples(train_dataset)
print(f"Total samples: {len(train_dataset)}")
print(f"Spam samples: {spam_count}")
print(f"Ham samples: {ham_count}")


def augment_negative_samples(dataset, augment_factor=2):
    spam_samples = [data for data in dataset if data[1] == 1]  # 获取所有 spam 样本
    augmented_spam_samples = spam_samples * augment_factor  # 扩充负样本
    return ConcatDataset([dataset, augmented_spam_samples])  # 合并原始数据集和扩充的负样本

# 扩充数据集
train_dataset = augment_negative_samples(train_dataset, augment_factor=4)

# 重新统计样本数
spam_count, ham_count = count_samples(train_dataset)
print(f"After augmentation:")
print(f"Total samples: {len(train_dataset)}")
print(f"Spam samples: {spam_count}")
print(f"Ham samples: {ham_count}")


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


class ModuleConfig:
    block_size: int = 512
    vocab_size: int = 50257
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128

config = ModuleConfig()


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # manual implementation of attention
        # this materializes the large (T,T) matrix for all the queries and keys
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = NewGELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MyModule(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
            l_f = nn.Linear(config.n_embd, 2),
        ))
        self.config = config
        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 if not hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') else 0.02/math.sqrt(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(self, idx, targets=None):
        device = idx.device
        _, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = x.mean(dim=1) # 时间维度池化: 使用 x.mean(dim=1) 将 (batch_size, block_size, n_embd) 压缩为 (batch_size, n_embd)，提取序列全局信息。
        x = self.transformer.l_f(x)
        logits = x

        loss = None
        if targets is not None:
            loss = F.cross_entropy(x, targets)

        return logits, loss


def evaluate_model(model, val_loader, device):
    """
    在验证集上评估模型性能，针对 Spam 和 Ham 分别计算 Precision, Recall, F1-score，并计算总体 Accuracy。
    Args:
        model (nn.Module): 要评估的模型。
        val_loader (DataLoader): 验证集数据加载器。
        device (str): 使用的设备（'cpu' 或 'cuda'）。
    Returns:
        metrics (dict): 包含验证集的 7 个指标：Accuracy, Precision, Recall, F1-score (Spam 和 Ham)。
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0

    # 初始化统计变量
    spam_tp, spam_fp, spam_fn = 0, 0, 0
    ham_tp, ham_fp, ham_fn = 0, 0, 0
    correct = 0
    total = 0

    with torch.no_grad():  # 关闭梯度计算
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            # 前向传播
            logits, loss = model(x, y)
            total_loss += loss.item()

            # 获取预测值和目标值
            predictions = torch.argmax(logits, dim=1)  # 预测类别
            targets = y

            # 更新分类统计
            for pred, target in zip(predictions, targets):
                if pred == target:
                    correct += 1
                if target == 1:  # Spam 类别
                    if pred == 1:
                        spam_tp += 1
                    else:
                        spam_fn += 1
                if target == 0:  # Ham 类别
                    if pred == 0:
                        ham_tp += 1
                    else:
                        ham_fn += 1
                if pred == 1 and target == 0:  # 误将 Ham 预测为 Spam
                    spam_fp += 1
                if pred == 0 and target == 1:  # 误将 Spam 预测为 Ham
                    ham_fp += 1

            total += targets.size(0)

    # 总体 Accuracy
    accuracy = correct / total

    # Spam 指标
    spam_precision = spam_tp / (spam_tp + spam_fp) if (spam_tp + spam_fp) > 0 else 0
    spam_recall = spam_tp / (spam_tp + spam_fn) if (spam_tp + spam_fn) > 0 else 0
    spam_f1 = 2 * spam_precision * spam_recall / (spam_precision + spam_recall) if (spam_precision + spam_recall) > 0 else 0

    # Ham 指标
    ham_precision = ham_tp / (ham_tp + ham_fp) if (ham_tp + ham_fp) > 0 else 0
    ham_recall = ham_tp / (ham_tp + ham_fn) if (ham_tp + ham_fn) > 0 else 0
    ham_f1 = 2 * ham_precision * ham_recall / (ham_precision + ham_recall) if (ham_precision + ham_recall) > 0 else 0

    # 平均损失
    avg_loss = total_loss / len(val_loader)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "spam_precision": spam_precision,
        "spam_recall": spam_recall,
        "spam_f1": spam_f1,
        "ham_precision": ham_precision,
        "ham_recall": ham_recall,
        "ham_f1": ham_f1
    }

    return metrics



# attempt to autodetect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")
model = MyModule(config)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

model.to(device)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    t0 = time.time()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    t1 = time.time()
    metrics = evaluate_model(model, val_loader, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], {(t1-t0):.2f} s, \n"
        f"Train Loss: {loss.item():.8f}, "
        f"Val Loss: {metrics['loss']:.8f}, "
        f"Accuracy: {metrics['accuracy']:.8f}, \n"
        f"Spam Precision: {metrics['spam_precision']:.4f}, "
        f"Spam Recall: {metrics['spam_recall']:.4f}, "
        f"Spam F1: {metrics['spam_f1']:.4f}, \n"
        f"Ham Precision: {metrics['ham_precision']:.4f}, "
        f"Ham Recall: {metrics['ham_recall']:.4f}, "
        f"Ham F1: {metrics['ham_f1']:.4f}")
