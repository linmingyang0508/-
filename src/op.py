import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, TrainingArguments, Trainer
# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=2048, dropout=dropout)
        self.encoder = nn.Linear(input_size, d_model)
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, src, tgt):
        src = self.encoder(src)
        tgt = self.encoder(tgt)
        output = self.transformer(src, tgt)
        output = self.decoder(output)
        return output

# 构建数据集和数据加载器
class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, audio_files, transcripts):
        self.audio_files = audio_files
        self.transcripts = transcripts

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        transcript = self.transcripts[idx]
        waveform, sample_rate = torchaudio.load(audio_file)
        return waveform, transcript

# 定义训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 定义评估函数
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, target)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# 设置超参数
input_size = 40  # 输入特征大小
output_size = 256  # 输出标签大小
d_model = 512  # 模型维度
nhead = 8  # 多头注意力头数
num_layers = 6  # Transformer层数
dropout = 0.1  # Dropout概率
batch_size = 32
learning_rate = 0.001
num_epochs = 10



# 加载数据集
data_files = {"train": r"C:\Users\ACER\Desktop\vhf_transformer11\data\processed_data\dataset.json"}
dataset_split_not_found_err_msg = "Error: The specified dataset split was not found in the data files."
dataset = load_dataset(
    "json",
    data_files=data_files,
    keep_in_memory=True
)
# 初始化优化器
trainer = Trainer(
   
    data_collator=lambda x: x,

)
# 实例化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(input_size, output_size, d_model, nhead, num_layers, dropout).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    train_loss = train(model, trainer.train(), criterion, optimizer, device)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f},')
# 保存训练好的模型和处理器配置
trainer.save_model(r"C:\Users\ACER\Desktop\vhf_transformer12\models\trained_models\wav2vec2-asr-chinese")
