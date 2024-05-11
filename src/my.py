import os
import json

# 定义音频和文本文件的目录
audio_dir = r"C:\Users\ACER\Desktop\vhf_transformer11\data\audio"
label_dir = r"C:\Users\ACER\Desktop\vhf_transformer11\data\labels"

# 获取两个目录中的文件名
audio_files = sorted(os.listdir(audio_dir))
label_files = sorted(os.listdir(label_dir))

# 确保两个目录中的文件数量相同
assert len(audio_files) == len(label_files)

# 创建一个空的列表来保存数据
data = []

# 遍历每个文件
for audio_file, label_file in zip(audio_files, label_files):
    # 创建一个词典来保存音频文件和文本文件的路径
    item = {
        'audio': os.path.join(audio_dir, audio_file),
        'label': os.path.join(label_dir, label_file),
    }
    # 将词典添加到列表中
    data.append(item)

# 定义输出的JSON文件的路径
output_path = r"C:\Users\ACER\Desktop\vhf_transformer11\data\processed_data\dataset.json"

# 将数据写入JSON文件
with open(output_path, 'w') as f:
    json.dump(data, f)