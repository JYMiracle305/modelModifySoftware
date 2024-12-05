## 导出bge-m3模型为onnx

import os

## 修改hf的地址为镜像地址
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModel, AutoTokenizer

# 加载 BGE-M3 模型和分词器
model_name = "BAAI/bge-m3"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 设置模型为评估模式
model.eval()

# 创建示例输入
input_text = "测试模型的输入"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)

# 确保输入在 GPU/CPU 上与模型相同
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# 导出为 ONNX 文件
onnx_filename = "bge_m3_model.onnx"

torch.onnx.export(
        model,
        (input_ids, attention_mask),  # 模型输入
        onnx_filename,                 # 输出 ONNX 文件名
        export_params=True,
        opset_version=12,             # 使用较新的 ONNX opset 版本
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],  # 输入名
        output_names=['output'],                     # 输出名
        dynamic_axes={
            'input_ids': {0: 32},  # 0 是 batch_size 的轴索引
            'attention_mask': {0: 32},  # 0 是 batch_size 的轴索引
            'output': {0: 32}  # 假设输出的 batch_size 也是动态的
        }
)

print(f"Model exported to {onnx_filename}")