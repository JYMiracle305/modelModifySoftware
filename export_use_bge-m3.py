## 下载bge-m3模型，并进行推理

import os

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
input_text = "这是一个示例输入"  # 您可以更改为所需的输入
inputs = tokenizer(input_text, return_tensors="pt")

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
            'input_ids': {0: 'batch_size'},        # 动态 batch size
            'attention_mask': {0: 'batch_size'},   # 动态 batch size
            'output': {0: 'batch_size'}             # 动态 batch size
            }
        )

print(f"Model exported to {onnx_filename}")


import onnxruntime as ort
from transformers import AutoTokenizer

# 加载 BGE-M3 的分词器
model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载 ONNX 模型
onnx_session = ort.InferenceSession("bge_m3_model.onnx")

# 准备输入数据
input_text = "今天天氣不錯"  # 您可以更改为所需的输入
inputs = tokenizer(input_text, return_tensors="np", padding='max_length', max_length=7, truncation=True)

# 进行推理，确保输入数据类型与导出时的对应
onnx_inputs = {
    'input_ids': inputs['input_ids'],
    'attention_mask': inputs['attention_mask']
}
output = onnx_session.run(['output'], onnx_inputs)

print("ONNX model output:", output)