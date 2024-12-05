# 对比模型用onnx和infiniTensor运行的结果

import torch
import torch.onnx
import onnx
import numpy
import sys
from onnx import ModelProto, ValueInfoProto
from pyinfinitensor.onnx import OnnxStub, backend
from onnxruntime import InferenceSession


def infer(model: ModelProto, input) -> dict:
    collection = set()
    for node in model.graph.node:
        for output in node.output:
            collection.add(output)
    model.graph.output.extend([ValueInfoProto(name=x) for x in collection])
    session = InferenceSession(model.SerializeToString())
    i = session.get_inputs()[0].name
    return dict(
        zip(
            [x.name for x in session.get_outputs()],
            [x.flatten() for x in session.run(None, {i: input})],
        )
    )


model0 = onnx.load(sys.argv[1])
#model0.eval()  # 设置为评估模式

# 定义固定的 batch_size 和 sequence_length
fixed_batch_size = 2
fixed_sequence_length = 256

input_data = torch.randn(fixed_batch_size, fixed_sequence_length)

print("gen_new_inputs", input_data)
# 导出模型，固定输入数据的维度
torch.onnx.export(model0, input_data, "simple_bge_m3.onnx", verbose=True)


#model1 = OnnxStub(model0, backend.cuda_runtime())

#model1.run()
#outputs = next(iter(model.outputs.values())).copyout_numpy()
#input_shape = [x.dim_value for x in model1.graph.input[0].type.tensor_type.shape.dim]
#input = numpy.random.random(input_shape).astype(numpy.float32)

#output0 = infer(model0, input)[model0.graph.output[0].name]
#output1 = infer(model1, input)[model1.graph.output[0].name]
#print("result", outputs)
print("OK")