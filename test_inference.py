import sys
import onnx
import torch
import numpy as np
from pyinfinitensor.onnx import OnnxStub, backend
from onnxruntime import InferenceSession

def initInputs(model):
    inputs = []
    for input in model.graph.input:
        shape = [(dim.dim_value if dim.dim_value > 0 else 1) for dim in input.type.tensor_type.shape.dim]
        print(shape)
        data_type = input.type.tensor_type.elem_type
        input = np.random.random(shape).astype(toNumpyType(data_type))
        inputs.append(input)

    return inputs


def toNumpyType(typecode: int):
    if typecode == 1:
        return np.float32
    elif typecode == 2:
        return np.uint8
    elif typecode == 3:
        return np.int8
    elif typecode == 4:
        return np.uint16
    elif typecode == 5:
        return np.int16
    elif typecode == 6:
        return np.int32
    elif typecode == 7:
        return np.int64
    elif typecode == 8:
        return np.string_
    elif typecode == 9:
        return np.bool_
    elif typecode == 10:
        return np.float16
    elif typecode == 11:
        return np.double
    elif typecode == 12:
        return np.uint32
    elif typecode == 13:
        return np.uint64
    elif typecode == 14:
        return np.uint16
    else:
        raise RuntimeError("Unsupported data type.")

if __name__ == '__main__':
    args = sys.argv
    if len(sys.argv) != 2:
        print("Usage: python onnx_inference.py model_name.onnx")
        exit()
    model_path = sys.argv[1]
    # print(model_path)

    onnx_model = onnx.load(model_path)
    input_data = initInputs(onnx_model)

    model = OnnxStub(onnx_model, backend.cuda_runtime())
    for idata, itensor in zip(input_data, model.inputs.items()):
        itensor[1].copyin_numpy(idata)
    model.run()
    outputs = [output.copyout_numpy() for output in model.outputs.values()]
    print(outputs)
