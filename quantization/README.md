# Document Classification Quantization
Quantization is a widely-used model compression technique that can reduce model size while also improving inference latency. The full precision data converts to low-precision, there is little degradation in model accuracy, but the inference performance of quantized model can gain higher performance by saving the memory bandwidth and accelerating computations with low precision instructions. Intel provides several lower precision instructions (ex: 8-bit or 16-bit multipliers), inference can get benefits from them. We use Intel neural compressor for post training dynamic quantization. The qunatization criteria is fixed as follows:

```
    model:
      name: speedup
      framework: onnxrt_integerops

    device: cpu

    quantization:
      approach: post_training_dynamic_quant

    tuning:
      accuracy_criterion:
        relative: 0.01
      exit_policy:
        timeout: 0
      random_seed: 9527
        """
```

[![N|Solid](https://cnvrg.io/wp-content/uploads/2018/12/logo-dark.png)](https://nodesource.com/products/nsolid)

### Calibration data

User will have to provide a .csv file that will will have two columns. First column is called `text` and it will contain example sentences and second column is called `label` which will contain the corresponding label for the text provided in the first column. This data is used to get the right calibration while quantization of various layers in order to achieve the target accuracy criteria.

For example:
| text | label |
| :---:   | :---: |
| this is an example text about sales | sales |
| this is an example text about marketing | marketing |
| ... | ... |


### Inputs
- `--onnx_model_folder` path to the onnx model you want to quantize.
- `--setfit_model_folder` path to the original setfit trained model. This is required to load the classification head and tokenizer.
- `--calibration_data` path to the calibration data file.

### Outputs 
Output will be an onnx model with tokenizer that can be used for inference.

## How to run
```
python3 quant.py --setfit_model_folder <path to the folder>  
```
Example:
```
python3 train.py --setfit_model_folder /cnvrg 
```

# About Intel Neural compressor
Intel® Neural Compressor aims to provide popular model compression techniques such as quantization, pruning (sparsity), distillation, and neural architecture search on mainstream frameworks such as TensorFlow, PyTorch, ONNX Runtime, and MXNet, as well as Intel extensions such as Intel Extension for TensorFlow and Intel Extension for PyTorch.

# Reference
https://github.com/intel/neural-compressor