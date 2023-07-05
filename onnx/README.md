# Document Classification ONNX
This library allows you to convert your setfit trained models to ONNX models. This means that if your model was pytorch or tensorflow based it can be converted to ONNX to achieve speedup on CPUs.

[![N|Solid](https://cnvrg.io/wp-content/uploads/2018/12/logo-dark.png)](https://nodesource.com/products/nsolid)


### Inputs
- `--model_folder` path to the model you want to convert to ONNX.


### Outputs 
A list of artifacts that contain the converted model, tokenizer and the label encoder. 

## How to run
```
python3 conv_to_onnx.py --model_folder <name of .csv file>  
```
Example:
```
python3 conv_to_onnx.py --model_folder /cnvrg  
```

