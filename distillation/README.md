# Document Classification Distillation
This library allows you to distill your pretrained text classification model to a smaller model of your choosing. User needs to provide path to the pretrained teacher model (Which is the larger model) and provide the huggingface hub id or location of the smaller student model. The smaller model will learn to mimic the output distribution of the larger model. Once distillation is over and the smaller model will behave like the larger model attaining lareger model's accuracy while retaining the size of the smaller model.

[![N|Solid](https://cnvrg.io/wp-content/uploads/2018/12/logo-dark.png)](https://nodesource.com/products/nsolid)

### Training data
The training data consists of unlabelled examples of text from the same distribution as the examples of text used for text classification training and inference. Meaning if you want text classification done in legal documents, the unlabelled data should contain text examples from legal domain. The unlablled data is just a set of sentences. 1000+ examples of text are recommended. The input file should be a .csv file with the following shape.
For example:
| text | 
| :---:   | 
| this is an example text 1 | 
| this is an example text 2 | 
| ... | ... |


### Inputs
- `--teacher_model` path to the trained teacher model.
- `--student_model` path to the student model or the huggingface hub id of the student model. Default is **sentence-transformers/paraphrase-MiniLM-L3-v2**
- `--train_data` path to .csv file containing unlabelled training data.
- `--test_data` path to .csv file used for evaluation. This file should contain two columns, **text** and **label** where the second column is the ground truth values.
- `--batch_size` default value is **16**. This parameter is used to set the number of examples used at a single time to calculte loss and update weights of the model. Higher the number, better the results however, high number can lead to crashing due to memory constraints. If you are not getting any output after training, try reducing this number and restart the training.
- `--num_iterations` The number of text pairs to generate for contrastive learning. Default is set to **20**.
- `--epochs` number of training iterations for the model. Default is set to **1**, you can increase these if your tranining loss is high at the end of training.


### Outputs 
Output will contain the distilled student model along with the labelencoder.pkl which you can use to decode the model output to label names.

## How to run
```
python3 distill.py --train_data <name of .csv file>  
```
Example:
```
python3 distill.py -train_data 'data.csv'
```

# Reference
https://github.com/huggingface/setfit