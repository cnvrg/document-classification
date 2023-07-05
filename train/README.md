# Document Classification Train
This library allows you to create and train a model specifically on your choice of text and keywords. For example You can teach the model to classify marketing docuemnts as 'marketing' and sales documents as 'sales' with high confidence by providing sufficient examples of marketing and sales text. You can extend this training to any text with any keywords. Each text shall be associated with one class. Based on the training data the model will learn to identify unique contextual relationships and words that are associated with each class and help you in future classification of text. You need as little as 8 examples per label to start training.

[![N|Solid](https://cnvrg.io/wp-content/uploads/2018/12/logo-dark.png)](https://nodesource.com/products/nsolid)

### Flow
- The user has to upload a single csv file containing examples of text and associated labels.
- The model is trained on the given dataset and a model file is produced. The model can then be used for a personalized business document classification.

The labels file or .csv file that will will have two columns. First column is called `text` and it will contain example sentences and second column is called `label` which will contain the corresponding label for the text provided in the first column.
For example:
| text | label |
| :---:   | :---: |
| this is an example text about sales | sales |
| this is an example text about marketing | marketing |
| ... | ... |


### Inputs
- `--train_data` path to .csv file used for training.
- `--eval_data` path to .csv file used for evaluation. The format of this file is same as the train file. If no file is provided, evaluation will not be performed.
- `--batch_size` default value is 4. This parameter is used to set the number of examples used at a single time to calculte loss and update weights of the model. Higher the number, better the results however, high number can lead to crashing due to memory constraints. If you are not getting any output after training, try reducing this number and restart the training.
- `--num_iterations` The number of text pairs to generate for contrastive learning. Default is set to 20.
- `--epochs` number of training iterations for the model. Default is set to 1, you can increase these if your tranining loss is high at the end of training.
- `--model_id` default is sentence-transformers/paraphrase-mpnet-base-v2 . The huggingface model id for the sentence transformer. If you want to use a specific sentence tranformer provide its' huggingface id and it will downloaded and used for training.

### Outputs 
A list of artifacts that contain the trained model, tokenizer and the label encoder. If you provided an eval file, a file called `confusion_matrix.png` will be created as well.

## How to run
```
python3 train.py --data <name of .csv file>  
```
Example:
```
python3 train.py -data 'data.csv'
```


# About SetFit
SetFit is an efficient and prompt-free framework for few-shot fine-tuning of Sentence Transformers. It achieves high accuracy with little labeled data - for instance, with only 8 labeled examples per class on the Customer Reviews sentiment dataset, SetFit is competitive with fine-tuning RoBERTa Large on the full training set of 3k examples.

# Reference
https://github.com/huggingface/setfit