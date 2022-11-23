# Document Classification Train
This library allows you to create and train a model specifically on your choice of text and keywords. For example You can teach the model to classify marketing docuemnts as 'marketing' and sales documents as 'sales' with high confidence by providing sufficient examples of marketing and sales text. You can extend this training to any text with any keywords. Each text shall be associated with one class. Based on the training data the model will learn to identify unique contextual relationships and words that are associated with each class and help you in future organization of text.

[![N|Solid](https://cnvrg.io/wp-content/uploads/2018/12/logo-dark.png)](https://nodesource.com/products/nsolid)

### Flow
- The user has to upload the training dataset which is a collection of documents and their mappings to their class. The dataset should be in the format of a folder containing all the relevant documents you want to use for training, these documents can be further bifurcated into more folders inside the main folder. Also, a single .csv file containing names of the documents and their mappings to a single class.
- The model is trained on the given dataset and a model file is produced. The model can then be used for a personalized business document classification.

The labels file or .csv file that will contain the names of the documents their classes/labels will have two columns. First column is called `document` and it will names of all documents present in the training folder and second column is called `class`, it will contain the labels you want
your model to learn so that future documents can be associated with them with certain level of confidence. For example:
| document | class |
| :---:   | :---: |
| materials.pdf | sales |
| meeting_notes.pdf | marketing |
| brochure.docx | marketing |

*Supported formats for documents are:*
- .pdf
- .txt
- .doc
- .docx
  
Documents that have other formats will be skipped

### Inputs
- `--documents_dir` path to the folder containing the training documents which will be used for training.
- `--labels` path to the .csv file containing mapping of documents name to their classes.
- `--epochs` number of training iterations for the model. Default is set to 200, you can increase these if your tranining loss is high at the end of training.

### Outputs 
- `model.pt` refers to the file which contains the retrained model. This can be used in the future for detecting the intent of a costumer's message.
- `classes.json` refers to the file containing all the unique classes found in the training dataset.
   
## How to run
```
python3 train.py -data <name of data file>
```
Example:
```
python3 train.py -data 'data.csv'
```


# About BERT
BERT is a method of pre-training language representations, meaning that we train a general-purpose "language understanding" model on a large text corpus (like Wikipedia), and then use that model for downstream NLP tasks that we care about (like question answering). BERT outperforms previous methods because it is the first unsupervised, deeply bidirectional system for pre-training NLP.

# Reference
https://github.com/google-research/bert