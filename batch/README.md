# Document Classification
This library takes a directory path and tags/labels as input. It extracts text from supported documents in the given directory and assigns a probability value to each document for each tag. The probabilities for each document for all tags, add up to 1. The output is a csv file.
The method works by posing the sequence to be classified as the NLI premise and to construct a hypothesis from each candidate label. For example, if we want to evaluate whether a sequence belongs to the class "politics", we could construct a hypothesis of This text is about politics.. The probabilities for entailment and contradiction are then converted to label probabilities.

### Supported documents for text extraction
- .pdf
- .txt
- .doc
- .docx
  
*Please note that the files which are not supported will have empty values for probabilities for each label in the csv against their names.*

### The input directory structure looks like below
The user will provide the path to folder main which can contain multiple folder inside and each of these folders will have the targe files.
```
main
    |
    folder 1
        |
        file1
        file2
        ..
    folder 2
        |
        file 3
        file 4
        ..
    folder 3
        |
        file 5
        file 6
        ..
    ..
```
## Input Args

`--dir` : The path to the folder containing all folders inside which the targe files are stored.

`--labels` : The target labels provided as comma separate values without any space. Please note the labels have to be provided in quotes. (For example "label1,labe2")

## Input Command
```
cnvrg run  --datasets='[{id:{dataset_id},commit:{dataset_commit_id}]' --machine="{compute_size} --image=python:3.8.6 --sync_before=false python3 batch_predict.py --dir {path_to_dir} --labels {labels}
```

## Sample Input Command
```
cnvrg run  --datasets='[{id:"pdf2",commit:"827bfc458708f0b442009c9c9836f7e4b65557fb"}]' --machine="AWS-ON-DEMAND.xlarge-memory,AWS-SPOT.gpu-large,AWS-SPOT.gpu" --image=python:3.8.6 --sync_before=false python3 batch_predict.py --dir /data/pdf2 --labels "medicine,cooking"
```

## Sample Output

For directory structure that looks somthing like this:
```
main
    |
    first
        |
        one.pdf
    second
        |
        two.txt
```

We will have an output.csv that will look something like this:

| folder_name | document_name    | label_1   | probability_1   | label_2   | probability_2   |
| :---:   | :---: | :---: | :---: | :---: | :---: |
| first | one.pdf   | cooking   | 0.8 | cleaning | 0.2 |
| second | two.txt   | cooking   | 0.3 | cleaning | 0.7 |

### Reference
https://huggingface.co/facebook/bart-large-mnli








