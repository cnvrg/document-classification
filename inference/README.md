# Document Classification
This model takes text and set of tags as input. It assigns a probability value to each tag based on how appropriate the tag is to the given input text. The method works by posing the sequence to be classified as the NLI premise and to construct a hypothesis from each candidate label. For example, if we want to evaluate whether a sequence belongs to the class "politics", we could construct a hypothesis of This text is about politics.. The probabilities for entailment and contradiction are then converted to label probabilities.

### Input Types:
User can choose to provide text for classification, a URL pointing to a .pdf or .txt for example "hhtp://xyz.com/resources/file.pdf", the url should end in .pdf or .txt. User can also provide a public url for a .pdf or .txt file stored in the google drive.
User can directly upload base64 encoded files to endpoint to recieve an output. Supported file types for direct upload are:
- .pdf
- .txt
- .doc
- .docx

Input text needs to be provided with key **context** which can be a file, simple text or a url. Along with this user needs to provide labels with key **labels**.

In the try live section, the first input field is for text or url and second input field is for labels which need to be provided together separated by commas.

### Input Command

curl -X POST \
    http://dcinference1-4-1.aks-cicd-Grazitti-8766.cnvrg.io/api/v1/endpoints/zavjmsjnkxsrrspnaswa \
-H 'Cnvrg-Api-Key: eQ7sjdoFyLpCPFstmRCYsFKN' \
-H 'Content-Type: application/json' \
-d '{"context": "text you want to classify", "labels": ["tag1", "tag2", "tag3"]}
### Response
```
{
    "prediction":
    {
      'tag 1' : 0.91,
      'tag 2' : 0.08,
      'tag 3' : 0.01
    }
}
```

### Reference
https://huggingface.co/facebook/bart-large-mnli








