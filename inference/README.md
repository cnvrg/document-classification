# Document Classification
This model takes text and set of tags as input. It assigns a probability value to each tag based on how appropriate the tag is to the given input text. The method works by posing the sequence to be classified as the NLI premise and to construct a hypothesis from each candidate label. For example, if we want to evaluate whether a sequence belongs to the class "politics", we could construct a hypothesis of This text is about politics.. The probabilities for entailment and contradiction are then converted to label probabilities.

An example json response for some input text looks like below:
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








