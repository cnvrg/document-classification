# Document Classification
This library deploys the trained and/or optimised text classification model. User can will be able to deploy the model and receive results in JSON format.

### Input Types:
User can choose to provide text for classification, a URL pointing to a .pdf or .txt for example "hhtp://xyz.com/resources/file.pdf", the url should end in .pdf or .txt. User can also provide a public url for a .pdf or .txt file stored in the google drive.
User can directly upload base64 encoded files to endpoint to recieve an output. Supported file types for direct upload are:
- .pdf
- .txt
- .doc
- .docx

Input text needs to be provided with key **context** which can be a file, simple text or a url. 

In the try live section, enter the text you want to classify once the endpoint is deployed. On the right side user will get labels along with probabilities for each of the label.

### Sample Input Command

curl -X POST \
    http://dcinference1-4-1.aks-cicd-Grazitti-8766.cnvrg.io/api/v1/endpoints/zavjmsjnkxsrrspnaswa \
-H 'Cnvrg-Api-Key: eQ7sjdoFyLpCPFstmRCYsFKN' \
-H 'Content-Type: application/json' \
-d '{"context": "text you want to classify"}

### Sample Response
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










