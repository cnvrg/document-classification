# Document Classification Inference
The response from the endpoint will contain the class with the highest confidence and the confidence value for the text used for inference.
User can choose to provide text for classification, a URL pointing to a .pdf or .txt for example "hhtp://xyz.com/resources/file.pdf", the url should end in .pdf or .txt. User can also provide a public url for a .pdf or .txt file stored in the google drive.
User can directly upload base64 encoded files to endpoint to recieve an output. Supported file types for direct upload are:
- .pdf
- .txt
- .doc
- .docx

Input needs to be provided with key **context**

### Sample Input Command

```
curl -X POST \
    {link to your deployed endpoint} \
-H 'Cnvrg-Api-Key: {your_api_key}' \
-H 'Content-Type: application/json' \
-d '{"context": "Apart from counting words and characters, our online editor can help you to improve word choice and writing style, and, optionally, help you to detect grammar mistakes and plagiarism. To check word count, simply place your cursor into the text box above and start typing. You'll see the number of characters and words increase or decrease as you type, delete, and edit them."}'
``` 

An example json response from the endpoint is given below:
```
{
    "prediction":
    [
        {
            'android': '0.79'
            'ios': '0.21'
            
        }
    ]
}
```