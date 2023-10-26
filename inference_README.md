Use this blueprint to deploy a document-classifier model and its API endpoint. To use this pretrained document-classifier model, create a ready-to-use API-endpoint that is quickly integrated with your input data in the form of raw text along with custom label names, returning for each label an associated probability value.

This inference blueprint’s model was trained using [Hugging Face multi_nli datasets](https://huggingface.co/datasets/multi_nli). To use custom document data according to your specific business, run this counterpart’s [training blueprint](https://metacloud.cloud.cnvrg.io/marketplace/blueprints/document-classification-train), which trains the model and establishes an endpoint based on the newly trained model.

Complete the following steps to deploy this document-classifier endpoint:
1. Click the **Use Blueprint** button.
2. In the dialog, select the relevant compute to deploy the API endpoint and click the **Start** button.
3. The cnvrg software redirects to your endpoint. Complete one or both of the following options:
   - Use the Try it Live section with any document file or link to be classified.
   - Use the bottom integration panel to integrate your API with your code by copying in the code snippet.

An API endpoint that classifies documents has now been deployed. To learn how this blueprint was created, click [here](https://github.com/cnvrg/document-classification).

## Example Input
Text:   
```
I love making new dishes for my family.
```  
Labels: 
```
Cooking,Dancing
```
## Example Output

```
{
  "cooking": "0.9635761380195618",
  "flying": "0.0028416530694812536"
}
```