You can deploy the Document classification model to use it via API calls. Once deployed the model will take raw text, pdf, txt, doc, docx files or a url pointing directely to a file. User can also provide a public url for a .pdf or .txt file stored in the google drive. You can provide custom label names and get for each label a probability value associated with it. This blueprint supports one click deployment. Follow the below steps to get started.

1. Click on `Use Blueprint` button
2. In the pop up, choose the relevant compute you want to use to deploy your API endpoint
3. You will be redirected to your endpoint
4. You can now use the `Try it Live` section with any text or link. 
5. You can now integrate your API with your code using the integration panel at the bottom of the page
6. You will now have a functioning API endpoint that returns the probabilites for each label for the input text!

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