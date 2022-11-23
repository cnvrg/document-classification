# Train
You can use this blueprint to train a tailored model that classify a text according to the custom data used in training. In order to train this model with your data, you would need to provide a set of documents and their class.
1. Click on the Use Blueprint button
2. You will be redirected to your blueprint flow page
3. In the flow, edit the following tasks to provide your data:
In the Train task: * Under the documents_dir and label parameters provide the path to the dataset
4. Click on the 'Run Flow' button
5. In a few minutes you will train a new intent detection model and deploy as a new API endpoint.
6. Go to the 'Serving' tab in the project and look for your endpoint
7. You can use the Try it Live section with any text to infer the intent.
8. You can also integrate your API with your code using the integration panel at the bottom of the page

Congrats! You have trained and deployed a custom model that can classify the text!














