# Train
You can use this blueprint to train and optimize a tailored model that classify a text according to the custom data used in training. In order to train this model with your data, you would need to provide a set of documents and their class. We are using few shot learning framework called [Setfit](https://github.com/huggingface/setfit) which means you only need to provide as little as 8 examples per label.
1. Click on the Use Blueprint button
2. You will be redirected to your blueprint flow page
3. In the flow, edit the following tasks to provide your data:

In the Training task: * Under the train_data provide the path to the .csv file containing labelled examples.

[Optional]

In the distillation task: * Under the train_data provide the path to the .csv file containing unlabelled examples. You can deleted this task if you don't have these examples.

In the quantization task: * Under the calibration_data provide the path to the .csv file containing labelled examples. You can delete this task if you don't have these examples.

4. Click on the 'Run Flow' button
5. In a few minutes you will train a new intent detection model and deploy as a new API endpoint.
6. Go to the 'Serving' tab in the project and look for your endpoint
7. You can use the Try it Live section with any text to infer the intent.
8. You can also integrate your API with your code using the integration panel at the bottom of the page

Congrats! You have trained and deployed a custom model that can classify the text!














