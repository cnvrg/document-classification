Use this blueprint to train a custom model on textual content within a set of documents. This blueprint also establishes an endpoint that can be used to classify documents based on the newly trained model.

To train this model with your data, provide in S3 a `documents_dir` dataset directory with multiple subdirectories containing the different classes of documents. The blueprint supports document files in .pdf, .txt, .docx, and .doc formats. Other documents formats are skipped.

Complete the following steps to train the document-classifier model:
1. Click the **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. In the flow, click the **S3 Connector** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `bucketname` - Value: enter the data bucket name
     - Key: `prefix` - Value: provide the main path to the images folder
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
3. Return to the flow and click the **Train** task to display its dialog.
   * Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `documents_dir` − Value: provide the path to the directory including the S3 prefix in the following format: ` /input/s3_connector/dc_classification_train_data`
     - Key: `labels path` − Value: provide the path to the CSV file containing mapping of document names to their classes in the following format: `/input/s3_connector/dc_classification_train_data/file.csv`
     - Key: `epochs` − Value: provide the number of training iterations for the model 

     NOTE: You can use the prebuilt example data paths provided.
     
   * Click the **Advanced** tab to change resources to run the blueprint, as required.
4.	Click the **Run** button. The cnvrg software launches the training blueprint as set of experiments, generating a trained document-classifier model and deploying it as a new API endpoint.
5. Track the blueprint's real-time progress in its Experiment page, which displays artifacts such as logs, metrics, hyperparameters, and algorithms.
6. Click the **Serving** tab in the project, locate your endpoint, and complete one or both of the following options:
   - Use the Try it Live section with any document file or link to be classified.
   - Use the bottom integration panel to integrate your API with your code by copying in the code snippet.

A custom model and API endpoint, which can classify a document’s textual content, have now been trained and deployed. To learn how this blueprint was created, click [here](https://github.com/cnvrg/document-classification).