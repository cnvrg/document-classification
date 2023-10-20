Use this blueprint to classify text from document files in .pdf, .txt, .docx, and .doc formats. The blueprint skips other documents formats.
The blueprint’s input files are placed in a directory and its output is stored in CSV format. Provide the path containing folders with the specified-formatted document files. Also, provide as CSV the tags or labels to associate with these files. The blueprint outputs a `result.csv` in including each tag’s probability for each file.

Complete the following steps to run the document-classifier model in batch mode:
1. Click **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. Click the **S3 Connector** task to display its dialog.
   - Within the **Parameters** tab, provide the following Key-Value information:
     - Key: `bucketname` − Value: provide the data bucket name
     - Key: `prefix` − Value: provide the main path to the files folders
   - Click the **Advanced** tab to change resources to run the blueprint, as required.
3. Click the **Batch-Predict** task to display its dialog.
   - Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `--dir` − Value: provide the S3 path to the folder containing the folders storing the target files in the following format: `/input/s3_connector/dc_classification_data/`
     - Key: `--labels` − Value: provide the target labels in quotes as CSVs without spaces
 
     NOTE: You can use prebuilt data example paths provided.

   - Click the **Advanced** tab to change resources to run the blueprint, as required.
4. Click the **Run** button. The cnvrg software deploys a document-classifier model that classifies text in a batch of files and outputs a CSV file with the document classifications.
5. Select **Batch Predict > Experiments > Artifacts** and locate the output CSV file.
6. Click the **result.csv** File Name, click the Menu icon, and select **Open File** to view the output CSV file.

A custom model that classifies text in different formatted document files has been deployed in batch mode. To learn how this blueprint was created, click [here](https://github.com/cnvrg/document-classification).