import pandas as pd
from extractor import text_extraction
import os
from text_breaker import breaker


def get_all_documents(root_path, storage):
    """
    This function takes a path to a root folder and returns a list of
    all files by recursively traversing all the folders inside the root folder.

    - Args: A valid path to a folder, an empty list 
    - Returns: List of all files with their paths.

    """
    for files in os.listdir(root_path):
        if os.path.isdir(os.path.join(root_path, files)):
            get_all_documents(os.path.join(root_path, files), storage)
        else:
            storage.append(os.path.join(root_path, files))
    return storage


def create_dataframe(documents, labels):
    """
    This function is used to create a dataframe with two columns, first column
    will contain the text extracted from input files and second column
    will contain the corresponding label present in labels file against
    the name of the file from which was text was extracted.

    Args:
        - documents is a list of documents with their complete paths for example
        /cnvrg/folder/document.pdf
        - labels is a csv file containing two columns, document and class where
        document column has the file names and class column has the label name.

    Returns:
        - A single dataframe with two columns text and class, text contains extracted
        text from the document and class contains the label name from the label.csv
    """
    main_df = pd.DataFrame(columns=["text", "class"])
    text_extract = text_extraction()
    for document in documents:
        text = text_extract.master_extractor(document)
        if text is not None:
            document_name = os.path.basename(document)
            print("")
            try:
                document_class = list(
                    labels["class"][labels["document"] == document_name]
                )[0]
            except IndexError:
                print(
                    f"Skipping file {document_name} because the class is not provided in the label file"
                )
                continue
            temp = {"text": text, "class": document_class}
            main_df = main_df.append(temp, ignore_index=True)
        else:
            continue

    return main_df


def expand_df(df):
    """
    This function takes a dataframe as input and breaks the single row into multiple rows
    such that text in no one row is longer than the transformer input limit, and for
    each breakup, the label remains the same. For example:
    text | class
    This... | medicine

    Here assume This... is longer than the transformer input limit, so we break up
    This... into multiple subparts and we create the following dataframe:
    text | class
    This .. | medicine
    .. continuation text | medicine

    Args:
     - A single dataframe containing two columns text and class columns

    Returns:
     - A single dataframe containing two columns text and class where every
     value under text column in shorter than the tranformer input limit.

     Transformer limit defined in the breakup
    """
    expanded_df = pd.DataFrame(columns=["text", "class"])
    breaking_up = breaker()
    # for each row get the text and the label
    for row in df.iterrows():
        text = row[1]["text"]
        label = row[1]["class"]

        broken_text = breaking_up.breakup(text)
        for each_breakup in broken_text:
            temp = {"text": each_breakup, "class": label}
            expanded_df = expanded_df.append(temp, ignore_index=True)

    return expanded_df


def preprocess(root_path, labels_file):
    document_list = []
    document_list = get_all_documents(root_path, document_list)
    labels_df = pd.read_csv(labels_file)
    extracted_text_df = create_dataframe(document_list, labels_df)
    df = expand_df(extracted_text_df)

    return df
