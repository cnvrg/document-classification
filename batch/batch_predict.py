from extractor import master_extractor
import os
import argparse
import traceback
import pandas as pd
import itertools
import numpy as np
from predictor import setup_model
import json

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")


# argument parsing
def argument_parser():
    parser = argparse.ArgumentParser(description="""Creator""")
    parser.add_argument(
        "--trained_model",
        action="store",
        dest="trained_model",
        required=False,
        help="""path to the trained model""",
    )
    parser.add_argument(
        "--trained_classes",
        action="store",
        dest="trained_classes",
        required=False,
        help="""path to the classes.json file""",
    )
    parser.add_argument(
        "--dir",
        action="store",
        dest="dir",
        required=True,
        help="""directory containing all pdf files""",
    )
    parser.add_argument(
        "--labels",
        action="store",
        dest="labels",
        required=False,
        help="""Label names you want to associate the files with""",
    )
    return parser.parse_args()


def validation(args):
    """
    check if the pdf directory provided is a valid path if not raise an exception
    check if the user wants to use pretrained model or the trained model and check if the
    paths provided are valid

    Arguments
    - argument parser

    Raises
    - An assertion error if the path provided is not a valid directory
    """
    assert os.path.exists(args.dir), " Path to the files provided does not exist "
    if args.trained_model is not None:
        assert os.path.exists(
            args.trained_model
        ), "Path provided to the trained model file does not exist"
    if args.trained_classes is not None:
        assert os.path.exists(
            args.trained_classes
        ), "Path provided to the trained classes json file does not exist"


def merge(dict1, dict2):
    """
    Takes two dictionaries as input, merges them and returns a new dictionary
    """
    res = {**dict1, **dict2}
    return res


def map_dict(dict1, dict2):
    """
    Takes two dictionaries as input and assigns keys and values of second
    dictionary as values to the new dictionary.
    The new dictionary has keys as the values and keys first dictionary.
    """
    new_dict = {}
    for val1, val2 in zip(dict1.items(), dict2.items()):
        new_dict[val1[0]] = val2[0]
        new_dict[val1[1]] = val2[1]
    return new_dict

def get_all_documents(root_path, storage):
    """
    This function takes a path to a root folder and returns a list of
    all files by recursively traversing all the folders inside the root folder.

    - Args: A valid path to a folder, an emoty list 
    - Returns: List of all files with their paths.

    """
    for files in os.listdir(root_path):
        if os.path.isdir(os.path.join(root_path, files)):
            get_all_documents(os.path.join(root_path, files), storage)
        else:
            storage.append(os.path.join(root_path, files))
    return storage

def main():
    # command line execution
    args = argument_parser()
    validation(args)
    trained = False

    if args.trained_model is not None:
        predictions = setup_model(args.trained_model, args.trained_classes)
        trained = True
        class_file = open(args.trained_classes)
        label_names = json.load(class_file)
    else:
        predictions = setup_model()
        label_names = args.labels.split(",")
    direc = args.dir

    # check if the path provided is a valid directory
    labels_dict = {}
    for i, label in enumerate(label_names):
        labels_dict["label_" + str(i + 1)] = "probability_" + str(i + 1)

    to_write = pd.DataFrame(
        columns=["folder_name", "document_name"]
        + list(itertools.chain(*labels_dict.items()))
    )

    allfiles = []
    allfiles = get_all_documents(direc, allfiles)

    for required_files in allfiles:
        head, tail = os.path.split(os.path.dirname(required_files))
        meta = {
            "folder_name": tail,
            "document_name": os.path.basename(required_files),
        }
        try:
            text = master_extractor(required_files)
        except Exception:
            print(
                "While extracting text from file: ",
                meta["document_name"],
                " following error occurred",
            )
            print(traceback.format_exc())
            continue
        if text is not None and trained == False:
            prediction = predictions.predict(
                {"context": text, "labels": label_names}
            )
            mapped_values = map_dict(labels_dict, prediction)
            to_write = to_write.append(
                merge(meta, mapped_values), ignore_index=True
            )
        elif text is not None and trained == True:
            prediction = predictions.predict({"text": text})
            mapped_values = map_dict(labels_dict, prediction)
            to_write = to_write.append(
                merge(meta, mapped_values), ignore_index=True
            )
        else:
            prediction = {label: np.nan for label in label_names}
            mapped_values = map_dict(labels_dict, prediction)
            to_write = to_write.append(
                merge(meta, mapped_values), ignore_index=True
            )
    # to_write.to_csv(cnvrg_workdir + "/result.csv", index=False)
    to_write.to_csv("result.csv", index=False)


if __name__ == "__main__":
    main()
