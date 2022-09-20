from extractor import master_extractor
from dc_inference import predict
import os
import argparse
import traceback
import pandas as pd
import itertools
import numpy as np

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")


# argument parsing
def argument_parser():
    parser = argparse.ArgumentParser(description="""Creator""")
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
        required=True,
        help="""Label names you want to associate the files with""",
    )
    return parser.parse_args()


def validation(args):
    """
    check if the pdf directory provided is a valid path if not raise an exception

    Arguments
    - argument parser

    Raises
    - An assertion error if the path provided is not a valid directory
    """
    assert os.path.exists(args.dir), " Path to the files provided does not exist "


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


def main():
    # command line execution
    args = argument_parser()
    direc = args.dir
    label_names = args.labels.split(",")
    # check if the path provided is a valid directory
    labels_dict = {}
    for i, label in enumerate(label_names):
        labels_dict["label_" + str(i + 1)] = "probability_" + str(i + 1)

    validation(args)

    to_write = pd.DataFrame(
        columns=["folder_name", "document_name"]
        + list(itertools.chain(*labels_dict.items()))
    )

    allfolders = []

    # traverse all the folders in the given directory
    for folder in os.listdir(direc):
        if os.path.isdir(os.path.join(direc, folder)):
            allfolders.append(os.path.join(direc, folder))

    # traverse each folder for files
    for folder in allfolders:
        for required_files in os.listdir(folder):
            if not os.path.isdir(required_files):  # check if it is not a dir
                meta = {
                    "folder_name": os.path.basename(folder),
                    "document_name": required_files,
                }
                try:
                    text = master_extractor(os.path.join(folder, required_files))
                except Exception:
                    print(
                        "While extracting text from file: ",
                        meta["document_name"],
                        " following error occurred",
                    )
                    print(traceback.format_exc())
                    continue
                if text is not None:
                    prediction = predict({"context": text, "labels": label_names})
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
    to_write.to_csv(cnvrg_workdir + "/result.csv", index=False)


if __name__ == "__main__":
    main()
