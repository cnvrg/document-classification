import argparse
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from train import run_training
from sklearn.preprocessing import LabelEncoder
import pickle

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")


def argument_parser():
    parser = argparse.ArgumentParser(description="""Training""")
    parser.add_argument(
        "--train_data",
        action="store",
        dest="train_data",
        required=True,
        help="""path to the csv file""",
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        dest="batch_size",
        required=False,
        type=int,
        default=16,
        help="""Number of examples to be used together at one time for training""",
    )
    parser.add_argument(
        "--num_iterations",
        action="store",
        dest="num_iterations",
        required=False,
        type=int,
        default=20,
        help="""The number of text pairs to generate for contrastive learning""",
    )
    parser.add_argument(
        "--epochs",
        action="store",
        dest="epochs",
        required=False,
        type=int,
        default=1,
        help="""The number of epochs to use for contrastive learning""",
    )
    parser.add_argument(
        "--model_id",
        action="store",
        dest="model_id",
        required=False,
        default="sentence-transformers/paraphrase-mpnet-base-v2",
        help="""The id of sentence transformer model to use from huggingface""",
    )
    parser.add_argument(
        "--test_data",
        action="store",
        dest="test_data",
        required=False,
        default=False,
        help="""Path to the evaluation dataset""",
    )

    return parser.parse_args()


def argument_validation(data):

    # check if the file exists
    assert os.path.exists(data), (
        " Path to the data file provided " + data + " does not exist "
    )

    # validate that the labels file is a csv file
    assert data.endswith(".csv"), " The data file provided " + data + " must be a csv"

    # check if the file is csv file with two columns text label
    df = pd.read_csv(data)
    assert set(df.columns) == {"text", "label"}, (
        "The data file provided "
        + data
        + " is expected to have two columns, text and label"
    )


def preprocess(args):

    # use label encoder to encode the labels

    le = LabelEncoder()
    train_data = pd.read_csv(args.train_data)
    train_data["label"] = le.fit_transform(train_data["label"])
    output = open(cnvrg_workdir + "/labelencoder.pkl", "wb")
    pickle.dump(le, output)
    output.close()

    if args.test_data:
        eval_data = pd.read_csv(args.test_data)
        eval_data["label"] = le.transform(eval_data["label"])
    else:
        eval_data = False
    return train_data, eval_data, le


def main():
    args = argument_parser()
    argument_validation(args.train_data)
    if args.test_data:
        argument_validation(args.test_data)
    ds = DatasetDict()
    train_data, eval_data, le = preprocess(args)
    train_dataset = Dataset.from_pandas(train_data)
    if args.test_data:
        eval_dataset = Dataset.from_pandas(eval_data)
        ds["eval"] = eval_dataset

    ds["train"] = train_dataset

    run_training(
        ds, args.batch_size, args.num_iterations, args.epochs, args.model_id, le
    )


if __name__ == "__main__":
    main()
