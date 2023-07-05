from setfit import SetFitModel, DistillationSetFitTrainer
import argparse
from datasets import Dataset, DatasetDict
import pickle
import os
import pandas as pd
from sentence_transformers.losses import CosineSimilarityLoss
import shutil
cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")


def argument_parser():
    parser = argparse.ArgumentParser(description="""Training""")
    parser.add_argument(
        "--student_model",
        action="store",
        dest="student_model",
        required=False,
        default="""sentence-transformers/paraphrase-MiniLM-L3-v2""",
    )
    parser.add_argument(
        "--teacher_model",
        action="store",
        dest="teacher_model",
        required=True,
        help="""path to the trained teacher model""",
    )
    parser.add_argument(
        "--train_data",
        action="store",
        dest="train_data",
        required=True,
        help="""path to the csv file""",
    )
    parser.add_argument(
        "--test_data",
        action="store",
        dest="test_data",
        required=False,
        default=False,
        help="""Path to the evaluation dataset""",
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
    return parser.parse_args()


def argument_validation(data):

    # check if the file exists
    assert os.path.exists(data), (
        " Path to the data file provided " + data + " does not exist "
    )

    # validate that the labels file is a csv file
    assert data.endswith(".csv"), " The data file provided " + data + " must be a csv"


def load_models(args):
    student_model = SetFitModel.from_pretrained(args.student_model)
    teacher_model = SetFitModel.from_pretrained(args.teacher_model)
    return student_model, teacher_model


def distill(student_model, teacher_model, data, batch_size, num_iterations, epochs, le):
    #check if test dataset is provided
    if len(list(data.keys())) == 2:
        student_trainer = DistillationSetFitTrainer(
            teacher_model=teacher_model,
            train_dataset=data["train"],
            student_model=student_model,
            eval_dataset=data["eval"],
            loss_class=CosineSimilarityLoss,
            metric="accuracy",
            batch_size=16,
            num_iterations=20,
            num_epochs=1,
        )
        student_trainer.train()
        eval(student_trainer)
    else:
        student_trainer = DistillationSetFitTrainer(
            teacher_model=teacher_model,
            train_dataset=data["train"],
            student_model=student_model,
            loss_class=CosineSimilarityLoss,
            metric="accuracy",
            batch_size=16,
            num_iterations=20,
            num_epochs=1,
        )
        student_trainer.train()

    MiniLM_L3_distilled_model = student_trainer.model
    MiniLM_L3_distilled_model._save_pretrained(save_directory=cnvrg_workdir)


def preprocess(args):

    # use label encoder to encode the labels
    train_data = pd.read_csv(args.train_data)
    pkl_file = open(args.teacher_model+'/labelencoder.pkl', 'rb')
    encoder = pickle.load(pkl_file) 
    try:
        train_data['label'] = encoder.transform(train_data['label']) 
    except:
        pass
    if args.test_data:

        eval_data = pd.read_csv(args.test_data)
        eval_data["label"] = encoder.transform(eval_data["label"])
    else:
        eval_data = False
    try:
        shutil.move(args.teacher_model+'/labelencoder.pkl',cnvrg_workdir)
    except:
        pass
    return train_data, eval_data, encoder


def eval(student_trainer):
    # Student Train and evaluate
    metrics = student_trainer.evaluate()
    print("Student results: ", metrics)


def main():
    args = argument_parser()
    argument_validation(args.train_data)
    if args.test_data:
        argument_validation(args.test_data)
    student_model, teacher_model = load_models(args)

    ds = DatasetDict()

    train_data, eval_data, le = preprocess(args)
    train_dataset = Dataset.from_pandas(train_data)
    train_dataset = train_dataset.add_column("label", [0] * len(train_dataset))
    if args.test_data:
        eval_dataset = Dataset.from_pandas(eval_data)
        ds["eval"] = eval_dataset

    ds["train"] = train_dataset

    distill(
        student_model,
        teacher_model,
        ds,
        args.batch_size,
        args.num_iterations,
        args.epochs,
        le,
    )


if __name__ == "__main__":
    main()
