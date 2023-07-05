from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from pathlib import Path
import os
import argparse
import shutil
cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")


def argument_parser():
    parser = argparse.ArgumentParser(description="""Training""")
    parser.add_argument(
        "--model_folder",
        action="store",
        dest="model_folder",
        required=True,
        default=False,
        help="""Path to the model folder containing model, tokenizer, config.json you want to convert to onnx""",
    )

    return parser.parse_args()


def argument_validation(data):

    # check if the file exists
    assert os.path.exists(data), (
        " Path to the model file provided " + data + " does not exist "
    )
    

def conversion(model_path):
    onnx_path = Path(cnvrg_workdir)
    ort_model = ORTModelForFeatureExtraction.from_pretrained(model_path, from_transformers=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    ort_model.save_pretrained(onnx_path)
    tokenizer.save_pretrained(onnx_path)
    

def move_files_to_folder(list_of_files, destination_folder):
    """
    move a list of files to a target directory
    """
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)


def move_files(model_path):
    
    files = [os.path.join(model_path, f) for f in os.listdir(model_path)]
    move_files_to_folder(files, cnvrg_workdir)


def main():
    args = argument_parser()
    argument_validation(args.model_folder)

    conversion(args.model_folder)
    #move all setfit files to /cnvrg
    move_files(args.model_folder)


if __name__ == "__main__":
    main()
    


