import functools
from neural_compressor.experimental import Quantization, common
import evaluate
import onnxruntime
from transformers import AutoTokenizer
from optimum.pipelines import ORTModelForFeatureExtraction
from setfit.exporters.utils import mean_pooling
from setfit import SetFitModel
import pandas as pd
import os
import argparse
import pickle
from tqdm.auto import tqdm
import shutil

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")


def argument_parser():
    parser = argparse.ArgumentParser(description="""Training""")
    parser.add_argument(
        "--onnx_model_folder",
        action="store",
        dest="onnx_model_folder",
        required=False,
        default=None,
        help="""Path to the onnx model folder""",
    )
    parser.add_argument(
        "--setfit_model_folder",
        action="store",
        dest="setfit_model_folder",
        required=True,
        default=False,
        help="""Path to the model folder containing model, tokenizer, config.json you want to convert to onnx. We will only load model head from here.""",
    )
    parser.add_argument(
        "--calibration_data",
        action="store",
        dest="calibration_data",
        required=False,
        default=False,
        help="""Path to the test.csv used to evaluate the model in quantization""",
    )
    # TODO add hyperparams for quantization
    return parser.parse_args()


def argument_validation(data, model=True):

    if model:
        # check if the model file exists
        assert os.path.exists(data), (
            " Path to the model file provided " + data + " does not exist "
        )
    else:
        assert os.path.exists(data + "/labelencoder.pkl"), (
            " Label encoder does not exist in the setfit folder "
            + data
            + "/labelencoder.pkl"
            + " does not exist "
        )


class OnnxSetFitModel:
    def __init__(self, ort_model, tokenizer, model_head):
        self.ort_model = ort_model
        self.tokenizer = tokenizer
        self.model_head = model_head

    def predict(self, inputs):
        encoded_inputs = self.tokenizer(
            inputs, padding=True, truncation=True, return_tensors="pt"
        )
        outputs = self.ort_model(**encoded_inputs)
        embeddings = mean_pooling(
            outputs["last_hidden_state"], encoded_inputs["attention_mask"]
        )
        return self.model_head.predict(embeddings)

    def __call__(self, inputs):
        return self.predict(inputs)


def build_dynamic_quant_yaml():
    yaml = """
    model:
      name: speedup
      framework: onnxrt_integerops

    device: cpu

    quantization:
      approach: post_training_dynamic_quant

    tuning:
      accuracy_criterion:
        relative: 0.01
      exit_policy:
        timeout: 0
      random_seed: 9527
        """
    with open("onnx_dynamic.yaml", "w", encoding="utf-8") as f:
        f.write(yaml)


def dataloader(path_to_data, path_to_encoder):

    test_dataset = pd.read_csv(path_to_data)
    pkl_file = open(path_to_encoder + "/labelencoder.pkl", "rb")
    encoder = pickle.load(pkl_file)
    test_dataset["label"] = encoder.transform(test_dataset["label"])
    return test_dataset


def move_files(model_path):
    shutil.move(model_path, cnvrg_workdir)


def move_files_to_folder(list_of_files, destination_folder):
    """
    move a list of files to a target directory
    """
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)


def run_conversion_onnx(model_path):
    os.system(f"python conv_to_onnx.py --model_folder {model_path}")


def main():
    args = argument_parser()
    argument_validation(args.setfit_model_folder)
    argument_validation(args.setfit_model_folder, model=False)

    # check if onnx was provided
    # if onnx not present convert model to onnx
    if args.onnx_model_folder is None:
        run_conversion_onnx(args.setfit_model_folder)
        args.onnx_model_folder = cnvrg_workdir
        args.setfit_model_folder = cnvrg_workdir
    else:
        argument_validation(args.onnx_model_folder)

    # load the setfit model to get the head
    model_head = SetFitModel.from_pretrained(args.setfit_model_folder).model_head

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.setfit_model_folder)

    # load the test dataset
    if args.calibration_data:
        test_dataset = dataloader(args.calibration_data, args.setfit_model_folder)

    onnx_path = args.onnx_model_folder

    metric = evaluate.load("accuracy")

    # eval function
    def eval_func(model):
        ort_model = ORTModelForFeatureExtraction.from_pretrained(onnx_path)
        ort_model.model = onnxruntime.InferenceSession(model.SerializeToString(), None)
        onnx_setfit_model = OnnxSetFitModel(ort_model, tokenizer, model_head)
        preds = []
        chunk_size = 100
        for i in tqdm(range(0, len(test_dataset["text"]), chunk_size)):
            preds.extend(
                onnx_setfit_model.predict(
                    list(test_dataset["text"])[i : i + chunk_size]
                )
            )
        labels = test_dataset["label"]

        accuracy = metric.compute(predictions=preds, references=labels)
        return accuracy["accuracy"]

    def do_quantize():
        build_dynamic_quant_yaml()

        onnx_output_path = cnvrg_workdir + "/model.onnx"
        quantizer = Quantization("onnx_dynamic.yaml")

        quantizer.model = common.Model(args.onnx_model_folder + "/model.onnx")
        if args.calibration_data:
            quantizer.eval_func = functools.partial(eval_func)
            quantized_model = quantizer()
        else:
            quantized_model = quantizer()

        quantized_model.save(onnx_output_path)

    do_quantize()

    files = [
        os.path.join(args.setfit_model_folder, f)
        for f in os.listdir(args.setfit_model_folder)
    ]
    move_files_to_folder(files, cnvrg_workdir)


if __name__ == "__main__":
    main()
