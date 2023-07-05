
from setfit import SetFitModel
from setfit.exporters.utils import mean_pooling
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

import numpy as np
import torch
import magic
import pickle
import os
import base64

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
from extractor import text_extraction
from breakup import breaker
from link_file import extract_data




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
        return self.model_head.predict_proba(embeddings)

    def predict_proba(self, inputs):
        return self.predict(inputs)
    
    
def load_ort_model(path):
    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        path, file_name="model.onnx"
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    modelhead = SetFitModel.from_pretrained(path).model_head
    onnx_setfit_model = OnnxSetFitModel(ort_model, tokenizer, modelhead)

    return onnx_setfit_model


class setup_model:
    def __init__(self):
        
        self.extracting = text_extraction()
        if torch.cuda.is_available():
            self.device = torch.device("gpu")
        else:
            self.device = torch.device("cpu")

        self.is_onnx_model = False

        if os.path.exists("/input/quantization"):

            self.model = load_ort_model("/input/quantization")
            pkl_file = open("/input/quantization/labelencoder.pkl", 'rb')
            self.encoder = pickle.load(pkl_file)
            pkl_file.close()
            self.tokenizer = AutoTokenizer.from_pretrained("/input/quantization")
            self.is_onnx_model = True

        elif os.path.exists("/input/onnx"):

            self.model = load_ort_model("/input/onnx")
            pkl_file = open("/input/onnx/labelencoder.pkl", 'rb')
            self.encoder = pickle.load(pkl_file)
            pkl_file.close()
            self.tokenizer = AutoTokenizer.from_pretrained("/input/onnx")
            self.is_onnx_model = True

        elif os.path.exists("/input/distillation"):
            pkl_file = open("/input/distillation/labelencoder.pkl", 'rb')
            self.encoder = pickle.load(pkl_file)
            pkl_file.close()
            self.model = SetFitModel.from_pretrained("/input/distillation")
            self.tokenizer = AutoTokenizer.from_pretrained("/input/distillation")

        elif os.path.exists("/input/train"):
            pkl_file = open("/input/train/labelencoder.pkl", 'rb')
            self.encoder = pickle.load(pkl_file)
            pkl_file.close()
            self.model = SetFitModel.from_pretrained("/input/train")
            self.tokenizer = AutoTokenizer.from_pretrained("/input/train")    
     
        self.predict = self.trained_predictor
        self.breakup = breaker(self.tokenizer)

    def get_prediction_torch(self, text):
        process = self.breakup.breakup(text)
        arr = np.zeros((len(list(self.encoder.classes_))))
        total = 0
        for textblock in process:
            if(len(textblock)) == 0:
                continue
            preds = self.model.predict_proba([textblock])
            arr += np.array(preds[0])
            total += 1
        arr = arr / total
        dict1 = {}
        for prob, label in zip(arr, list(self.encoder.classes_)):
            dict1[label] = str(format(prob, "f"))
        return dict1

    def get_prediction_onnx(self, text):
        process = self.breakup.breakup(text)
        arr = np.zeros((len(list(self.encoder.classes_))))
        total = 0
        for textblock in process:
            if(len(textblock)) == 0:
                continue
            preds = self.model.predict_proba(textblock)[0]
            arr += np.array(preds)
            total += 1
        arr = arr / total
        dict1 = {}
        for prob, label in zip(arr, list(self.encoder.classes_)):
            dict1[label] = str(format(prob, "f"))
        return dict1

    def trained_predictor(self, data):
        #check whether data is a base64 string, if yes decode it, else continue with normal 
        if self.isBase64(data['context']):
            decoded = base64.b64decode(data['context'])
            file_ext = magic.from_buffer(decoded, mime=True).split("/")[-1]
            savepath = f"file.{file_ext}"  # decode the input file
            f = open(savepath, "wb")
            f.write(decoded)
            f.close()
            message = self.extracting.master_extractor(savepath)
        else:
            message = data["context"]
            try:
                message = extract_data(data["context"])
            except:
                message = data["context"]

        if self.is_onnx_model:
            return self.get_prediction_onnx(message)
        else:
            return self.get_prediction_torch(message)

    def isBase64(self, string):
        try:
            return base64.b64encode(base64.b64decode(string)).decode("utf-8") == string
        except Exception:
            return False


predictor = setup_model()


def predict(data):
    result = predictor.predict(data)
    return result
