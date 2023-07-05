import os
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import json
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
from breakup import breaker
from link_file import extract_data
import base64
from extractor import text_extraction
import magic


class setup_model:
    def __init__(self):

        self.extracting = text_extraction()
        if torch.cuda.is_available():
            self.device = torch.device("gpu")
        else:
            self.device = torch.device("cpu")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        np.set_printoptions(suppress=True)
        nli_model = AutoModelForSequenceClassification.from_pretrained(
            "facebook/bart-large-mnli"
        )
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        if torch.cuda.is_available():
            self.pipe = pipeline(
                task="zero-shot-classification",
                tokenizer=tokenizer,
                model=nli_model,
                device=0,
            )
        else:
            self.pipe = pipeline(
                task="zero-shot-classification", tokenizer=tokenizer, model=nli_model
            )
        self.predict = self.predefined_predict
        self.breakup = breaker()

    def predefined_predict(self, data):
        if self.isBase64(data["context"]):
            decoded = base64.b64decode(data["context"])
            file_ext = magic.from_buffer(decoded, mime=True).split("/")[-1]
            savepath = f"file.{file_ext}"  # decode the input file
            f = open(savepath, "wb")
            f.write(decoded)
            f.close()
            sequences = self.extracting.master_extractor(savepath)
            print(sequences)
        else:
            try:
                sequences = extract_data(data["context"])
            except:
                sequences = data["context"]
        print(sequences)
        process = self.breakup.breakup(sequences)

        if isinstance(data["labels"], str):
            candidate_labels = data["labels"].split(",")
        else:
            candidate_labels = data["labels"]
        arr = np.zeros((len(candidate_labels)))
        total = 0
        for textblock in process:
            if (len(textblock)) == 0:
                continue
            ans = self.pipe(
                sequences=sequences,
                candidate_labels=candidate_labels,
                multi_label=False,
            )
            arr += ans["scores"]
            total += 1
        arr = arr / total
        dict1 = {}
        for prob, label in zip(arr, ans["labels"]):
            dict1[label] = str(format(prob, "f"))
        # average score across all chunks for all labels
        return dict1

    def isBase64(self, string):
        try:
            return base64.b64encode(base64.b64decode(string)).decode("utf-8") == string
        except Exception:
            return False


predictor = setup_model()


def predict(data):
    result = predictor.predict(data)
    return result
