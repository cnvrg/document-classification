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
from bert_model import BERT_Arch
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

        if os.path.exists("/input/train"):
            model_path = "/input/train/model.pt"
            intents_path = "/input/train/classes.json"
            f = open(intents_path)
            self.intents = json.load(f)
            number_of_labels = len(self.intents)
            # Import the DistilBert pretrained model
            self.model = BERT_Arch(
                DistilBertModel.from_pretrained("distilbert-base-uncased"),
                number_of_labels,
            )
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
            self.model.load_state_dict(torch.load(model_path))
            self.predict = self.trained_predictor
            self.breakup = breaker("trained")
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            np.set_printoptions(suppress=True)
            nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
            tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
            if torch.cuda.is_available():
                self.pipe = pipeline(task='zero-shot-classification', 
                                     tokenizer=tokenizer, model=nli_model, 
                                     device=0)
            else:
                self.pipe = pipeline(task='zero-shot-classification', 
                                     tokenizer=tokenizer, model=nli_model)
            self.predict = self.predefined_predict
            self.breakup = breaker()

    def predefined_predict(self, data):
        if self.isBase64(data['context']):
            decoded = base64.b64decode(data['context'])
            file_ext = magic.from_buffer(decoded, mime=True).split("/")[-1]
            savepath = f"file.{file_ext}"  # decode the input file
            f = open(savepath, "wb")
            f.write(decoded)
            f.close()
            sequences = self.extracting.master_extractor(savepath)
            print(sequences)
        else:
            try:
                print("checking")
                sequences = extract_data(data["context"])
            except:
                sequences = data["context"]
        process = self.breakup.breakup(sequences)
        
        candidate_labels = data["labels"]
        arr = np.zeros((len(candidate_labels)))
        total = 0
        for textblock in process:
            if(len(textblock)) == 0:
                continue
            ans = self.pipe(sequences=sequences, 
                            candidate_labels=candidate_labels, 
                            multi_label=False)
            arr += ans["scores"]
            total += 1
        arr = arr / total
        dict1 = {}
        for prob, label in zip(arr, ans['labels']):
            dict1[label] = str(format(prob,"f"))
        # average score across all chunks for all labels
        return dict1

    def get_prediction(self, text):

        chunks = self.breakup.breakup(text)
        dict1 = {}
        for chunk in chunks:
            test_text = [chunk]
            self.model.eval()

            tokens_test_data = self.tokenizer(
                test_text,
                max_length=8,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
            )
            test_seq = torch.tensor(tokens_test_data["input_ids"])
            test_mask = torch.tensor(tokens_test_data["attention_mask"])
            with torch.no_grad():
                preds = self.model(test_seq.to(self.device), test_mask.to(self.device))
            sm = torch.nn.Softmax(dim=1)
            probabilities = sm(preds)
            confidence = probabilities.detach().cpu().numpy()[0]
            
            for prob, label in zip(confidence, self.intents):
                try:
                    dict1[label] += prob
                except:
                    dict1[label] = prob
        
        #average the dict1 values across all texts
        for keys in dict1.keys():
            dict1[keys] = str(format(dict1[keys]/len(chunks), "f"))

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
        return self.get_prediction(message)

    def isBase64(self, string):
        try:
            return base64.b64encode(base64.b64decode(string)).decode("utf-8") == string
        except Exception:
            return False

predictor = setup_model()


def predict(data):
    result = predictor.predict(data)
    return result
