import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
np.set_printoptions(suppress=True)


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

limit = 1024

def breakup(input_text):
    """
    This function takes a single string as input and breaks it up into 
    multiple strings each of which has a length less than the limit set. 
    The strings are broken down at full stops 
    closest to the the limit set.

    Args:
        - A single string
    Returns:

        - A list of strings each of which has length less than the limit set after conversion into tokens."""

    # add full stop at the end of the text if not already present to mark end
    if input_text[-1] != ".":
        input_text += "."
    encoded_input = tokenizer(
        input_text
    )  # encode the entire text to get the total token size

    process = []
    to_loop = (
        len(encoded_input["input_ids"]) // limit + 1
    )  # check the number of chunks we can make of 512 token size

    for i in range(to_loop):
        breakup = tokenizer.decode(
            encoded_input["input_ids"][:limit]
        )  # convert first 512 tokens to raw text.

        end_sentence = breakup.rfind(
            "."
        )  # find the last full stop in the text to find the end of the last complete sentence

        if end_sentence != -1:
            process.append(
                breakup[0 : end_sentence + 1]
            )  # break the raw text at the last complete sentence and add it to the list
            input_text = input_text[end_sentence + 1 :]  # take the remaining raw text
            encoded_input = tokenizer(input_text)  # convert it into tokens again
        else:
            process.append(
                breakup
            )  # if full stop not found add the entire text to the list
            input_text = input_text[len(breakup) :]  # take the remaining raw text
            encoded_input = tokenizer(input_text)  # convert it into tokens again

    return process

def predict(data):

    input_text = data["context"]  # load text to classify
    if isinstance(data["labels"], str):
        candidate_labels = data["labels"].split(",")
    else:
        candidate_labels = data["labels"]

    # load labels
    #candidate_labels = list(candidate_labels)
    process = breakup(data['context'])
    arr = np.zeros((len(candidate_labels)))
    total = 0
    for textblock in process:
        if len(textblock) == 0:
            continue
        ans = classifier(textblock, candidate_labels)
        arr += ans["scores"]
        total += 1
    arr = arr / total
    dict1={}
    for prob,label in zip(arr,candidate_labels):
        dict1[label]=str(prob)
    # average score across all chunks for all labels
    return dict1
