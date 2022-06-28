import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer
np.set_printoptions(suppress=True)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")


def predict(data):
    """

    """
    input_text = data["context"]  # load text to classify
    candidate_labels = data["labels"]  # load labels
    candidate_labels = list(candidate_labels)
    if input_text[-1] != ".":
        input_text += "."
    encoded_input = tokenizer(
        input_text
    )  # encode the entire text to get the total token size
    process = []
    to_loop = (
        len(encoded_input["input_ids"]) // 1024 + 1
    )  # check the number of chunks we can make of 1024 token size
    # break data into multiple strings so that length after tokenizing doesn't exceed 1024.
    for i in range(to_loop):
        breakup = tokenizer.decode(
            encoded_input["input_ids"][:1024], skip_special_tokens=True
        )
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
            input_text = input_text[1024:]  # take the remaining raw text
            encoded_input = tokenizer(input_text)  # convert it into tokens again
    arr = np.zeros((len(candidate_labels)))
    total = 0
    for textblock in process:
        if len(textblock) == 0:
            continue
        ans = classifier(textblock, candidate_labels, multi_class=True)
        arr += ans["scores"]
        total += 1
    arr = arr / total
    dict1={}
    for prob,label in zip(arr,candidate_labels):
        dict1[label]=str(prob)
    # average score across all chunks for all labels
    return dict1
