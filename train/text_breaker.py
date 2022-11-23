import numpy as np
from transformers import AutoTokenizer
import os


class breaker:

    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        np.set_printoptions(suppress=True)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.limit = 512

    def breakup(self, input_text):
        """
        This function takes a single string as input and breaks it up into
        multiple strings each of which has a length less than the limit set.
        The strings are broken down at full stops
        closest to the the limit set.

        Args:
            - A single string
        Returns:

            - A list of strings each of which has length less than the limit set, after conversion into tokens."""

        # add full stop at the end of the text if not already present to mark end
        if input_text[-1] != ".":
            input_text += "."
        encoded_input = self.tokenizer(
            input_text
        )  # encode the entire text to get the total token size

        process = []
        to_loop = (
            len(encoded_input["input_ids"]) // self.limit + 1
        )  # check the number of chunks we can make of 512 token size

        for i in range(to_loop):
            breakup = self.tokenizer.decode(
                encoded_input["input_ids"][:self.limit], skip_special_tokens=True
            )  # convert first 512 tokens to raw text.

            end_sentence = breakup.rfind(
                "."
            )  # find the last full stop in the text to find the end of the last complete sentence

            if end_sentence != -1:
                process.append(
                    breakup[0: end_sentence + 1]
                )  # break the raw text at the last complete sentence and add it to the list
                input_text = input_text[end_sentence + 1:]  # take the remaining raw text
                encoded_input = self.tokenizer(input_text)  # convert it into tokens again
            else:
                process.append(
                    breakup
                )  # if full stop not found add the entire text to the list
                input_text = input_text[len(breakup):]  # take the remaining raw text
                encoded_input = self.tokenizer(input_text)  # convert it into tokens again

        return process
