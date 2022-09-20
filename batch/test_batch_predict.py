import unittest

from extractor import (
    _process_file_ocr,
    _process_file_pdf,
    extract_pdf,
)
from dc_inference import breakup, predict


class Test_batch(unittest.TestCase):
    """Defining the sample data and files to carry out the testing"""

    def setUp(self):
        self.pdf = "sample.pdf"
        self.text = "I love making different dishes for my family."


class Test_extractor(Test_batch):
    """Testing the extractor code used to extract text from pdfs"""

    def test_pdf(self):
        result = _process_file_pdf(self.pdf)
        self.assertIsInstance(result, dict)
        self.assertEqual(result[0], "This is the sample digital pdf page.")
        self.assertEqual(result[1], "")

    def test_ocr(self):
        result = _process_file_ocr(self.pdf, [1], {})
        self.assertIsInstance(result, dict)
        self.assertEqual(result[1], "This is a sample scanned page. ")

    def test_extract(self):
        result = extract_pdf(self.pdf)
        self.assertIsInstance(result, dict)


class Test_dc_inference(Test_batch):
    """Testing the document classification"""

    def test_breakup(self): 
        result = breakup(self.text)
        self.assertIsInstance(result, list)

    def test_classification(self):
        result = predict({"context": self.text, "labels": ["cooking", "cleaning"]})
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
