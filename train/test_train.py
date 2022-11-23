import unittest
from extractor import text_extraction
from text_breaker import breaker


class Test_Train(unittest.TestCase):
    """Defining the sample data and files to carry out the testing"""

    def setUp(self):
        self.pdf = "sample.pdf"
        self.text = "I love making different dishes for my family."
        self.extraction = text_extraction()
        self.breaking = breaker()


class Test_extractor(Test_Train):
    """Testing the extractor code used to extract text from pdfs"""

    def test_pdf(self):
        result = self.extraction._process_file_pdf(self.pdf)
        self.assertIsInstance(result, dict)
        self.assertEqual(result[0], "This is the sample digital pdf page.")
        self.assertEqual(result[1], "")

    def test_ocr(self):
        result = self.extraction._process_file_ocr(self.pdf, [1], {})
        self.assertIsInstance(result, dict)
        self.assertEqual(result[1], "This is a sample scanned page. ")

    def test_extract(self):
        result = self.extraction.extract_pdf(self.pdf)
        self.assertIsInstance(result, dict)


class Test_text_breaker(Test_Train):
    """Testing the text breaker"""

    def test_breakup(self): 
        result = self.breaking.breakup(self.text)
        self.assertIsInstance(result, list)
    
    
if __name__ == "__main__":
    unittest.main()
