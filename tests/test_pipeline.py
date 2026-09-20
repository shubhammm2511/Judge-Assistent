import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chunking import chunk_pages  # noqa: E402
from ingestion import extract_pdf_pages  # noqa: E402
from rag_pipeline import build_context, build_messages  # noqa: E402


class PipelineSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pdf_path = PROJECT_ROOT / "data" / "Agent-as-Judge.pdf"
        cls.pages = extract_pdf_pages(cls.pdf_path)
        cls.chunks = chunk_pages(cls.pages)

    def test_pdf_extraction_preserves_page_metadata(self):
        self.assertEqual(len(self.pages), 44)
        self.assertEqual(self.pages[0]["metadata"]["source"], "Agent-as-Judge.pdf")
        self.assertEqual(self.pages[0]["metadata"]["page"], 1)
        self.assertTrue(self.pages[0]["text"])

    def test_chunking_preserves_source_and_page(self):
        self.assertGreater(len(self.chunks), 0)
        for chunk in self.chunks:
            self.assertTrue(chunk["text"])
            self.assertIn("source", chunk["metadata"])
            self.assertIn("page", chunk["metadata"])
            self.assertIn("chunk_id", chunk["metadata"])

    def test_prompt_contains_question_and_retrieved_context(self):
        sample = {
            **self.chunks[0],
            "similarity": 0.75,
        }
        question = "What is the DevAI dataset?"
        messages = build_messages(question, [sample])

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn(question, messages[1]["content"])
        self.assertIn(sample["text"], build_context([sample]))


if __name__ == "__main__":
    unittest.main()
