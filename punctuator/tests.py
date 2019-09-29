from __future__ import absolute_import

import time
import os
import unittest
import tempfile
import shutil
from io import StringIO

from . import punc
from .punc import Punctuator, download_model


class Tests(unittest.TestCase):

    def test_punctuate(self):

        samples = [
            (
                'mary had a little lamb its fleece was white as snow and anywhere that mary went the lamb was sure to go',
                'Mary had a little lamb, its fleece was white as snow and anywhere that mary went, the lamb was sure to go.'
            ),
            (
                "they say it's only as cold as it feels in your mind i don't buy into that theory much what do you think",
                "They say it's only as cold as it feels in your mind. I don't buy into that theory much. What do you think."
            ),
            (
                "he's a do me a favor go home to your wife",
                "He's a do me: a favor go home to your wife.",
            ),
            (
                "they'll even negotiate your rate with the insurance company",
                "They'll even negotiate your rate with the insurance company.",
            ),
            (
                "for me i wanted to get into commentary some sort of way i didn't know how to do that so i left the firm and i started a business",
                "For me, I wanted to get into commentary some sort of way. I didn't know how to do that. So I left the firm and I started a business."
            ),
        ]

        # Create temp directory for downloading data.
        d = tempfile.mkdtemp()
        os.makedirs(punc.PUNCTUATOR_DATA_DIR, exist_ok=True)
        model_file = os.path.join(punc.PUNCTUATOR_DATA_DIR, 'Demo-Europarl-EN.pcl')
        print('Temp dir:', d)
        os.chdir(d)
        try:

            # Download pre-trained model.
            if not os.path.isfile(model_file):
                model_file = download_model()
            print('Model file:', model_file)

            # Create punctuator.
            t0 = time.time()
            p = Punctuator(model_file=model_file)
            td = time.time() - t0
            print('Loaded in %s seconds.' % td)

            # Add punctuation.
            for input_text, expect_output_text in samples:
                fout = StringIO()
                actual_output_text = p.punctuate(input_text)
                print('expect_output_text:', expect_output_text)
                print('actual_output_text:', actual_output_text)
                self.assertEqual(actual_output_text, expect_output_text)

            # Serialize the entire punctuator, not just the model.
            print('Writing...')
            t0 = time.time()
            fn = 'data.pickle'
            p.save(fn)
            td = time.time() - t0
            print('Wrote in %s seconds.' % td)

            # Load puncutator.
            print('Loading...')
            t0 = time.time()
            p2 = Punctuator.load(fn)
            td = time.time() - t0
            print('Loaded in %s seconds.' % td)

            # Confirm punctuations match previous.
            for input_text, expect_output_text in samples:
                fout = StringIO()
                actual_output_text = p2.punctuate(input_text)
                print('expect_output_text:', expect_output_text)
                print('actual_output_text:', actual_output_text)
                self.assertEqual(actual_output_text, expect_output_text)

        finally:
            shutil.rmtree(d)


if __name__ == '__main__':
    unittest.main()
