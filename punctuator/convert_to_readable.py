import sys
from io import open

from .data import EOS_TOKENS, PUNCTUATION_VOCABULARY


def convert(input_text, out_f, with_newlines=False):
    """
    Translates punctuation tokens to normal punctuation.
    """
    last_was_eos = True
    first = True
    for token in input_text.split():
        if token in PUNCTUATION_VOCABULARY:
            out_f.write(token[:1])
        else:
            out_f.write(('' if first else ' ') + (token.title() if last_was_eos else token))

        last_was_eos = token in EOS_TOKENS
        if with_newlines and last_was_eos:
            out_f.write('\n')
            first = True
        else:
            first = False


if __name__ == "__main__":

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        sys.exit("Input file path argument missing")

    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        sys.exit("Output file path argument missing")

    with_newlines = len(sys.argv) > 3 and bool(int(sys.argv[3]))

    with open(input_file, 'r', encoding='utf-8') as in_f, open(output_file, 'w', encoding='utf-8') as out_f:
        input_text = in_f.read()
        convert(input_text, out_f, with_newlines=with_newlines)
