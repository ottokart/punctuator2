import sys
import codecs
from data import EOS_TOKENS, PUNCTUATION_VOCABULARY

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

    with codecs.open(input_file, 'r', 'utf-8') as in_f, codecs.open(output_file, 'w', 'utf-8') as out_f:
        last_was_eos = True
        first = True
        for token in in_f.read().split():
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
