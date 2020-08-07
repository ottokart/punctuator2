#!/usr/bin/env python
import sys
import codecs

mapping = {"COMMA": ",COMMA", "PERIOD": ".PERIOD", "QUESTION": "?QUESTIONMARK", "O": ""}
counts = dict((p, 0) for p in mapping)

print('Reading %s and outputting to %s.' % (sys.argv[1], sys.argv[2]))
with codecs.open(sys.argv[1], 'r', 'utf-8', 'ignore') as f_in, \
     codecs.open(sys.argv[2], 'w', 'utf-8') as f_out:

    for i, line in enumerate(f_in):

        line = line.replace('?', '')

        parts = line.split()

        if len(parts) == 0:
            continue

        if len(parts) == 1:
            word = ""
            punct = parts[0]
        else:
            word, punct = parts

        counts[punct] += 1

        f_out.write("%s %s " % (word, mapping[punct]))

print("Counts:")
for p, c in counts.items():
    print("%s: %d" % (p, c))
