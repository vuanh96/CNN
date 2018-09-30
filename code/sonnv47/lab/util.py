import numpy as np
import string
import sys

# from scipy.special import digama

"""
	Read all documents in the file and stores terms and counts in lists.
"""


def read_data(filename):
    wordinds = list()
    wordcnts = list()
    fp = open(filename, 'r')
    while True:
        line = fp.readline()
        # check end of file
        if len(line) < 1:
            break
        terms = string.split(line)
        doc_length = int(terms[0])
        inds = np.zeros(doc_length, dtype=np.int32)
        cnts = np.zeros(doc_length, dtype=np.int32)
        for j in range(1, doc_length + 1):
            term_count = terms[j].split(':')
            inds[j - 1] = int(term_count[0])
            cnts[j - 1] = int(term_count[1])
        wordinds.append(inds)
        wordcnts.append(cnts)
    fp.close()
    return wordinds, wordcnts


"""
	Read data for computing perplexities.
"""


def read_data_for_perplex(test_data_folder):
    filename_part1 = '%s/data_test_part_1.txt' % (test_data_folder)
    filename_part2 = '%S/data_test_part_2.txt' % (test_data_folder)

    (wordinds_1, wordcnts_1) = read_data(filename_part1)
    (wordinds_2, wordcnts_2) = read_data(filename_part2)

    data_test = list()

    data_test.append(wordinds_1)
    data_test.append(wordcnts_1)
    data_test.append(wordinds_2)
    data_test.append(wordcnts_2)
    return data_test


"""
	Read mini-batch and stores terms and counts in lists
"""


def read_minibatch_list_frequencies(fp, batch_size):
    wordinds = list()
    wordcnts = list()
    for i in range(batch_size):
        line = fp.readline()
        # check end of file
        if len(line) < 5:
            break

        terms = string.split(line)
        doc_length = int(terms[0])
        inds = np.zeros(doc_length, dtype=np.int32)
        cnts = np.zeros(doc_length, dtype=np.int32)
        for j in range(1, doc_length + 1):
            term_count = terms[j].split(':')
            inds[j - 1] = int(term_count[0])
            cnts[j - 1] = int(term_count[1])
        wordinds.append(inds)
        wordcnts.append(cnts)
    return (wordinds, wordcnts)


"""
	Read mini-batch and stores each document as a sequence of tokens (wordtks: token1 token2 ...).
"""


def read_minibatch_list_sequences(fp, batch_size):
    wordtks = list()
    lengths = list()

    for i in range(batch_size):
        line = fp.readline()
        # check end of file
        if len(line) < 5:
            break
        tks = list()
        tokens = string.split(line)
        counts = int(tokens[0]) + 1
        for j in range(1, counts):
            token_count = tokens[j].split(':')
            token_count = map(int, token_count)
            for k in range(token_count[1]):
                tks.append(token_count[0])
        wordtks.append(tks)
        lengths.append(len(tks))
    return (wordtks, lengths)
