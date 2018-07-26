from __future__ import print_function

__author__ = 'duc'
"""
Implementation of Bi-directional LSTM-CNNs-CRF model for NER.
"""

import sys
import os

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import uuid

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from neuronlp2.io import get_logger, conll03_data, CoNLL03Writer
from neuronlp2.models import BiRecurrentConvCRF, BiVarRecurrentConvCRF
from neuronlp2 import utils

uid = uuid.uuid4().hex[:6]


def evaluate(output_file):
    score_file = "tmp/score_%s" % str(uid)
    os.system("examples/eval/conll03eval.v2 < %s > %s" % (output_file, score_file))
    with open(score_file, 'r') as fin:
        fin.readline()
        line = fin.readline()
        fields = line.split(";")
        acc = float(fields[0].split(":")[1].strip()[:-1])
        precision = float(fields[1].split(":")[1].strip()[:-1])
        recall = float(fields[2].split(":")[1].strip()[:-1])
        f1 = float(fields[3].split(":")[1].strip())
    return acc, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description='Tuning with bi-directional RNN-CNN-CRF')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    parser.add_argument('--model_dir', help='dir path for saving model file.', required=True)
    parser.add_argument('--model_name', help='file name for saving model file.', required=True)
    parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"

    args = parser.parse_args()

    logger = get_logger("NERCRF")

    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet, pos_alphabet, \
    chunk_alphabet, ner_alphabet = conll03_data.create_alphabets("data/alphabets/ner_crf/", test_path, data_paths=[test_path, test_path],
                                                                 embedd_dict=None, max_vocabulary_size=50000)

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    logger.info("Chunk Alphabet Size: %d" % chunk_alphabet.size())
    logger.info("NER Alphabet Size: %d" % ner_alphabet.size())

    logger.info("Reading Data")
    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    num_labels = ner_alphabet.size()

    data_test = conll03_data.read_data_to_tensor(test_path, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, device=device)

    writer = CoNLL03Writer(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet)

    logger.info("constructing network...")

    model_dir = args.model_dir
    model_name = args.model_name
    model_path = os.path.join(model_dir, model_name)
    network.load_state_dict(torch.load(model_path))
    network = torch.load(model_path)

    # How to set the dropout prob to 1?
    batch_size = args.batch_size
    with torch.no_grad():
        tmp_filename = 'tmp/%s_analyze' % (str(uid))
        writer.start(tmp_filename)

        for batch in conll03_data.iterate_batch_tensor(data_test, batch_size):
            word, char, pos, chunk, labels, masks, lengths = batch
            preds, _ = network.decode(word, char, target=labels, mask=masks, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
            writer.write(word.cpu().numpy(), pos.cpu().numpy(), chunk.cpu().numpy(), preds.cpu().numpy(), labels.cpu().numpy(), lengths.cpu().numpy())
        writer.close()
        test_acc, test_precision, test_recall, test_f1 = evaluate(tmp_filename)

        print("Test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%" % (test_acc, test_precision, test_recall, test_f1))


if __name__ == '__main__':
    main()
