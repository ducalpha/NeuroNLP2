from __future__ import print_function

__author__ = 'duc'
"""
Analyze LSTM-CNNs-CRF model for NER.
"""

import sys
import os

sys.path.append(".")
sys.path.append("..")

import argparse
import uuid
import torch
from neuronlp2.io import get_logger, conll03_data, CoNLL03Writer
import NERCRF

uid = uuid.uuid4().hex[:6]

def perform_ner(batch_size, model_dir, model_name, file_path):
    logger = get_logger("NERCRF")

    logger.info("Creating Alphabets")
    test_path = file_path
    # Set everything to None because the alphabets are supposed to be loaded from training phase.
    word_alphabet, char_alphabet, pos_alphabet, \
    chunk_alphabet, ner_alphabet = conll03_data.create_alphabets("data/alphabets/ner_crf/", None)

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    logger.info("Chunk Alphabet Size: %d" % chunk_alphabet.size())
    logger.info("NER Alphabet Size: %d" % ner_alphabet.size())

    logger.info("Reading Data")
    # TODO: hard code 'cuda' for now.
    data_test = conll03_data.read_data_to_tensor(test_path, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, device=torch.device('cuda'))

    logger.info("Constructing Network...")
    network = torch.load(os.path.join(model_dir, model_name))

    writer = CoNLL03Writer(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet)
    with torch.no_grad():
        network.eval()  # This set dropout to an identity function.
        tmp_filename = 'tmp/%s_analyze' % (str(uid))
        writer.start(tmp_filename)

        for batch in conll03_data.iterate_batch_tensor(data_test, batch_size):
            word, char, pos, chunk, labels, masks, lengths = batch
            preds, _ = network.decode(word, char, target=labels, mask=masks, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
            writer.write(word.cpu().numpy(), pos.cpu().numpy(), chunk.cpu().numpy(), preds.cpu().numpy(), labels.cpu().numpy(), lengths.cpu().numpy())
        writer.close()
        test_acc, test_precision, test_recall, test_f1 = NERCRF.evaluate(tmp_filename)

        print("Test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%" % (test_acc, test_precision, test_recall, test_f1))

def main():
    parser = argparse.ArgumentParser(description='Tuning with bi-directional RNN-CNN-CRF')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    parser.add_argument('--model_dir', help='dir path for saving model file.', required=True)
    parser.add_argument('--model_name', help='file name for saving model file.', required=True)
    parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    args = parser.parse_args()
    perform_ner(args.batch_size, args.model_dir, args.model_name, args.test)

if __name__ == '__main__':
    main()
