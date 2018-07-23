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
    parser.add_argument('--cuda', action='store_true', help='using GPU')
    parser.add_argument('--embedding', choices=['glove', 'senna', 'sskip', 'polyglot'], help='Embedding for words', required=True)
    parser.add_argument('--embedding_dict', help='path for embedding dict')
    parser.add_argument('--model_dir', help='dir path for saving model file.', required=True)
    parser.add_argument('--model_name', help='file name for saving model file.', required=True)
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"

    args = parser.parse_args()

    logger = get_logger("NERCRF")

    mode = args.mode
    test_path = args.test
    embedding = args.embedding
    embedding_path = args.embedding_dict

    embedd_dict, embedd_dim = utils.load_embedding_dict(embedding, embedding_path)

    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet, pos_alphabet, \
    chunk_alphabet, ner_alphabet = conll03_data.create_alphabets("data/alphabets/ner_crf/", train_path, data_paths=[dev_path, test_path],
                                                                 embedd_dict=embedd_dict, max_vocabulary_size=50000)

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    logger.info("Chunk Alphabet Size: %d" % chunk_alphabet.size())
    logger.info("NER Alphabet Size: %d" % ner_alphabet.size())

    logger.info("Reading Data")
    device = torch.device('cuda') if args.cuda else torch.device('cpu')

    data_train = conll03_data.read_data_to_tensor(train_path, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, device=device)
    num_data = sum(data_train[1])
    num_labels = ner_alphabet.size()

    data_test = conll03_data.read_data_to_tensor(test_path, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, device=device)

    writer = CoNLL03Writer(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet)

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / embedd_dim)
        table = np.empty([word_alphabet.size(), embedd_dim], dtype=np.float32)
        table[conll03_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in embedd_dict:
                embedding = embedd_dict[word]
            elif word.lower() in embedd_dict:
                embedding = embedd_dict[word.lower()]
            else:
                embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('oov: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table()
    logger.info("constructing network...")

    char_dim = args.char_dim
    window = 3
    num_layers = args.num_layers
    tag_space = args.tag_space
    initializer = nn.init.xavier_uniform_
    if args.dropout == 'std':
        network = BiRecurrentConvCRF(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), num_filters, window, mode, hidden_size, num_layers, num_labels,
                                     tag_space=tag_space, embedd_word=word_table, p_in=p_in, p_out=p_out, p_rnn=p_rnn, bigram=bigram, initializer=initializer)
    else:
        network = BiVarRecurrentConvCRF(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), num_filters, window, mode, hidden_size, num_layers, num_labels,
                                        tag_space=tag_space, embedd_word=word_table, p_in=p_in, p_out=p_out, p_rnn=p_rnn, bigram=bigram, initializer=initializer)

    network = network.to(device)

    model_path = os.join(model_dir, model_file)
    network.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        tmp_filename = 'tmp/%s_analyze' % (str(uid))
        writer.start(tmp_filename)

        for batch in conll03_data.iterate_batch_tensor(data_test, batch_size):
            word, char, pos, chunk, labels, masks, lengths = batch
            preds, _ = network.decode(word, char, target=labels, mask=masks, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
            writer.write(word.cpu().numpy(), pos.cpu().numpy(), chunk.cpu().numpy(), preds.cpu().numpy(), labels.cpu().numpy(), lengths.cpu().numpy())
        writer.close()
        test_acc, test_precision, test_recall, test_f1 = evaluate(tmp_filename)

        print("Test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (test_acc, test_precision, test_recall, test_f1, best_epoch))


if __name__ == '__main__':
    main()
