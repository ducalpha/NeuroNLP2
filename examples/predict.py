from __future__ import print_function

__author__ = 'duc'
""" Predict NER for sentences using LSTM-CNNs-CRF. """

import sys

sys.path.append(".")
sys.path.append("..")

import argparse
import uuid
import torch
from neuronlp2.io import get_logger, conll03_data, CoNLL03Writer

uid = uuid.uuid4().hex[:6]
logger = get_logger("NERCRF")


def _create_alphabets():
  # Set everything to None because the alphabets are supposed to be loaded from training phase.
  word_alphabet, char_alphabet, pos_alphabet, \
  chunk_alphabet, ner_alphabet = conll03_data.create_alphabets("data/alphabets/ner_crf/", None)

  logger.info("Word Alphabet Size: %d" % word_alphabet.size())
  logger.info("Character Alphabet Size: %d" % char_alphabet.size())
  logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
  logger.info("Chunk Alphabet Size: %d" % chunk_alphabet.size())
  logger.info("NER Alphabet Size: %d" % ner_alphabet.size())
  return word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet


def predict(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, network, data_test, batch_size):
  # Todo: no label
  writer = CoNLL03Writer(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet)
  with torch.no_grad():
    network.eval()  # This set dropout to an identity function.
    tmp_filename = 'tmp/%s_analyze' % (str(uid))
    writer.start(tmp_filename)

    for batch in conll03_data.iterate_batch_tensor(data_test, batch_size):
      word, char, pos, chunk, labels, masks, lengths = batch
      # TODO: no target.
      preds, _ = network.decode(word, char, target=labels, mask=masks, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
      writer.write(word.cpu().numpy(), pos.cpu().numpy(), chunk.cpu().numpy(), preds.cpu().numpy(),
                   labels.cpu().numpy(), lengths.cpu().numpy())
    writer.close()


def predict_ner(batch_size, model_path, test_file_path):
  logger.info("Creating Alphabets")
  word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet = _create_alphabets()

  logger.info("Reading Data")
  data_test = conll03_data.read_data_to_tensor(test_file_path, word_alphabet, char_alphabet,
                                               pos_alphabet, chunk_alphabet, ner_alphabet, device=torch.device('cuda'))

  logger.info("Constructing Network...")
  network = torch.load(model_path)
  predict(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, network, data_test, batch_size)


def main():
  parser = argparse.ArgumentParser(description='Make prediction with a trained model of bi-directional RNN-CNN-CRF')
  parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
  parser.add_argument('--model', help='Trained model', required=True)
  parser.add_argument('--test', help='Test sentences file path')  # "data/POS-penn/wsj/split1/wsj1.test.original"

  args = parser.parse_args()
  batch_size = args.batch_size
  model_path = args.model
  test_file_path = args.test

  predict_ner(batch_size, model_path, test_file_path)


if __name__ == '__main__':
  main()
