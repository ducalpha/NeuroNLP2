from __future__ import print_function

__author__ = 'duc'
"""
Generate ConLL03 format from a document.
"""

import spacy
import argparse


def _doc_to_conll03(doc):
  output = []
  for sent in doc.sents:
    for token in enumerate(sent):
      line = '{} O O O O'.format(i, token.text)
      output.append(line)
    output.append('\n')
  return output

def doc_to_conll03(doc):
  return '\n'.join(_doc_to_conll03(doc))

def write_to_output(input_file_path, output_text):
  output_file_path = input_file_path + '.conll03'
  with open(output_file_path, 'w') as f:
    f.write(output_text)


def convert(input_file_path):
  with open(input_file_path, 'r') as f:
    text = f.read()
  nlp = spacy.load('en')
  doc = nlp(text)
  conll03_text = doc_to_conll03(doc)
  write_to_output(input_file_path, conll03_text)


def main()
  parser = argparse.ArgumentParser(description='Tuning with bi-directional RNN-CNN-CRF')
  parser.add_argument('--input', help='file path of the input sentences.', required=True)
  args = parser.parse_args()
  input_file_path = args.input()

  convert(input_file_path)


if __name__ == '__main__':
  main()
  # todo: call convert conll03 to neuronlp2
