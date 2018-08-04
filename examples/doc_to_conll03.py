from __future__ import print_function

__author__ = 'duc'
"""
Generate ConLL03 format from a document.
"""

import spacy
import argparse
import warnings
from oppnlp.data.prep.ner import conll03_to_neuronlp2

nlp = spacy.load('en')

def _doc_to_conll03(doc):
  output = []
  num_sents = 0
  for sent in doc.sents:
    for token in sent:
      if token.is_space:
         continue
      line = '{} {} O O'.format(token.text, token.tag_)
      output.append(line)
    output.append('\n')
    num_sents += 1
    if num_sents > 1:
      warnings.warn('NLP separates 1 line into multiple sentences. Assume 1 sentence per line')
  return output

def doc_to_conll03(doc):
  return '\n'.join(_doc_to_conll03(doc))

def write_to_output(input_file_path, conll03_sents):
  output_file_path = input_file_path + '.conll03'
  with open(output_file_path, 'w') as f:
    f.write(''.join(conll03_sents))
  return output_file_path


def convert(input_file_path):
  output_texts = []
  with open(input_file_path, 'r') as f:
    for line in f:
        doc = nlp(line.strip())
        output_texts.append(doc_to_conll03(doc))
  return write_to_output(input_file_path, output_texts)


def main():
  parser = argparse.ArgumentParser(description='Convert a list of sentences into CoNLL03 format.')
  parser.add_argument('--input', help='file path of the input sentences.', required=True)
  args = parser.parse_args()
  input_file_path = args.input

  output_file_path = convert(input_file_path)
  conll03_to_neuronlp2.transform(output_file_path, output_file_path + '.neuronlp2')


if __name__ == '__main__':
  main()
