from __future__ import print_function

import argparse

def conll03_to_html(conll03_file_path):
    output = []
    with open(conll03_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                output.append('<br>')
                continue
            parts = line.split()
            word = parts[1]
            ner = parts[4]
            if ner != 'O':
                output.append('<strong>{}</strong>'.format(word))
            else:
                output.append(word)
    output_html_file_path = conll03_file_path + '.html'
    output_html_str = ' '.join(output)
    # print(output_html_str)
    with open(output_html_file_path, 'w') as f:
        f.write(output_html_str)
    

def main():
    parser = argparse.ArgumentParser(description='Convert from ConLL03 to HTML')
    parser.add_argument('--file') # conll03_file_path
    args = parser.parse_args()
    conll03_to_html(args.file)

if __name__ == '__main__':
    main()
