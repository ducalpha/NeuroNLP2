#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python examples/analyze_NERCRF.py --mode LSTM --gpu \
 --test "data/OPP-115/opp115.test.bio.conll03" \
 --batch_size 16 --hidden_size 512 --num_layers 1 \
 --char_dim 30 --num_filters 30 --tag_space 128 \
 --dropout std --p_in 0.33 --p_rnn 0.33 0.5 --p_out 0.5 --bigram \
 --embedding glove --embedding_dict "data/glove/glove.6B/glove.6B.300d.gz" \
 --model_dir "examples/" --model_name 'nercrf_network.pt'
