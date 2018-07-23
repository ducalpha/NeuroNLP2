#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python examples/analyze_NERCRF.py --gpu \
 --test "data/conll2003/english/eng.test.bioes.conll" \
 --model_path "examples/" --model_name 'nercrf_network.pt'
