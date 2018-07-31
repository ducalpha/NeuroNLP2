#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python examples/analyze_NERCRF.py \
 --test "data/OPP-115/opp115.test.bio.conll03" \
 --model_dir "examples/" --model_name 'nercrf_network.pt'
