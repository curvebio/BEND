#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python scripts/train_on_task.py --config-name gene_finding embedder=dnabert2-bs-seq 2>&1 | tee logs/dnabert2-bs-seq_gene_finding.log &

CUDA_VISIBLE_DEVICES=2 python scripts/train_on_task.py --config-name enhancer_annotation embedder=dnabert2-bs-seq 2>&1 | tee logs/dnabert2-bs-seq_enhancer_annotation.log &

CUDA_VISIBLE_DEVICES=1 python scripts/train_on_task.py --config-name gene_finding embedder=hyenadna-bs-seq 2>&1 | tee logs/hyenadna-bs-seq_gene_finding.log &

CUDA_VISIBLE_DEVICES=0 python scripts/train_on_task.py --config-name enhancer_annotation embedder=hyenadna-bs-seq 2>&1 | tee logs/hyenadna-bs-seq_enhancer_annotation.log &

CUDA_VISIBLE_DEVICES=2 python scripts/train_on_task.py --config-name histone_modification embedder=dnabert2-bs-seq 2>&1 | tee logs/dnabert2-bs-seq_histone_modification.log &

CUDA_VISIBLE_DEVICES=3 python scripts/train_on_task.py --config-name histone_modification embedder=hyenadna-bs-seq 2>&1 | tee logs/hyenadna-bs-seq_histone_modification.log &


CUDA_VISIBLE_DEVICES=1 python scripts/train_on_task.py --config-name cpg_methylation embedder=dnabert2-bs-seq 2>&1 | tee logs/dnabert2-bs-seq_cpg_methylation.log &

CUDA_VISIBLE_DEVICES=0 python scripts/train_on_task.py --config-name cpg_methylation embedder=hyenadna-bs-seq 2>&1 | tee logs/hyenadna-bs-seq_cpg_methylation.log &





CUDA_VISIBLE_DEVICES=2 python scripts/train_on_task.py --config-name chromatin_accessibility embedder=dnabert2-bs-seq 2>&1 | tee logs/dnabert2-bs-seq_chromatin_accessibility.log &

CUDA_VISIBLE_DEVICES=2 python scripts/train_on_task.py --config-name cpg_methylation embedder=dnabert2-bs-seq 2>&1 | tee logs/dnabert2-bs-seq_cpg_methylation.log &


