# Detection of alternative splicing: deep sequencing or deep learning?

## Description

This repository contains code to reproduce the results for the publication "Detection of alternative splicing: deep sequencing or deep learning?" by Hackl et al.

The study addresses the question: Can we utilize the vast repository of publicly available RNA-seq data for AS detection, despite often lacking the sequencing depth typically required? We show that sequence-based tools such as DeepSplice and SpliceAI show promising performance in retrieving novel and unannotated splice junctions, even when RNA-seq data are limited, but are not suitable for de novo splice junction detection. Our results demonstrate the potential of sequence-based tools for initial hypothesis development and as additional filters in standard RNA-seq pipelines, especially when sequencing depth is limited. Nonetheless, validation with higher sequencing depths remains essential for confirmation of splice events.

## Pipeline to reproduce results

Download all data from https://doi.org/10.5281/zenodo.16843445 and unpack into a data folder in the same directory.

Create the conda environment using the yml file provided: conda env create -f eval.yml

To get stats on the performance on the different evaluation scenarios, run python evaluation_scenarios.py
To reproduce the plots from the paper, run plots.ipynb.
To reproduce AlphaGenome results, run alphagenome.ipynb.
