# FLAN-T5-Small for Text Normalization in Text-to-Speech Applications

This repository contains the code and scripts used to fine-tune the [FLAN-T5-small](https://huggingface.co/google/flan-t5-small) model for the task of text normalization, specifically for text-to-speech (TTS) applications. The model is fine-tuned on the [Google Text Normalization challenge dataset](https://www.aclweb.org/anthology/P17-2032/) by Sproat & Jaitly (2017).

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model](#model)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Full paper](#full_paper)
- [Miscellaneous](#misc)


## Overview
Text normalization is an essential step in TTS systems, converting raw input text into a format suitable for speech synthesis. For example, converting "15th June" to "the fifteenth of June" or "12:30 PM" to "twelve thirty p m". This repository demonstrates how to fine-tune a pre-trained FLAN-T5-small model to handle text normalization tasks effectively.

## Dataset
The model is trained using the Google Text Normalization challenge dataset introduced by Sproat & Jaitly (2017). The dataset consists of pairs of verbalized and non-verbalized text, making it ideal for training models to handle text normalization in various contexts, including dates, numbers, addresses, and more.

The full dataset, consisting of English, Polish and Russian datasets can be found in this link: https://www.kaggle.com/datasets/richardwilliamsproat/text-normalization-for-english-russian-and-polish

## Model
The base model used is the [FLAN-T5-small](https://huggingface.co/google/flan-t5-small), a lightweight variant of T5, fine-tuned on a variety of instruction-following tasks. This model is fine-tuned for text normalization, focusing on handling specific TTS needs.

## Preprocessing
Since each line in each file contains a pair of pre-verbalized and verbalized words, I preprocess the data in the way to make each line a pair of preverbalized and verbalized sentences. I essentially consider the task of text normalization as machine translation. Hence, the format of preverbalized - verbalized sentences is needed. 

Preprocessing is a folder composed of files, each responsible for a step in the processing pipeline as follows:

Step 1: Turn <self> in the cell in the third column into the corresponding non-normalized word in the cell in the second column. Turn sil in the cell in the third column into the corresponding punctuation mark in the cell in the second column

Step 2: Append the words line by line to make trios of classes of prenormalized words, preverbalized sentence, and verbalized sentence

Step 3: Make a data frame with 3 columns, namely “Classes”, “Preverbalized sentence”, and “Verbalized sentence” representing the trios from Step 2


## Training
To fine-tune the model, use the official_ft.py script.

Training was done on 1 NVIDIA V100 GPU. 

## Inference
To make an inference after the model has been trained for a checkpoint, use the full_ft_inference.py script.

## Miscellaneous
To complete this project, I also did extensive data analysis of the whole dataset. The objective of data analysis in this project is to discover the distribution of each non-standard-word class in each file, and in each language presented in the whole dataset. 

## Evaluation
The metric of evaluation is accuracy. 

## Full paper
Full paper is available upon request. The full paper is an 8000-word report, completed as my dissertation for the MSc. Speech and Language Processing at the University of Edinburgh. 
