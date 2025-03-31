HumanAI Foundation OCR Task - GSOC 2025

This repository contains the test results for the OCR task conducted by the HumanAI Foundation as part of GSoC 2025.

Model Overview

The model utilized for this task is qantev’s Spanish TrOCR, which has been pre-trained on printed Spanish documents. To enhance its performance on handwritten text, the model was fine-tuned using:

1.Rodrigo Dataset (handwritten text)

2.Synthetic Data generated using VRD Handwritten Text Generator

PEFT LoRA Adapters applied to the encoder and decoder attention modules

Evaluation Metrics
The fine-tuned model was evaluated using the following metrics:

1.Word Error Rate (WER)

2.Character Error Rate (CER)

3.BLEU Score

Transcriptions
The generated transcriptions are stored in the pred_transcriptions folder, which consists of two directories:

1.test_transcriptions/ – Contains transcriptions of the handwritten PDFs provided in the handwritten_test folder.

2.remaining_transcriptions/ – Contains transcriptions of the remaining PDF documents.

Each transcription is provided page-wise for every PDF.
