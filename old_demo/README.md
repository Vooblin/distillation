# Distillation Demo

Installation:
1. `./download_requirements.sh`
2. `./rundocker.sh`

Main Information:
* Datasets: IMDB
* Teacher models: BERT
* Student models: BiLSTM, BiGRU, Attention, Transformer

To run distillation:
1. Run script to download and preprocess input data.  
   Example --- `./scripts/data/imdb.sh`
2. Run script to fine tune Base BERT.  
   Example --- `./scripts/teacher/bert_imdb.sh`
3. (Optional) Run script to train student model from scratch (to compare results with distillation).  
   Example --- `./scripts/student/attention_imdb.sh`
4. Run script to distill knowledge from teacher to student.  
   Example --- `./scripts/distillation/attention_bert_imdb.sh`
5. Results of steps 2-4 will be in directory `results`. Examples:
   * `results/bert_imdb.txt`
   * `results/attention_imdb.txt`
   * `results/distil_attention_bert_imdb.txt`

## Experiment results

Knowledge Distillation (KD) is a compression technique in which a small model is trained to reproduce the behavior of a larger model (or an ensemble of models).
Some last papers (<https://arxiv.org/abs/1903.12136>, <https://arxiv.org/pdf/1909.10351.pdf>) describe distilling knowledge from BERT to smaller models.
We reproduced this technique and we show the use BERT KD on the following models: BiLSTM, BiGRU, Attention, Transformer.
For demonstration we chose IMDB dataset, but we also made interfaces to run BERT KD on MNLI-m and Yelp5 datasets.
The results of experiments are shown in the table below.

#### IMDB

BERT Base: 86.79%

|Model, layers |Single|Distilled |
|--------------|------|----------|
|Attention, 1  |79.32%|**80.26%**|
|Attention, 2  |79.62%|**80.34%**|
|Transformer, 1|79.30%|**80.44%**|
|Transformer, 2|80.05%|**80.57%**|
|BiLSTM, 1     |74.87%|**79.30%**|
|BiLSTM, 2     |76.62%|**79.74%**|
|BiGRU, 1      |76.92%|**80.48%**|
|BiGRU, 2      |78.37%|**81.38%**|

## Further work directions

*  Conduct experiments of distilling knowledge from BERT Base to BERT Small
(for example, reproduce [DistilBERT](https://medium.com/huggingface/distilbert-8cf3380435b5) or [TinyBERT](https://arxiv.org/pdf/1909.10351.pdf))
*  Combine KD with other compression techniques, such as quantization or prunning