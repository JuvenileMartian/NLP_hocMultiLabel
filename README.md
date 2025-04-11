# Training and Evaluation
python train.py --vocab_file ./vocab.txt --model "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12" 

--vocab_file
  ./vocab.txt
  ./vocab_extip.txt

--model
  "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12"
  "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16"
  "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
  "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16"

# Typical output
The value of loss function evaluated on dev set and test set for each epoch, in system prompt.
The heatmap of confusion, in ./figs folder.
Tensorboard record, in ./checkpoints folder
