import torch.nn as nn

from transformers import AutoModel


def select_model(args):
    model = BlueBertForMultiLabelClassification(args)
    return model


def equip_multi_gpu(model, args):
    model = nn.DataParallel(model, device_ids=args.gpus)
    return model

class BlueBertForMultiLabelClassification(nn.Module):
    def __init__(self, args):
        super(BlueBertForMultiLabelClassification, self).__init__()

        assert args.model in ["bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
                              "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16",
                              "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
                              "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16"]
        
        print('Using model', args.model)
        self.bert = AutoModel.from_pretrained(args.model)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, args.num_aspects)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits