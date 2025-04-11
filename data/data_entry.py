from data import tokenization
import csv
import os
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, abstract_id=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.abstract_id = abstract_id

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True,
                 abstract_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example
        self.abstract_id = abstract_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class HoCProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""    

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, args):
        """See base class."""
        label_list = []
        # num_aspect=FLAGS.num_aspects
        aspect_value_list = args.aspect_value_list
        num_aspects = args.num_aspects
        for i in range(num_aspects):
            for value in aspect_value_list:
                label_list.append(str(i) + "_" + str(value))
        return label_list  # [ {'0_-2': 0, '0_-1': 1, '0_0': 2, '0_1': 3,....'19_-2': 76, '19_-1': 77, '19_0': 78, '19_1': 79}]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            # if set_type == "test":
            #  text_a = tokenization.convert_to_unicode(line[1])
            #  label = "0"
            # else:
            #  text_a = tokenization.convert_to_unicode(line[3])
            #  label = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            text_a = tokenization.convert_to_unicode(line[1])
            abstract_id = line[2].split('_')[0]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, abstract_id=abstract_id))
        return examples

class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels, abstract_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.abstract_ids = abstract_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx].clone().detach(),
            "attention_mask": self.attention_mask[idx].clone().detach(),
            "labels": self.labels[idx].float(), # float for multi-label classification
            "abstract_ids": self.abstract_ids[idx]
        }

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    # print("label_map:",label_map,";length of label_map:",len(label_map))
    label_id = None
    if "," in example.label:  # multiple label
        # get list of label
        label_id_list = []
        label_list = example.label.split(",")
        for label_ in label_list:
            label_id_list.append(label_map[label_])
        label_id = [0 for l in range(len(label_map))]
        for j, label_index in enumerate(label_id_list):
            label_id[label_index] = 1
    else:  # single label
        label_id = label_map[example.label]
    """     if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % (example.guid))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            if "," in example.label: tf.logging.info("label: %s (id_list = %s)" % (str(example.label),
                                                                                str(
                                                                                    label_id_list)))  # if label_id is a list, try print multi-hot value: label_id_list
            tf.logging.info("label: %s (id = %s)" % (str(example.label), str(label_id)))  # %d """

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        abstract_id = example.abstract_id,
        is_real_example=True)
    return feature

def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer):
    """Convert a set of `InputExample`s to a TFRecord file."""

    token_vectors = []
    mask_vectors = []
    label_vectors = []
    abstract_ids = []

    for (ex_index, example) in tqdm(enumerate(examples), total=len(examples), desc="Processing"):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)


        #features = collections.OrderedDict()
        #features["input_ids"] = create_int_feature(feature.input_ids)
        #features["input_mask"] = create_int_feature(feature.input_mask)
        #features["segment_ids"] = create_int_feature(feature.segment_ids)

        # if feature.label_id is already a list, then no need to add [].
        #if isinstance(feature.label_id, list):
            #label_ids = feature.label_id
        #label_ids = [feature.label_id]
        #features["label_ids"] = create_int_feature(label_ids)
        #features["is_real_example"] = create_int_feature(
            #[int(feature.is_real_example)])

        #tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        token_vectors.append(torch.tensor(feature.input_ids))
        mask_vectors.append(torch.tensor(feature.input_mask))
        label_vectors.append(torch.tensor(feature.label_id[1::2]))
        abstract_ids.append(feature.abstract_id)

    return torch.stack(token_vectors), torch.stack(mask_vectors), torch.stack(label_vectors), abstract_ids

def label_abstract_reduce(labels, abstract_ids):
    result = defaultdict(lambda:0)
    for abstract_id, label in zip(abstract_ids, labels):
        result[abstract_id] |= label
    return result
 
def select_loader(args, mode='train'):
    # usually we need loader in training, and dataset in eval/test
    processor = HoCProcessor()
    label_list = processor.get_labels(args)
    tokenizer = tokenization.FullTokenizer(args)

    if mode=='train':
        examples = processor.get_train_examples(args.data_dir)
    elif mode=='dev':
        examples = processor.get_dev_examples(args.data_dir)
    if mode=='test':
        examples = processor.get_test_examples(args.data_dir)

    input_ids, attention_mask, labels, abstract_ids = file_based_convert_examples_to_features(
        examples, label_list, args.max_seq_length, tokenizer)
    
    
    dataset = TextDataset(input_ids, attention_mask, labels, abstract_ids)

    if mode=='train':
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    
    return loader

def get_labels(args, mode='train'):
    # usually we need loader in training, and dataset in eval/test
    processor = HoCProcessor()
    label_list = processor.get_labels(args)
    tokenizer = tokenization.FullTokenizer(args)

    if mode=='train':
        examples = processor.get_train_examples(args.data_dir)
    elif mode=='dev':
        examples = processor.get_dev_examples(args.data_dir)
    if mode=='test':
        examples = processor.get_test_examples(args.data_dir)

    input_ids, attention_mask, labels, abstract_ids = file_based_convert_examples_to_features(
        examples, label_list, args.max_seq_length, tokenizer)
    
    
    return [example.text_a for example in examples], labels.numpy(), abstract_ids