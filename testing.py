import torch
from icecream import ic
from torch.utils.data import Dataset, DataLoader

from util.tokenizer import Tokenizer
from util.dataset_encoder import EncodedDataset
from util.model import ModelBuilder
from util.log import LOG

max_seq_len = 155
tokenizer = Tokenizer()
raw_training, raw_validation = tokenizer.create_datasets()
tokenized_en, tokenized_target = tokenizer.create_tokenizers(raw_training)

LOG.info('Encoding datasets')
training_ds = EncodedDataset(raw_training, max_seq_len)
validation_ds = EncodedDataset(raw_validation, max_seq_len)

training_dataloader = DataLoader(training_ds, batch_size=5, shuffle=True)
validation_dataloader = DataLoader(validation_ds, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_model = ModelBuilder().build_model(tokenized_en.get_vocab_size(), tokenized_target.get_vocab_size(), max_seq_len, max_seq_len, d_model=512).to(device)
# ic(test_model)
LOG.info(f'Model Information: \nf{test_model}')
