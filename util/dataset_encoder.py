import torch

from torch.utils.data import Dataset
from util.tokenizer import Tokenizer

class EncodedDataset(Dataset):
    def __init__(self, raw_datdaset, max_seq_len):
        """
        Transforms raw datasets to encoded datsets to be processed by the model

        TL;DR: use this code call:
            # create a dataloader to use for model training and validation
            train_ds = EncodeDataset(raw_train_dataset, max_seq_len)
            val_ds = EncodeDataset(raw_validation_dataset, max_seq_len)

            train_dataloader = DataLoader(train_ds, batch_size = 5, shuffle = True)
            val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)
        """
        super().__init__()
        self.raw_dataset = raw_datdaset
        self.mex_seq_len = max_seq_len

        # TODO: add check for filename 
        tokenizer = Tokenizer()
        self.tokenizer_en = tokenizer.tokenizer_en
        self.tokenizer_target = tokenizer.tokenizer_target

    def __len__(self):
        return len(self.raw_dataset)
    
    def __getitem__(self, index, language):
        # fetch single data point for given index, contains both english and target language
        raw_text = self.raw_dataset[index]

        # seperate english and target text tokenizers
        source_text = raw_text['translation']['en']
        target_text = raw_text['translation'][language]

        # encoding source and target texts with tokenizers
        source_text_encoded = self.tokenizer_en.encode(source_text).ids
        target_text_encoded = self.tokenizer_target.encode(target_text).ids

        # convert CLS, SEP, and PAD tokens to their corresponding index id 
        CLS_ID = torch.tensor([self.tokenizer_target.token_to_id('[CLS]')], dtype=torch.int64)
        SEP_ID = torch.tensor([self.tokenizer_target.token_to_id('[SEP]')], dtype=torch.int64)
        PAD_ID = torch.tensor([self.tokenizer_target.token_to_id('[PAD]')], dtype=torch.int64)

        # to train, the sequence length of each input should be equal max seq length
        num_source_padding = self.max_seq_len - len(source_text_encoded) - 2
        num_target_padding = self.max_seq_len - len(target_text_encoded) - 1 

        encoder_padding = torch.tensor([PAD_ID] * num_source_padding, dtype=torch.int64)
        decoder_padding = torch.tensor([PAD_ID] * num_target_padding, dtype=torch.int64)

        # encoder_input has the first token as start of sentence - CLS_ID, 
        # followed by source encoding which is followed by end of sentence 
        # token - SEP. To reach required max_seq_len, additional PAD token
        # added at end. There is no SEP for decoder_input
        encoder_input = torch.cat([CLS_ID, torch.tensor(source_text_encoded, dtype=torch.int64), SEP_ID, encoder_padding], dim=0)
        decoder_input = torch.cat([CLS_ID, torch.tensor(source_text_encoded, dtype=torch.int64), encoder_padding], dim=0)

        # target_label is required for the loss calculation during training to 
        # compare between the predicted and target label. target_label has the
        # first token as target encoding followed by actual target encoding.
        # There is no start of sentense - CLS in target label. To reach the 
        # required max_seq_len, additional PAD token will be added at the end
        target_label = torch.cat([torch.tensor(target_text_encoded, dtype=torch.int64), SEP_ID, decoder_padding], dim=0)

        # Since there is extra padding token with input encoding, the token
        # does not need to be trained by the model. An encoder mask will 
        # nullify the padding value prior to producing output of self attention
        # in encoder block
        encoder_mask = (encoder_input != PAD_ID).unsqueeze(0).unsqueeze(0).int()

        # Minimize a token influence future tokens during decoding. Adds a 
        # casual mask during masked multihead
        decoder_mask = (decoder_input != PAD_ID).unsqueeze(0).unsqueeze(0).int()

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            "target_label": target_label,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            'source_text': source_text,
            'target_text': target_text,
        }
    
def casual_mask(size):
    """
    Ensures any token coming in after current token will be masked.
    The value will be replaced by -infinity then converted to zero
    or near zero after softmax operation. The model will ignore
    these values or won't learn anything.
    """
    # creating square matrix of size 'size x size' filled with ones
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
