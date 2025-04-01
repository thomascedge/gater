import torch 
from util.model import ModelBuilder
from util.tokenizer import Tokenizer

class Gater():
    def __init__(self, language:str):
        self.language = language
        self.language_fullname = Tokenizer._convert_language(language).lower()

    def start_model(self, user_input, max_seq_len, device):
        # validation user input text
        user_input = str(user_input).strip()

        # model define device, tokenizers, and model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer_en = Tokenizer.from_file('./english/tokenizer_en.json')
        tokenizer_target = Tokenizer.from_file(f'./{self.language}/tokenizer_{self.language}.json')

        # build model
        source_vocab_size, target_vocab_size = tokenizer_en.get_vocab_size(), tokenizer_target.get_vocab_size()
        model = ModelBuilder().build_model(source_vocab_size, target_vocab_size, max_seq_len, max_seq_len, d_model=512).to(device)

        # load specific checkpoint of model saved during training
        checkpoint_number = 9
        model_filename = f'./gater_lib/model_{checkpoint_number}.pt'
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])

        # begin inferencing
        model.eval()
        with torch.no_grad():
            # precompute encoder output and reuse it for every genereation step
            source_text_encoding = tokenizer_en.encode(user_input)
            source_text_encoding = torch.cat([
                torch.tensor([tokenizer_en.token_to_id('[CLS]')], dtype=torch.int64),
                torch.tensor(source_text_encoding.ids, dtype=torch.int64),
                torch.tensor([tokenizer_en.token_to_id('[SEP]')], dtype=torch.int64),
                torch.tensor([tokenizer_en.token_to_id('[PAD]')] * (max_seq_len - len(source_text_encoding.ids) - 2), dtype=torch.int64)
            ], dim=0).to(device)
            source_mask = (source_text_encoding != tokenizer_en.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
            encoder_output = model.encode(source_text_encoding, source_mask)

            # init the decoder input with sos token
            decoder_input = torch.empty(1, 1).fill_(tokenizer_target.token_to_id('[CLS]')).type_as(source_text_encoding).to(device)

            # generate word by word translation
            while decoder_input.size(1) < max_seq_len:
                # build mask for target and calculate output
                decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
                out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

                # project next token
                prob = model.project(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source_text_encoding).fill_(next_word.item()).to(device)], dim=1)

                # print the translate word
                print(f'{tokenizer_target.decode([next_word.item()])}', end=' ')

                # break if end of sentence token predicted
                if next_word == tokenizer_target.token_to_id('[SEP]'):
                    break
            
            # convert ids to tokens
            return tokenizer_target.decode(decoder_input[0].tolist())

    def translate(self, text):
        max_seq_len = 155
        device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.start_model(text, max_seq_len, device)


gater = Gater('fr')
english_text = 'I am going to the store tomorrow morning.'
translate_text = gater.translate(english_text)