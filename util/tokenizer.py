"""
STEP ONE OF GATER

Transforms sentences into tokens or individual words and assigns an id to each token.
"""

import io
import sys
import yaml

from datetime import datetime
from torch.utils.data import random_split
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer as Tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from util.log import LOG

from icecream import ic

class Tokenizer():
    
    def __init__(self):
        self.tokenizer_en = None
        self.tokenizer_target = None

        self.create_tokenizers()

    def get_ds_iterator(self, training, lang):
        for data in training:
            yield data['translation'][lang]

    def create_tokenizers(self):
        LOG.info('Welcome to Gater 🐊')
        
        # get language option from terminal
        # TODO: change to pass language option from API
        language = sys.argv[1]

        # get full language name
        yaml_filename = "./config.yaml"
        with open(yaml_filename, 'r') as stream:
            yaml_data = yaml.safe_load(stream)
        
        supported_languages = yaml_data['supported_languages']
        
        if language not in supported_languages:
            LOG.critical('Language not supported.')
            LOG.critical(f'Current languages supported are: {', '.join(supported_languages.keys())}')
            exit(0)

        language_fullname = supported_languages[language]

        LOG.info(f'Starting English to {language_fullname} LLM.')

        language_fullname = language_fullname.lower()

        # load data into training, validation, and testing sets
        directory_path = Path("./gater_lib")
        directory_path.mkdir(parents=True, exist_ok=True)

        english_path = Path('./gater_lib/english/')
        english_path.mkdir(parents=True, exist_ok=True)
        
        directory_path = Path(f'./gater_lib/{language_fullname}')
        directory_path.mkdir(parents=True, exist_ok=True)

        LOG.info(f'Grabbing tokens for source and target langugages.')

        # dataset can be formatted <TARGET_LANGUAGE>-en or en-<TARGET_LANGAUGE>
        # want to ensure either is found
        language_dataset = [f'en-{language}', f'{language}-en']

        for dataset in language_dataset:
            try:
                training = load_dataset('Helsinki-NLP/opus-100', dataset, split='train')
                validation = load_dataset('Helsinki-NLP/opus-100', dataset, split='validation')
            except:
                pass

        raw_training, rt_to_skip = random_split(training, [1500, len(training) - 1500])
        raw_validation, vt_to_skip = random_split(validation, [1500, len(validation) - 1500])

        for lang in [('en', 'English'), (language, language_fullname)]:
            abbr, long = lang
            tokenizer = Tokenizers(BPE(unk_token='[UNK]'))
            trainer = BpeTrainer(min_frequency=2, special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer.train_from_iterator(self.get_ds_iterator(raw_training, abbr), trainer=trainer)
            tokenizer.save(f'./gater_lib/{long}/tokenizer_{abbr}.json')

        self.tokenizer_en = Tokenizers.from_file('./gater_lib/english/tokenizer_en.json')
        self.tokenizer_target = Tokenizers.from_file(f'./gater_lib/{language_fullname}/tokenizer_{language}.json')

        source_vocab_size = self.tokenizer_en.get_vocab_size()
        target_vocab_size = self.tokenizer_target.get_vocab_size()

        max_seq_len_source = 0
        max_seq_len_target = 0

        for data in raw_training:
            enc_ids = self.tokenizer_en.encode(data['translation']['en']).ids
            dec_ids = self.tokenizer_target.encode(data['translation'][language]).ids
            max_seq_len_source = max(max_seq_len_source, len(enc_ids))
            max_seq_len_target = max(max_seq_len_target, len(dec_ids))

        # write to yaml file
        yaml_data['tokens_last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        yaml_data['vocab_sizes']['en'] = max_seq_len_source
        yaml_data['vocab_sizes'][language] = max_seq_len_target

        with open(yaml_filename, 'w') as file:
            yaml.dump(yaml_data, file)

        LOG.info(f'Source length, {max_seq_len_source} and target length, {max_seq_len_target}, updated in config.yaml file.')
