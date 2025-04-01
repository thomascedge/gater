"""
STEP ONE OF GATER

Transforms sentences into tokens or individual words and assigns an id to each token.
"""

import io
import sys
import yaml
import glob

from datetime import datetime
from torch.utils.data import random_split
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer as Tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from .log import LOG

class Tokenizer():
    def __init__(self):
        self.language = self._welcome()
        self.language_fullname = None
        self.yaml_data = None
        self.yaml_filename = None

    def get_ds_iterator(self, training, lang):
        for data in training:
            yield data['translation'][lang]

    def create_datasets(self):
        language_fullname = self._convert_language(self.language)        

        LOG.info(f'Starting English to {language_fullname} LLM.')

        self.language_fullname = language_fullname.lower()

        # load data into training, validation, and testing sets
        directory_path = Path("./gater_lib")
        directory_path.mkdir(parents=True, exist_ok=True)

        english_path = Path('./gater_lib/english/')
        english_path.mkdir(parents=True, exist_ok=True)
        
        directory_path = Path(f'./gater_lib/{self.language_fullname}')
        directory_path.mkdir(parents=True, exist_ok=True)

        LOG.info(f'Grabbing tokens for source and target langugages.')

        # dataset can be formatted <TARGET_LANGUAGE>-en or en-<TARGET_LANGAUGE>
        # want to ensure either is found
        language_dataset = [f'en-{self.language}', f'{self.language}-en']

        for dataset in language_dataset:
            try:
                training = load_dataset('Helsinki-NLP/opus-100', dataset, split='train')
                validation = load_dataset('Helsinki-NLP/opus-100', dataset, split='validation')
            except:
                pass

        raw_training, rt_to_skip = random_split(training, [1500, len(training) - 1500])
        raw_validation, vt_to_skip = random_split(validation, [1500, len(validation) - 1500])

        return raw_training, raw_validation

    def create_tokenizers(self, raw_training_dataset):
        for lang in [('en', 'English'), (self.language, self.language_fullname)]:
            abbr, long = lang
            tokenizer = Tokenizers(BPE(unk_token='[UNK]'))
            trainer = BpeTrainer(min_frequency=2, special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer.train_from_iterator(self.get_ds_iterator(raw_training_dataset, abbr), trainer=trainer)
            tokenizer.save(f'./gater_lib/{long}/tokenizer_{abbr}.json')

        tokenizer_en = Tokenizers.from_file('./gater_lib/english/tokenizer_en.json')
        tokenizer_target = Tokenizers.from_file(f'./gater_lib/{self.language_fullname}/tokenizer_{self.language}.json')

        source_vocab_size = tokenizer_en.get_vocab_size()
        target_vocab_size = tokenizer_target.get_vocab_size()

        max_seq_len_source = 0
        max_seq_len_target = 0

        for data in raw_training_dataset:
            enc_ids = tokenizer_en.encode(data['translation']['en']).ids
            dec_ids = tokenizer_target.encode(data['translation'][self.language]).ids
            max_seq_len_source = max(max_seq_len_source, len(enc_ids))
            max_seq_len_target = max(max_seq_len_target, len(dec_ids))

        # write to yaml file
        self.yaml_data['tokens_last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.yaml_data['vocab_sizes']['en'] = max_seq_len_source
        self.yaml_data['vocab_sizes'][self.language] = max_seq_len_target

        with open(self.yaml_filename, 'w') as file:
            yaml.dump(self.yaml_data, file)

        LOG.info(f'Source length, {max_seq_len_source} and target length, {max_seq_len_target}, updated in config.yaml file.')

        return tokenizer_en, tokenizer_target

    def get_tokenizer_from_file(self):
        language, language_fullname = self._get_language()

        tokenizer_en = Tokenizers.from_file('./gater_lib/english/tokenizer_en.json')
        tokenizer_target = Tokenizers.from_file(f'./gater_lib/{language_fullname}/tokenizer_{language}.json')

        return tokenizer_en, tokenizer_target
    
    def _welcome(self):
        LOG.info('Welcome to Gater 🐊')
        
        # get language option from terminal
        # TODO: change to pass language option from API
        language = sys.argv[1]

        return language

    def _convert_language(self, language):
        # get full language name
        self.yaml_filename  = Path('./config.yaml').resolve()

        with open(self.yaml_filename, 'r') as stream:
            self.yaml_data = yaml.safe_load(stream)

        supported_languages = self.yaml_data['supported_languages']
        
        if self.language not in supported_languages:
            supported = ', '.join(supported_languages.keys())
            LOG.critical('Language not supported.')
            LOG.critical(f'Current languages supported are: {supported}')
            exit(0)

        language_fullname = supported_languages[self.language]

        return language_fullname