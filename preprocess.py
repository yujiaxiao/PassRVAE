import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
from collections import defaultdict, Counter

from utils import OrderedCounter
from torch.utils.data import DataLoader

from collections import defaultdict
from collections import Counter
from collections import OrderedDict
import json
import io
import os
import numpy as np
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
import tempfile
import h5py

class PTB(Dataset):
    def __init__(self, data_dir, split, create_data, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 20)
        self.min_occ = kwargs.get('min_occ', 3)

        self.raw_data_path = os.path.join(data_dir, '4iq-4class8-20%-'+split+'.txt')
        self.data_file = '4iq-4class8-20%-'+split+'.json'
        self.vocab_file = '4iq-4class8-20%-vocab.json'

        if create_data:
            print("Creating new %s ptb data."%split.upper())
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new."%(split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        input_data = np.asarray(self.data[idx]['input'])
        target_data = np.asarray(self.data[idx]['target'])
        length = self.data[idx]['length']


        return {
            'input': input_data,
            'target': target_data,
            'length': length
        }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']
        
    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    def unk_idx(self):
        return self.w2i['<unk>']
    

    @property
    

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w


    def _load_data(self, vocab=True):

        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):
        
        if self.split == 'train':
            self._create_vocab()
        else:
            self._load_vocab()
        
        #self._load_vocab()

        data = defaultdict(dict)
        with open(self.raw_data_path, 'r') as file:
            for i, line in enumerate(file):
                char_str=line.strip()
                char_ids = [self.w2i[char] for char in char_str if char in self.w2i]
                char_ids.insert(0,self.w2i['<sos>'])
                char_ids.append(self.w2i['<eos>'])               
                length=len(char_ids)-1
                input_ids=char_ids[:length]+[self.w2i['<pad>']] * (self.max_sequence_length - length)
                target_ids=char_ids[1:]+[self.w2i['<pad>']] * (self.max_sequence_length - length)

                data[i]['input'] = input_ids
                data[i]['target'] = target_ids
                data[i]['length'] = length

        with open(os.path.join(self.data_dir, self.data_file), 'w') as data_file:
            json.dump(data, data_file)

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."
        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<sos>', '<eos>','<unk>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        with open(self.raw_data_path, 'r') as file:

            for i, line in enumerate(file):
                words = list(line.strip())
                w2c.update(words)

            for w, c in w2c.items():
                if c > self.min_occ and w not in special_tokens:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocablurary of %i keys created." %len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()









