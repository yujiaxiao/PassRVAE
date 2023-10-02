import os
import json
import torch
import argparse

from model import PassRVAE
from utils import to_var,  interpolate

def main(args):
    def idx2char(sample, i2w, eos_idx):
        char_str = ""
        for char_id in sample:
            #print(char_id)
            #print(w2i['<pad>'])
            if char_id==eos_idx:
                break
            if str(char_id) in i2w.keys():
                #print(i2w[str(char_id)])
                #print(str(char_id))
                char_str+= i2w[str(char_id)]
        return char_str
        
    with open(args.data_dir+'/4iq-4class8-20%-vocab.json', 'r') as file:
        vocab = json.load(file)
    w2i, i2w = vocab['w2i'], vocab['i2w']

    model = PassRVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
    )

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    with open('./data/PassRVAE_4class8-20%-generation_10_7.txt', 'w') as file1:
        for _ in range(1000):
            samples, z = model.inference(n=args.num_samples)
            samples = samples.squeeze().tolist()
            
            for sample in samples:
                char_str = idx2char(sample, i2w, eos_idx=w2i['<eos>'])
                file1.write(char_str + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str,default="./pretrained/PassRVAE_4class8_20%.pytorch")
    parser.add_argument('-n', '--num_samples', type=int, default=10000)

    parser.add_argument('-dd', '--data_dir', type=str, default='./data')
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=31)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.1)
    parser.add_argument('-ls', '--latent_size', type=int, default=128)
    parser.add_argument('-nl', '--num_layers', type=int, default=3)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)
