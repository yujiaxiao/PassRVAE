import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader,Dataset
from collections import OrderedDict, defaultdict

from preprocess import PTB
from utils import to_var, idx2char, experiment_name
from model import PassRVAE
import torch

#from torchtext.data.utils import get_tokenizer
import time
from pytorchtools import EarlyStopping



def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    splits = ['train','valid'] + []#(['test'] if args.test else [])

    t1=time.time()

    datasets = OrderedDict()
    for split in splits:
        datasets[split] = PTB(
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ
        )
    t2=time.time()
    print("preprocess data time",t2-t1)

    params = dict(
        vocab_size=datasets['train'].vocab_size,
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
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
    #print("params",params)

    model = PassRVAE(**params)

    '''
    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)
    '''

    if torch.cuda.is_available():
        model = model.cuda()
    

    print(model)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    def kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-k * (step - x0))))
        elif anneal_function == 'linear':
            return min(1, step / x0)

    NLL = torch.nn.NLLLoss(ignore_index=datasets['train'].pad_idx, reduction='sum')

    def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0):
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        NLL_loss = NLL(logp, target)

        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)

        return NLL_loss, KL_loss, KL_weight

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0

    early_stopping = EarlyStopping(args.patience, verbose=False)

    for epoch in range(args.epochs):
        for split in splits:
            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split == 'train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available(),
                drop_last=True
            )

            tracker = defaultdict(tensor)

            if split == 'train':
                model.train()
            else:
                model.eval()

            for iteration, batch in enumerate(data_loader):
                batch_size = batch['input'].size(0)

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                logp, mean, logv, z = model(batch['input'], batch['length'])

                NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
                                                       batch['length'], mean, logv,
                                                       args.anneal_function, step, args.k, args.x0)

                loss = (NLL_loss + KL_weight * KL_loss) / batch_size

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.data.view(1, -1)), dim=0)

            print("%s Epoch %02d/%i, Mean ELBO %9.4f" % (split.upper(), epoch, args.epochs, tracker['ELBO'].mean()))

        
            if split=='valid':
                #mean_valid_loss.append(tracker['ELBO'].mean())
                #print("mean_valid_loss",mean_valid_loss)
                early_stopping(tracker['ELBO'].mean(),model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            if split == 'train':
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % epoch)
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s" % checkpoint_path)

        if early_stopping.early_stop:
            break

    t3=time.time()
    print("training time",t3-t2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    

    parser.add_argument('-c', '--load_checkpoint', type=str,default="/content/gdrive/MyDrive/PassRVAE/bin/2023-Jul-31-05:22:22/E4.pytorch")
    
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--vocab_file', type=str, default="4iq-4class8-20%-vocab.json")
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--load_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=31)
    parser.add_argument('--min_occ', type=int, default=1)
    #parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=20)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=3)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=128)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.1)

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='./logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='./bin')
    parser.add_argument('-patience','--patience',type=int,default=1)

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.anneal_function in ['logistic', 'linear']
    assert 0 <= args.word_dropout <= 1

    main(args)
