# PassRVAE

## This repo covers the implementation for our paper PassRVAE.

## Environments
Python 3.10.12 \
Pytorch 2.0.1+cu118 

## Datasets
In our experiments, datasets include RockYou and 4iQ. To get passwords of different composition policies (1class8, 2class8, 3class8, 4class8, 3class12 and 1class16), we use the following regular expressions. 
### passwords with at least one category class (uppercase letters, lowercase letters, symbols, digits)
grep -a -E '[[:upper:]]|[[:lower:]]|[[:digit:]]|[[:punct:]]'
### passwords with at least two category classes (uppercase letters, lowercase letters, symbols, digits)
grep -a -E '([[:upper:]].*[[:lower:]])|([[:upper:]].*[[:punct:]])|([[:upper:]].*[[:digit:]])|([[:digit:]].*[[:punct:]])|([[:lower:]].*[[:digit:]])|([[:lower:]].*[[:punct:]])' 
### passwords with at least three category classes (uppercase letters, lowercase letters, symbols, digits)
grep -a -E '([[:upper:]].*[[:lower:]].*[[:digit:]])|([[:upper:]].*[[:lower:]].*[[:punct:]])|([[:upper:]].*[[:digit:]].*[[:punct:]])|([[:lower:]].*[[:digit:]].*[[:punct:]])'
### passwords with at least four category classes (uppercase letters, lowercase letters, symbols, digits)
grep -a -E '[[:upper:]].*[[:lower:]].*[[:digit:]].*[[:punct:]]'

In the following steps, we choose the 20:80 split of 4class8 of 4iQ as an example.
## Download code
```
git clone https://github.com/yujiaxiao/PassRVAE PassRVAE
cd PassRVAE
```
## Train
```
python3 train.py
```
Output:\
TRAIN preprocessed file not found at ./data/4iq-4class8-20%-train.json. Creating new.\
Vocablurary of 114 keys created.\
VALID preprocessed file not found at ./data/4iq-4class8-20%-valid.json. Creating new.\
preprocess data time 15.288792371749878\
PassRVAE(\
  (embedding): Embedding(114, 300)\
  (embedding_dropout): Dropout(p=0.1, inplace=False)\
  (encoder_rnn): GRU(300, 256, num_layers=3, batch_first=True)\
  (decoder_rnn): GRU(300, 256, num_layers=3, batch_first=True)\
  (hidden2mean): Linear(in_features=768, out_features=128, bias=True)\
  (hidden2logv): Linear(in_features=768, out_features=128, bias=True)\
  (latent2hidden): Linear(in_features=128, out_features=768, bias=True)\
  (outputs2vocab): Linear(in_features=256, out_features=114, bias=True)\
)\
TRAIN Epoch 00/20, Mean ELBO   28.5835\
Model saved at ./bin/2023-Oct-02-12:56:35/E0.pytorch\
VALID Epoch 00/20, Mean ELBO   35.1942\
TRAIN Epoch 01/20, Mean ELBO   33.9293\
Model saved at ./bin/2023-Oct-02-12:56:35/E1.pytorch\
VALID Epoch 01/20, Mean ELBO   33.0856\
TRAIN Epoch 02/20, Mean ELBO   32.6987\
Model saved at ./bin/2023-Oct-02-12:56:35/E2.pytorch\
VALID Epoch 02/20, Mean ELBO   32.5687\
TRAIN Epoch 03/20, Mean ELBO   32.2161\
Model saved at ./bin/2023-Oct-02-12:56:35/E3.pytorch\
VALID Epoch 03/20, Mean ELBO   32.3595\
TRAIN Epoch 04/20, Mean ELBO   31.9364\
Model saved at ./bin/2023-Oct-02-12:56:35/E4.pytorch\
VALID Epoch 04/20, Mean ELBO   32.2145\
TRAIN Epoch 05/20, Mean ELBO   31.7499\
Model saved at ./bin/2023-Oct-02-12:56:35/E5.pytorch\
VALID Epoch 05/20, Mean ELBO   32.1876\
TRAIN Epoch 06/20, Mean ELBO   31.6235\
Model saved at ./bin/2023-Oct-02-12:56:35/E6.pytorch\
VALID Epoch 06/20, Mean ELBO   32.1332\
TRAIN Epoch 07/20, Mean ELBO   31.5345\
Model saved at ./bin/2023-Oct-02-12:56:35/E7.pytorch\
VALID Epoch 07/20, Mean ELBO   32.0708\
TRAIN Epoch 08/20, Mean ELBO   31.4716\
Model saved at ./bin/2023-Oct-02-12:56:35/E8.pytorch\
VALID Epoch 08/20, Mean ELBO   32.0599\
TRAIN Epoch 09/20, Mean ELBO   31.4336\
Model saved at ./bin/2023-Oct-02-12:56:35/E9.pytorch\
VALID Epoch 09/20, Mean ELBO   32.0248\
TRAIN Epoch 10/20, Mean ELBO   31.3976\
Model saved at ./bin/2023-Oct-02-12:56:35/E10.pytorch\
VALID Epoch 10/20, Mean ELBO   32.0790\
Early stopping
## Generate
Now you can use your trained model in the previous step to generate passwords. Or you can directly use our pretrained model to generate passwords.
```
python3 generate.py
```
Model loaded from ./pretrained/PassRVAE_4class8_20%.pytorch
