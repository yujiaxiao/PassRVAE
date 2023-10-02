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

## Download code
```
git clone https://github.com/yujiaxiao/PassRVAE PassRVAE
cd PassRVAE
```
## Train

## Generate
