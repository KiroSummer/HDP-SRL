# Should point to the srlconll library.
SRLPATH="../data/"

perl $SRLPATH/conll09-chinese/eval09.pl -q -g $1 -s $2

