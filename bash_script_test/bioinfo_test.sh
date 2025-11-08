#!/usr/bin/env bash
blastn -h       | head -n 3 > blasn_test.txt
nhmmer -h       | head -n 3 > nhmmer_test.txt
clustalo -h     | head -n 3 > clustalo_test.txt
seqkit -h       | head -n 3 > seqkit_test.txt
mafft --version > mafft_test.txt
samtools --version | head -n 1 > samtools_test.txt
esearch -version > esearch_test.txt
