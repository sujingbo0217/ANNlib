# Filtered Vamana

## Instructions

```shell
# Download git module and submodules
git clone https://github.com/sujingbo0217/ANNlib.git
git checkout filter
git submodule init && git submodule update

git checkout research
git submodule update

# Download datasets
mkdir data && cd data
mkdir data && cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
tar -xvzf sift.tar.gz
tar -xvzf gist.tar.gz
cd ../..

# Label generation
cd test
sh run_label_generator.sh

# Run experiment
sh run.sh
```