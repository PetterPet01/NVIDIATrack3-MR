# Pipeline Script

***Remember to run LLM Setup first***
## VADAR Setup
```bash
apt install libcurl4-openssl-dev -y
git clone https://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@github.com/PetterPet01/VADAR
gdown https://drive.google.com/uc?id=1Z5KlnUFLujVsmuTC7HDVjafyLilChs3Y -O ~/llamacpp.zip
unzip ~/llamacpp.zip
cd ~/VADAR
mkdir Data
cd Data
gdown https://drive.google.com/uc?id=1MPpA9JkZ5CmkyS_eRK7gU45gTehGgmy0 -O train_sample.zip
unzip train_sample.zip

cd ~/VADAR
python -m venv venv
source venv/bin/activate
sh setup.sh
```

## LLM Setup
```bash
pip install --upgrade pip
pip install uv

mkdir ~/sglang
cd ~/sglang

uv venv
. .venv/bin/activate
uv pip install "sglang[all]>=0.4.7.post1"

python3 -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-FP8 --context-length 10000 --mem-fraction-static 0.5
```