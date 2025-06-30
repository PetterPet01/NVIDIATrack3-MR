mkdir -p models
cd models

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

apt install clang build-essential ninja-build cmake python3.10-dev -y
export C_INCLUDE_PATH="/usr/include/python3.10:${C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="/usr/include/python3.10:${CPLUS_INCLUDE_PATH}"

pip install "numpy<2"
apt-get install python3.10-dev -y
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 xformers==0.0.29.post3 --extra-index-url https://download.pytorch.org/whl/cu126 --force-reinstall

git clone https://github.com/PetterPet01/UniK3D-demo.git
cd UniK3D-demo

pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121
cd ..

pip install -r requirements.txt
pip install IPython sentencepiece rich
pip install 'accelerate'

pip install langchain
pip install -U openmim
mim install mmengine

pip install iopath pyequilib==0.3.0 albumentations einops open3d imageio
pip install https://github.com/zju3dv/Wis3D/releases/download/2.0.0/wis3d-2.0.0-py3-none-any.whl

pip uninstall numpy scipy scikit-learn open3d -y
pip install --no-cache-dir "numpy<1.25,>=1.21" # Example: NumPy 1.24.x
pip install --no-cache-dir "scipy>=1.9,<1.12"    # Example: SciPy 1.10.x or 1.11.x
pip install --no-cache-dir "scikit-learn>=1.1,<1.4" # Example: scikit-learn 1.2.x or 1.3.x
pip install --no-cache-dir open3d # Let open3d pick up these versions

pip install transformers
pip install pycocotools
pip install openai
pip install sentence_transformers
pip install spacy wordfreq "numpy<1.28.0,>=1.21.6"
python -m spacy download en_core_web_sm
pip install IPython pandas autopep8
