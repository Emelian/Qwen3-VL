python -m pip install --upgrade pip setuptools wheel

pip install "torch==2.7.*" "torchvision==0.22.*" "torchaudio==2.7.0" --extra-index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=4.57.0"
pip install deepspeed==0.17.1 torchcodec==0.2 peft==0.17.1 huggingface-hub==1.0.0.rc4
pip install qwen-vl-utils==0.0.14
pip install -U vllm
pip install autoawq
pip install timm
