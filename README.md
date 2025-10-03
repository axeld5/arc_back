# arc_back
ARC Repo

To send .env to VM: scp -P XXXX .env user@YYYY:/path/to/destination/

To create .env in VM: echo "HF_TOKEN=" > .env
To create distant kernel uv run --active python -m ipykernel install --user --name project --display-name "Python (project)"

git clone https://github.com/axeld5/arc_back.git && cd arc_back
sudo snap install astral-uv --classic && sudo uv sync
sudo uv pip install unsloth unsloth-zoo
sudo uv pip install triton && sudo uv pip install kernels
sudo uv pip install openai-harmony
sudo uv pip install --force-reinstall vllm --torch-backend=auto

To serve vllm model
CUDA_VISIBLE_DEVICES=0 sudo uv run vllm serve qwen3_4b_singled_out_sft/merged --tensor-parallel-size 1 --max-model-len 32768