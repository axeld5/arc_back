# arc_back
ARC Repo

To send .env to VM: scp -P XXXX .env user@YYYY:/path/to/destination/

To create .env in VM: echo "HF_TOKEN=" > .env
To create distant kernel uv run --active python -m ipykernel install --user --name project --display-name "Python (project)"

git clone https://github.com/axeld5/arc_back.git && cd arc_back
sudo snap install astral-uv --classic && sudo uv sync
sudo uv pip install --force-reinstall vllm --torch-backend=auto
sudo uv pip install unsloth unsloth-zoo