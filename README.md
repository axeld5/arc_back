# arc_back
ARC Repo

To send .env to VM: scp -P XXXX .env user@YYYY:/path/to/destination/

To create .env in VM: echo "HF_TOKEN=your_token_here" > .env
To create distant kernel uv run --active python -m ipykernel install --user --name project --display-name "Python (project)"