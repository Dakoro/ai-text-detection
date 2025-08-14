clean:
	rm -f data/checkpoint/*.*

env:
	uv sync

build-img:
	docker build --no-cache -t ai-text-detection:latest .

run-container:
	docker run --gpus all -p 8000:8000 ai-text-detection:latest

run-server:
	uv run src/server.py