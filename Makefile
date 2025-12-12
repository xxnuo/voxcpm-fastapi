.PHONY: build compile push sync-clean sync-from-gpu sync-to-gpu test

VERSION := $(shell git rev-parse --short HEAD)
UV := ~/.local/bin/uv
CURL := $(shell if command -v axel >/dev/null 2>&1; then echo "axel"; else echo "curl"; fi)
REMOTE := nvidia@gpu
REMOTE_PATH := ~/projects/work/voxcpm-fastapi
DOCKER_NAME := voxcpm-fastapi

default:
	@echo "My custom develop script for voxcpm-fastapi, not for common use!"
	@echo "Checking dependencies..."
	@if ! which $(UV) > /dev/null 2>&1; then echo "$(UV) not found, please install it: curl -LsSf https://astral.sh/uv/install.sh | bash"; exit 1; fi
	@if ! which hf > /dev/null 2>&1; then echo "hf not found, please install it: uv tool install huggingface-hub"; exit 1; fi
	@if ! which modelscope > /dev/null 2>&1; then echo "modelscope not found, please install it: uv tool install modelscope"; exit 1; fi
	@if ! which rsync > /dev/null 2>&1; then echo "rsync not found, please install it first"; exit 1; fi
	@if ! which ssh > /dev/null 2>&1; then echo "ssh not found, please install it first"; exit 1; fi
	@echo "All dependencies are installed"

sync-from-gpu:
	rsync -arvzlt --delete --exclude-from=.rsyncignore $(REMOTE):$(REMOTE_PATH)/ ./

sync-to-gpu:
	ssh -t $(REMOTE) "mkdir -p $(REMOTE_PATH)"
	rsync -arvzlt --delete --exclude-from=.rsyncignore ./ $(REMOTE):$(REMOTE_PATH)

sync-clean:
	ssh -t $(REMOTE) "rm -rf $(REMOTE_PATH)"

prepare-git:
	git submodule update --init --recursive
	$(UV) sync --all-groups
	mkdir -p models

prepare-model:
	mkdir -p models
	hf download openbmb/VoxCPM1.5 --local-dir ./models/openbmb/VoxCPM1.5
	modelscope download --model iic/speech_zipenhancer_ans_multiloss_16k_base --local_dir ./models/iic/speech_zipenhancer_ans_multiloss_16k_base
	modelscope download --model iic/SenseVoiceSmall --local_dir ./models/iic/SenseVoiceSmall

prepare-pypi:
	$(UV) pip compile \
	--all-extras \
	--index-url http://wa.lan:10608/simple --trusted-host wa.lan \
	--no-deps --output-file requirements-pypi.txt ./voxcpm/pyproject.toml

compile: sync-to-gpu
	ssh -t $(REMOTE) "cd $(REMOTE_PATH) && \
		$(MAKE) prepare-pypi"
	$(MAKE) sync-from-gpu

build: compile
	ssh -t $(REMOTE) "cd $(REMOTE_PATH) && \
		docker build \
		 --shm-size=30g \
		-f Dockerfile \
		--progress=plain \
		-t $(DOCKER_NAME):$(VERSION)-dev \
		--network host \
		."
	@echo $(DOCKER_NAME):$(VERSION)-dev >> $(DOCKER_NAME).dev.version

test: build
	ssh -t $(REMOTE) "cd $(REMOTE_PATH) && \
		docker run --shm-size=30g -it --rm --gpus all \
		--name $(DOCKER_NAME) --network host \
		-v ./output:/app/output \
		-v /root/.cache/torch_extensions/:/root/.cache/torch_extensions/ \
		$(DOCKER_NAME):$(VERSION)-dev"

inspect: build
	ssh -t $(REMOTE) "cd $(REMOTE_PATH) && \
		docker run --shm-size=30g -it --rm --gpus all \
		--name $(DOCKER_NAME) --network host \
		-v ./output:/app/output \
		-v /root/.cache/torch_extensions/:/root/.cache/torch_extensions/ \
		$(DOCKER_NAME):$(VERSION)-dev bash"

push: compile
	ssh -t $(REMOTE) "cd $(REMOTE_PATH) && \
		docker build \
		 --shm-size=30g \
		-f Dockerfile \
		--progress=plain \
		-t $(DOCKER_NAME):$(VERSION) \
		-t $(DOCKER_NAME):latest \
		--network host \
		. && \
		docker push $(DOCKER_NAME):$(VERSION)"
	@echo $(DOCKER_NAME):$(VERSION) >> $(DOCKER_NAME).version
