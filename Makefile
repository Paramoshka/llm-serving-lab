IMAGE ?= mini-llm:local
DEVICE ?= cuda
EPOCHS ?= 10
MAX_CHARS ?= 200000
BATCH_SIZE ?= 128

DATA_FILE := data/dailydialog.txt

.PHONY: help data learn build test

help:
	@printf '%s\n' 'make data   - download DailyDialog to data/dailydialog.txt'
	@printf '%s\n' 'make learn  - train the model (DEVICE=cuda by default)'
	@printf '%s\n' 'make build  - train and build the CPU serving image'
	@printf '%s\n' 'make test   - run the test suite'

data: $(DATA_FILE)

$(DATA_FILE):
	python -m src.download_data --output $@

learn: data
	python -m src.train --data $(DATA_FILE) --device $(DEVICE) --epochs $(EPOCHS) --max-chars $(MAX_CHARS) --batch-size $(BATCH_SIZE)

build: learn
	docker build -t $(IMAGE) .

test:
	pytest -q
