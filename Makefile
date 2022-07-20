TEST_PATH=./tests
MODULES_PATH=./modules
RESULTS_PATH=./results
RUN_PATH=./train_segmentation.py \
		 ./train_classification.py ./make_segmentation.py ./make_frames.py ./make_features.py \
		  ./postprocess_segmentation.py ./localization.py ./abundance.py ./normalize_abundance.py

.PHONY: format
format:
	isort $(MODULES_PATH) $(RUN_PATH)
	black $(MODULES_PATH) $(RUN_PATH)

.PHONY: lint
lint:
	isort -c $(MODULES_PATH) $(RUN_PATH)
	black --check $(MODULES_PATH) $(RUN_PATH)
	mypy $(MODULES_PATH) $(RUN_PATH)

.PHONY: test
test:
	python -m unittest discover -s $(TEST_PATH) -t $(TEST_PATH)