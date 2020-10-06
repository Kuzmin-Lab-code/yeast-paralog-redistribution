TEST_PATH=./tests
MODULES_PATH=./modules
RESULTS_PATH=./results
RUN_PATH=./run.py

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