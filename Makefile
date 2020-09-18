TEST_PATH=./tests
MODULES_PATH=./modules
RESULTS_PATH=./results

.PHONY: format
format:
	isort $(MODULES_PATH)
	black $(MODULES_PATH)

.PHONY: lint
lint:
	isort -c $(MODULES_PATH)
	black --check $(MODULES_PATH)
	mypy $(MODULES_PATH)

.PHONY: test
test:
	python -m unittest discover -s $(TEST_PATH) -t $(TEST_PATH)