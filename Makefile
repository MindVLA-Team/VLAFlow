.PHONY: help check autoformat
help:
	@echo "make check       Check Python formatting and lint"
	@echo "make autoformat  Apply Python formatting and fixable lint"
check:
	black --check vlaflow deployment examples scripts
	ruff check vlaflow deployment examples scripts
autoformat:
	black vlaflow deployment examples scripts
	ruff check --fix-only vlaflow deployment examples scripts
