.PHONY: clean
clean: ## Remove generated files
	pip uninstall diff_gaussian_rasterization -y || true
	rm -rf build || true
	rm -rf dist || true
	rm -rf diff_gaussian_rasterization/__pycache__/ || true
	rm diff_gaussian_rasterization/*gnu.so || true

.PHONY: install
install: ## Build and install the package
	python setup.py install
