.PHONY: clean
clean: ## Remove generated files
	rm -rf build || true
	rm -rf dist || true
	rm -rf diff_gaussian_rasterization/__pycache__/ || true
	rm diff_gaussian_rasterization/*gnu.so || true
