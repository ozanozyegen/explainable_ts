# Welcome to the MakeFile
name := exp_ts

# Setup the folder hierarchy
setup:
	@mkdir data
	@mkdir data/external
	@mkdir data/processed
	@mkdir data/interim
	@mkdir data/raw
	@mkdir models
	@mkdir notebooks
	@mkdir references
	@mkdir reports
	@mkdir reports/figures
	@mkdir src/data
	@mkdir src/features
	@mkdir src/models
	@mkdir src/visualization
	@echo "PYTHONPATH=./src" > .env

# Extra
# Clean __pycache___ for unix
clean_unix:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

# Clean __pycache___ for windows
clean_win:
	@python -c "for p in __import__('pathlib').Path('.').rglob('*.py[co]'): p.unlink()"
	@python -c "for p in __import__('pathlib').Path('.').rglob('__pycache__'): p.rmdir()"

# Anaconda export environment
export:
	@conda env export > $(name).yml

# Anaconda load environment
load:
	@conda env create -f $(name).yml

# Run Tensorboard
tensorboard:
	@tensorboard --logdir outputs/

