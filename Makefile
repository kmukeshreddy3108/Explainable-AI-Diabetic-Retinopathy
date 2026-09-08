.PHONY: test run install clean

install:
	pip install -r requirements-dev.txt

test:
	python -m pytest tests/ -v --tb=short

run:
	streamlit run streamlit_app/app.py --server.port 8501

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
