.PHONY: run test lint fmt

run:
	streamlit run app/streamlit_app.py

test:
	pytest -q

fmt:
	black src tests app
	isort src tests app

lint:
	ruff check src tests app
