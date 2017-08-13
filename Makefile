init:
	pip install -r requirements.txt
upgrade:
	pip install --upgrade -r requirements-to-freeze.txt && pip freeze > requirements.txt
