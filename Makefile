init:
	pip3 install -r requirements.txt

checks:
	python3 -m unittest discover

hello:
	echo "Hello world"