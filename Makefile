init:
	pip3 install -r requirements.txt

checks:
	python3 -m unittest discover

remove-old:
	sudo rm -rf db_data

hello:
	echo "Hello world"