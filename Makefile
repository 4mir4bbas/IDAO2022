all: build

build:
	@echo 'starting....'
run:
	unzip -q data.zip
	bash train.sh
