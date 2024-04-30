PROJ_NAME=AppAlgorithmFace
HOME=$(shell pwd)
TARGET=target
TARGET_PROJ=${TARGET}/${PROJ_NAME}

build:
	${MAKE} clean
	mkdir -p ${TARGET_PROJ}/venv
	cp -r src scripts ${TARGET_PROJ}
	cd ${TARGET} && zip -r ${PROJ_NAME}.zip ${PROJ_NAME}
build-local:
	./venv/bin/pip3 install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt --default-timeout=10
	cd lib/jdconfig && ../../venv/bin/python setup.py install && rm -rf build
clean:
	rm -rf ${TARGET};

.PHONY: build build-local clean
