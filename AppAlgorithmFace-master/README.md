GPU 环境安装：
- https://linuxsoft.cern.ch/cern/centos/7/updates/x86_64/repoview/kernel-headers.html
- yum install -y kernel-devel-3.10.0-1062.4.1.el7.x86_64.rpm kernel-headers-3.10.0-1062.4.1.el7.x86_64.rpm
- sh cuda_10.1.243_418.87.00_linux.run

服务器环境安装：
- yum install -y gcc-c++ cmake libSM.x86_64 libXrender.x86_64 libXext.x86_64

Python3 及依赖安装：
- 切换到 root 用户
- 解压到当前目录，tar zxvf python3.tgz
- 更改用户，chown -R root:root python3
- 放到指定目录，mv python3 /usr/local/python3
- 创建虚拟环境，/usr/local/python3/bin/python3 -m venv /usr/local/{AppID}
- 升级 pip，/usr/local/{AppID}/bin/pip install --upgrade pip setuptools -i https://mirrors.aliyun.com/pypi/simple
- 安装 pip 依赖，/usr/local/{AppID}/bin/pip install -r requirements.txt --default-timeout=10 -i https://mirrors.aliyun.com/pypi/simple
- 安装本地依赖，cd lib/jdconfig && /usr/local/{AppID}/bin/python setup.py install && rm -rf build

拷贝模型文件：
- 按 conf/config.toml 中的配置，将配置文件拷贝到指定目录
