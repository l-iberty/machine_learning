# install theano
- ubuntu 14.04
- python 2.7

## 卸载apt-get安装的numpy,scipy(如果有的话)
```sh
$ sudo apt-get remove python-numpy
$ sudo apt-get remove python-scipy
```

## 安装各种库
```sh
$ sudo apt-get install gfortran
$ sudo apt-get install libopenblas-dev
$ sudo apt-get install liblapack-dev
$ sudo apt-get install libatlas-base-dev
$ sudo apt-get install python-dev python-nose python-pip
```

## 使用pip安装numpy和scipy
```sh
$ pip install numpy==1.10.4
# 安装完成后进行测试:
# $ python
# >>> import numpy
# >>> numpy.test()
$ pip install scipy==0.17.0
# 安装完成后进行测试:
# (python) >>> import scipy
# (python) >>> scipy.test()
```

## 编译安装Theano
pip安装Theano需要>=3.5的python，所以需要用源码编译安装
```sh
$ git clone https://github.com/Theano/Theano.git
$ cd Theano
$ python setup.py build
$ python setup.py install
$ pip install parameterized # 用于Theano的测试
# 测试Theano
# (python) >>> import theano
# (python) >>> theano.test()
```
