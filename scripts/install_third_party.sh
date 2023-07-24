set -x 

# Install Frankmocap
rm -r externals/frankmocap
mkdir -p externals
# my modification on relative path
git clone https://github.com/judyye/frankmocap.git externals/frankmocap
cd externals/frankmocap
export CC=gcc-9
export CXX=g++-9
bash scripts/install_frankmocap.sh
cd ../..

# install manopth
pip install "git+https://github.com/hassony2/manopth.git"


# install detectron2
# clone the repo in order to access pre-defined configs in PointRend project
cd externals
git clone --branch v0.6 https://github.com/facebookresearch/detectron2.git detectron2
# install detectron2 from source
pip install -e detectron2
cd ../
# See https://detectron2.readthedocs.io/tutorials/install.html for other installation options


# install neural_renderer for HOMAN
mkdir -p external
git clone https://github.com/hassony2/multiperson.git externals/multiperson
pip install externals/multiperson/neural_renderer

# install sdf
pip install git+https://github.com/zhifanzhu/sdf_pytorch


# install Roma rotation library
git clone https://github.com/naver/roma externals/roma
cd externals/roma && git checkout 22806dfb43329b9bf1dd2cead7e96720330e3151
pip install externals/roma


# Install torch_scatter (we use scatter_min)
# E.g. for python3.8 torch1.8 cuda 10.2:
# pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu102/torch_scatter-2.0.8-cp38-cp38-linux_x86_64.whl