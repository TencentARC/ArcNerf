#!/bin/bash

cd ./include/tiny-cuda-nn
cmake . -B build
cmake --build build --config RelWithDebInfo -j 12
cd bindings/torch
python setup.py install
cd ../../../..
