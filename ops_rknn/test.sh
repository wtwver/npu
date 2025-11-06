#!/bin/bash
cd /home/orangepi/tinygrad/npu/ops_rknn
g++ -o conv2d_simple conv2d_simple.cpp -I../include -lrknnrt -std=c++11
./conv2d_simple
