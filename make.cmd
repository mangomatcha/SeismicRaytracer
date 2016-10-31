@echo off
nvcc ray.cpp ray.cu -o ray
del ray.lib
del ray.exp