from skbuild import setup
import os

setup(
    name="GPUMeshSampling",
    description="Example Python package exposing CUDA/C++ via pybind11",
    packages=["gpu_mesh_sampling"],
    cmake_source_dir=".",               # where your CMakeLists.txt lives
    cmake_args=[
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_ARCHITECTURES=70;75;80"
    ],
    install_requires=["numpy", "pybind11"],
)
