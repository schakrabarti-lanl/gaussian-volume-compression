cmake_minimum_required(VERSION 3.13)

project(GPUMeshSampling LANGUAGES CXX CUDA)
set(Viskores_DIR  $ENV{VISKORES_DIR})
set(pybind11_DIR  $ENV{PYBIND11_DIR})

if(NOT Viskores_DIR)
  message(FATAL_ERROR "Please set VISKORES_DIR in your environment")
endif()
if(NOT pybind11_DIR)
  message(FATAL_ERROR "Please set PYBIND11_DIR in your environment")
endif()

find_package(Viskores REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(_gpu_mesh_sampling
    ext.cpp
    sample_mesh.cu
    sample_mesh.h
)

set_target_properties(_gpu_mesh_sampling PROPERTIES CUDA_ARCHITECTURES "70;75;86")

target_link_libraries(_gpu_mesh_sampling PRIVATE viskores::filter viskores::rendering)

install(
  TARGETS _gpu_mesh_sampling
  LIBRARY DESTINATION gpu_mesh_sampling
  RUNTIME DESTINATION gpu_mesh_sampling
  ARCHIVE DESTINATION gpu_mesh_sampling
)