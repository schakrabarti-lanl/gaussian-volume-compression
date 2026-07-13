#include <iostream>
#include <viskores/cont/Initialize.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

py::array sample_mesh(
    py::array_t<int64_t> dims_arr,
    py::array_t<float>   origin_arr,
    py::array_t<float>   spacing_arr,
    py::array            val_arr,
    py::array_t<float>   samp_arr
);

py::array_t<double> sample_meshu(
    py::array_t<float> pts_arr,
    py::array_t<int64_t> conn_arr,
    py::array_t<int64_t> cell_types_arr,
    py::array_t<int64_t> cell_offsets_arr,
    py::array_t<double> val_arr,
    py::array_t<float> samp_arr
);