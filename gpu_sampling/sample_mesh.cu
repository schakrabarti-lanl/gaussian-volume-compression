// sample_mesh_uniform.cpp

#include <iostream>
#include <numeric>
#include <cstring>
#include <cstdint>

#include <viskores/cont/Initialize.h>
#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/cont/DataSetBuilderExplicit.h>
#include <viskores/io/VTKDataSetReader.h>
#include <viskores/filter/resampling/Probe.h>
#include <viskores/rendering/Actor.h>
#include <viskores/rendering/CanvasRayTracer.h>
#include <viskores/rendering/MapperRayTracer.h>
#include <viskores/rendering/Scene.h>
#include <viskores/rendering/View3D.h>
#include <viskores/cont/Timer.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

py::array sample_mesh(
    py::array_t<int64_t> dims_arr,
    py::array_t<float>   origin_arr,
    py::array_t<float>   spacing_arr,
    py::array            val_arr,
    py::array_t<float>   samp_arr)
{
  // 0) Pick CUDA device for timing async kernels
  auto &tracker = viskores::cont::GetRuntimeDeviceTracker();
  tracker.ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  auto cudaTag = viskores::cont::DeviceAdapterTagCuda();
  viskores::cont::Timer timer(cudaTag);

  // 1) Initialization
  timer.Start();
  viskores::cont::Initialize();
  timer.Stop();
  std::cout << "Initialize: " 
            << timer.GetElapsedTime() << " s\n";

  // 2) Read and wrap input arrays
  timer.Reset();
  timer.Start();

  // dims
  auto dims_buf = dims_arr.request();
  auto dims_ptr = static_cast<int64_t*>(dims_buf.ptr);
  viskores::Id nx = dims_ptr[0],
                   ny = dims_ptr[1],
                   nz = dims_ptr[2];

  // origin
  auto orig_buf = origin_arr.request();
  auto o_ptr = static_cast<float*>(orig_buf.ptr);

  // spacing
  auto sp_buf = spacing_arr.request();
  auto s_ptr = static_cast<float*>(sp_buf.ptr);

  // values
  std::size_t totalPts = static_cast<std::size_t>(nx) * ny * nz;
  auto val_c = py::array::ensure(val_arr, py::array::c_style);
  if (!val_c) {
    throw std::runtime_error("Values array must be C-contiguous.");
  }
  if (static_cast<std::size_t>(val_c.size()) != totalPts) {
    throw std::runtime_error("Values array size does not match dims.");
  }
  const auto dtype = val_c.dtype();

  enum class ValueDtype { Float64, Float32, UInt16, UInt8 };
  ValueDtype val_dtype;

  // build uniform DataSet
  viskores::Id3 dims3{nx, ny, nz};
  viskores::Vec3f origin{ o_ptr[0], o_ptr[1], o_ptr[2] };
  viskores::Vec3f spacing{ s_ptr[0], s_ptr[1], s_ptr[2] };
  auto inData = viskores::cont::DataSetBuilderUniform::Create(
    dims3, origin, spacing, "coords"
  );

  if (dtype.is(py::dtype::of<double>())) {
    val_dtype = ValueDtype::Float64;
    const auto *v_ptr = static_cast<const double*>(val_c.data());
    std::vector<viskores::Float64> val_vec(v_ptr, v_ptr + totalPts);
    auto valHandle = viskores::cont::make_ArrayHandleMove(std::move(val_vec));
    inData.AddPointField("value", valHandle);
  } else if (dtype.is(py::dtype::of<float>())) {
    val_dtype = ValueDtype::Float32;
    const auto *v_ptr = static_cast<const float*>(val_c.data());
    std::vector<viskores::Float32> val_vec(v_ptr, v_ptr + totalPts);
    auto valHandle = viskores::cont::make_ArrayHandleMove(std::move(val_vec));
    inData.AddPointField("value", valHandle);
  } else if (dtype.is(py::dtype::of<std::uint16_t>())) {
    val_dtype = ValueDtype::UInt16;
    const auto *v_ptr = static_cast<const std::uint16_t*>(val_c.data());
    std::vector<viskores::UInt16> val_vec(v_ptr, v_ptr + totalPts);
    auto valHandle = viskores::cont::make_ArrayHandleMove(std::move(val_vec));
    inData.AddPointField("value", valHandle);
  } else if (dtype.is(py::dtype::of<std::uint8_t>())) {
    val_dtype = ValueDtype::UInt8;
    const auto *v_ptr = static_cast<const std::uint8_t*>(val_c.data());
    std::vector<viskores::UInt8> val_vec(v_ptr, v_ptr + totalPts);
    auto valHandle = viskores::cont::make_ArrayHandleMove(std::move(val_vec));
    inData.AddPointField("value", valHandle);
  } else {
    throw std::runtime_error(
      "Unsupported values dtype. Expected float64, float32, uint16, or uint8."
    );
  }

  timer.Stop();
  std::cout << "ReadDataSet: " 
            << timer.GetElapsedTime() << " s\n";

  // 3) Build explicit point‐vertex grid for sampling locations
  timer.Reset();
  timer.Start();

  auto samp_buf = samp_arr.request();
  std::size_t n_samples = samp_buf.shape[0];
  auto samp_ptr = static_cast<float*>(samp_buf.ptr);

  std::vector<viskores::Vec<float,3>> sample_coords(n_samples);
  std::memcpy(
    sample_coords.data(),
    samp_ptr,
    n_samples * 3 * sizeof(float)
  );

  std::vector<viskores::Id> sample_conn(n_samples);
  std::iota(sample_conn.begin(), sample_conn.end(), 0);

  auto explicitGrid = viskores::cont::DataSetBuilderExplicit::Create(
    sample_coords,
    viskores::CellShapeTagVertex{},
    static_cast<viskores::IdComponent>(1),
    sample_conn,
    "sample_coords"
  );

  timer.Stop();
  std::cout << "Build explicit grid: " 
            << timer.GetElapsedTime() << " s\n";

  // 4) Probe filter setup
  timer.Reset();
  timer.Start();

  viskores::filter::resampling::Probe probe;
  probe.SetGeometry(explicitGrid);
  probe.SetInvalidValue(-1.0);

  timer.Stop();
  std::cout << "Probe setup: " 
            << timer.GetElapsedTime() << " s\n";

  // 5) Probe execution
  timer.Reset();
  timer.Start();

  viskores::cont::DataSet sampled = probe.Execute(inData);

  timer.Stop();
  std::cout << "Probe execute: " 
            << timer.GetElapsedTime() << " s\n";

  // 6) Retrieve and return result
  timer.Reset();
  timer.Start();

  const auto array = sampled.GetPointField("value").GetData();
  py::array result;

  switch (val_dtype) {
    case ValueDtype::Float64: {
      auto concrete = array.AsArrayHandle<viskores::cont::ArrayHandle<viskores::Float64>>();
      concrete.SyncControlArray();
      auto readPortal = concrete.ReadPortal();
      std::size_t n = readPortal.GetNumberOfValues();
      py::array_t<double> out(n);
      auto out_ptr = out.mutable_data();
      for (std::size_t i = 0; i < n; ++i) {
        out_ptr[i] = readPortal.Get(i);
      }
      result = std::move(out);
      break;
    }
    case ValueDtype::Float32: {
      auto concrete = array.AsArrayHandle<viskores::cont::ArrayHandle<viskores::Float32>>();
      concrete.SyncControlArray();
      auto readPortal = concrete.ReadPortal();
      std::size_t n = readPortal.GetNumberOfValues();
      py::array_t<float> out(n);
      auto out_ptr = out.mutable_data();
      for (std::size_t i = 0; i < n; ++i) {
        out_ptr[i] = readPortal.Get(i);
      }
      result = std::move(out);
      break;
    }
    case ValueDtype::UInt16: {
      auto concrete = array.AsArrayHandle<viskores::cont::ArrayHandle<viskores::UInt16>>();
      concrete.SyncControlArray();
      auto readPortal = concrete.ReadPortal();
      std::size_t n = readPortal.GetNumberOfValues();
      py::array_t<std::uint16_t> out(n);
      auto out_ptr = out.mutable_data();
      for (std::size_t i = 0; i < n; ++i) {
        out_ptr[i] = static_cast<std::uint16_t>(readPortal.Get(i));
      }
      result = std::move(out);
      break;
    }
    case ValueDtype::UInt8: {
      auto concrete = array.AsArrayHandle<viskores::cont::ArrayHandle<viskores::UInt8>>();
      concrete.SyncControlArray();
      auto readPortal = concrete.ReadPortal();
      std::size_t n = readPortal.GetNumberOfValues();
      py::array_t<std::uint8_t> out(n);
      auto out_ptr = out.mutable_data();
      for (std::size_t i = 0; i < n; ++i) {
        out_ptr[i] = static_cast<std::uint8_t>(readPortal.Get(i));
      }
      result = std::move(out);
      break;
    }
  }

  timer.Stop();
  std::cout << "Data retrieval: " 
            << timer.GetElapsedTime() << " s\n";

  return result;
}

py::array_t<double> sample_meshu(
  py::array_t<float> pts_arr,
  py::array_t<int64_t> conn_arr,
  py::array_t<int64_t> cell_types_arr, 
  py::array_t<int64_t> cell_offsets_arr,
  py::array_t<double> val_arr,
  py::array_t<float> samp_arr
)
{
  // Pick the CUDA device for timing async kernels
  auto &tracker = viskores::cont::GetRuntimeDeviceTracker();
  tracker.ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  auto cudaTag = viskores::cont::DeviceAdapterTagCuda();
  viskores::cont::Timer timer(cudaTag);

  // 1) Initialization
  timer.Start();
  viskores::cont::Initialize();
  timer.Stop();
  std::cout << "Initialize: " 
            << timer.GetElapsedTime() << " s\n";

  // 2) Read the VTK dataset
  timer.Reset();
  timer.Start();
  auto pts_buf = pts_arr.request();
  auto conn_buf = conn_arr.request();
  auto cell_types_buf = cell_types_arr.request();
  auto cell_offsets_buf = cell_offsets_arr.request();
  auto val_buf = val_arr.request();

  // Build coordinates
  size_t n_pts = pts_buf.shape[0];
  auto pts_ptr = static_cast<float*>(pts_buf.ptr);
  std::vector<viskores::Vec<float,3>> coords;
  coords.reserve(n_pts);
  for (size_t i = 0; i < n_pts; ++i) {
    coords.emplace_back(
      pts_ptr[3*i + 0],
      pts_ptr[3*i + 1],
      pts_ptr[3*i + 2]
    );
  }

  // Build connectivity
  auto conn_ptr = static_cast<int64_t*>(conn_buf.ptr);
  std::vector<viskores::Id> conn_vec(conn_ptr, conn_ptr + conn_buf.shape[0]);

  // Build cell shapes and offsets
  auto cell_types_ptr = static_cast<int64_t*>(cell_types_buf.ptr);
  auto cell_offsets_ptr = static_cast<int64_t*>(cell_offsets_buf.ptr);
  size_t n_cells = cell_types_buf.shape[0];
  
  std::vector<viskores::UInt8> cell_shapes;
  std::vector<viskores::IdComponent> num_indices;
  cell_shapes.reserve(n_cells);
  num_indices.reserve(n_cells);

  // Map VTK cell types to VTK-m cell shapes
  for (size_t i = 0; i < n_cells; ++i) {
    int64_t vtk_cell_type = cell_types_ptr[i];
    int64_t start_offset = cell_offsets_ptr[i];
    int64_t end_offset = (i + 1 < n_cells) ? cell_offsets_ptr[i + 1] : conn_buf.shape[0];
    int64_t num_pts_in_cell = end_offset - start_offset;

    switch (vtk_cell_type) {
      case 1: // VTK_VERTEX
        cell_shapes.push_back(static_cast<viskores::UInt8>(viskores::CellShapeTagVertex::Id));
        break;
      case 3: // VTK_LINE
        cell_shapes.push_back(static_cast<viskores::UInt8>(viskores::CellShapeTagLine::Id));
        break;
      case 5: // VTK_TRIANGLE
        cell_shapes.push_back(static_cast<viskores::UInt8>(viskores::CellShapeTagTriangle::Id));
        break;
      case 9: // VTK_QUAD
        cell_shapes.push_back(static_cast<viskores::UInt8>(viskores::CellShapeTagQuad::Id));
        break;
      case 10: // VTK_TETRA
        cell_shapes.push_back(static_cast<viskores::UInt8>(viskores::CellShapeTagTetra::Id));
        break;
      case 12: // VTK_HEXAHEDRON
        cell_shapes.push_back(static_cast<viskores::UInt8>(viskores::CellShapeTagHexahedron::Id));
        break;
      case 13: // VTK_WEDGE
        cell_shapes.push_back(static_cast<viskores::UInt8>(viskores::CellShapeTagWedge::Id));
        break;
      case 14: // VTK_PYRAMID
        cell_shapes.push_back(static_cast<viskores::UInt8>(viskores::CellShapeTagPyramid::Id));
        break;
      default:
        throw std::runtime_error("Unsupported VTK cell type: " + std::to_string(vtk_cell_type));
    }
    num_indices.push_back(static_cast<viskores::IdComponent>(num_pts_in_cell));
  }

  // Build field data
  auto val_ptr = static_cast<double*>(val_buf.ptr);
  std::vector<viskores::Float64> val_vec(val_ptr, val_ptr + val_buf.shape[0]);
  viskores::cont::ArrayHandle<viskores::Float64> valHandle = viskores::cont::make_ArrayHandleMove(std::move(val_vec));

  // Create dataset with mixed cell types
  auto inData = viskores::cont::DataSetBuilderExplicit::Create(
    coords, cell_shapes, num_indices, conn_vec, "coords");
  
  inData.AddPointField("value", valHandle);
  timer.Stop();
  std::cout << "ReadDataSet: " 
            << timer.GetElapsedTime() << " s\n";

  // 3) Build an explicit point-vertex grid
  timer.Reset();
  timer.Start();
  auto samp_buf = samp_arr.request();
  size_t n_samples = samp_buf.shape[0];
  auto samp_ptr = static_cast<float*>(samp_buf.ptr);
  std::vector<viskores::Vec<float,3>> sample_coords(n_samples);
  std::memcpy(
    sample_coords.data(),
    samp_ptr,
    n_samples * 3 * sizeof(float)
  );
  std::vector<viskores::Id> sample_conn(n_samples);
  std::iota(sample_conn.begin(), sample_conn.end(), 0);
  auto explicitGrid = viskores::cont::DataSetBuilderExplicit::Create(
    sample_coords,
    viskores::CellShapeTagVertex{},
    static_cast<viskores::IdComponent>(1),
    sample_conn,
    "sample_coords"
  );
  timer.Stop();
  std::cout << "Build explicit grid: " 
            << timer.GetElapsedTime() << " s\n";

  // 4) Probe filter setup
  timer.Reset();
  timer.Start();
  viskores::filter::resampling::Probe probe;
  probe.SetGeometry(explicitGrid);
  probe.SetInvalidValue(-1.0);
  timer.Stop();
  std::cout << "Probe setup: " 
            << timer.GetElapsedTime() << " s\n";

  // 5) Probe filter execution
  timer.Reset();
  timer.Start();
  viskores::cont::DataSet sampled = probe.Execute(inData);
  timer.Stop();
  std::cout << "Probe execute: " 
            << timer.GetElapsedTime() << " s\n";

  // 6) Retrieve and synchronize field data
  timer.Reset();
  timer.Start();
  const auto array = sampled.GetPointField("value").GetData();
  auto concrete = array.AsArrayHandle<viskores::cont::ArrayHandle<viskores::Float64>>();
  concrete.SyncControlArray();
  auto readPortal = concrete.ReadPortal();
  std::size_t n = readPortal.GetNumberOfValues();
  py::array_t<double> result(n);
  auto buf = result.mutable_data();
  for (std::size_t i = 0; i < n; ++i)
  {
    buf[i] = readPortal.Get(i);
  }
  timer.Stop();
  std::cout << "Data retrieval: " 
            << timer.GetElapsedTime() << " s\n";

  return result;
}
