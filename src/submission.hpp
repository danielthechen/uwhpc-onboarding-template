#pragma once

#include <cstddef>
#include <algorithm>
#include <vector>
#include <stdexcept>

// Alignment target of 64 bytes based on the evaluator's -march=x86-64-v3
static constexpr std::size_t alignment = 64;

class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::size_t stride_;
  std::vector<double> data_matrix;

  static std::size_t calculate_stride(std::size_t columns){
    constexpr std::size_t grids_per_row = alignment / sizeof(double);
    std::size_t remainder = columns % grids_per_row;
    if (remainder == 0){
      return columns;
    }
    return columns + (grids_per_row - remainder);
  }

public:
  //Switched to member initialization to prevent double allocation
  Grid(std::size_t rows, std::size_t cols): 
    rows_(rows),
    cols_(cols),
    stride_(calculate_stride(cols)),
    data_matrix(rows * stride_, 0.0)
    {}
  
  // I chose 1D vector as it ensures data is contiguous for later optimization.
  double& operator()(std::size_t i, std::size_t j){
    if ((i >= rows_) or (j >= cols_)) throw std::out_of_range("Grid operator (,) index out of range");
    return data_matrix[i * stride_ + j];
  }
  double operator()(std::size_t i, std::size_t j) const{
    if ((i >= rows_) or (j >= cols_)) throw std::out_of_range("Grid operator (,) index out of range");
    return data_matrix[i * stride_ + j];
  }

  // Implemented pointer access to combat alias flags that prevent vectorization
  struct View {
    double* data;
    std::size_t rows, cols, stride;
  };

  struct ConstView {
    const double* data;
    std::size_t rows, cols, stride;
  };

  View view() {
    return View{data_matrix.data(), rows_, cols_, stride_};
  }

  ConstView view() const {
    return ConstView{data_matrix.data(), rows_, cols_, stride_};
  }

};

void apply_stencil(const Grid& old_grid, Grid& new_grid){
  Grid::ConstView old_view = old_grid.view();
  Grid::View new_view = new_grid.view();

  if ((old_view.rows != new_view.rows) or (old_view.cols != new_view.cols)){
    throw std::invalid_argument("old_grid and new_grid dimensions differ :(");
  }

  std::size_t columns = old_view.cols;
  std::size_t rows = old_view.rows;
  std::size_t stride = old_view.stride;

  if ((rows == 0) or (columns == 0)){
    return;
  }

  #pragma omp parallel for schedule(static) 
  for (std::size_t i = 1; i < rows - 1; i++){
    const double* __restrict row_mid = old_view.data + i * stride;
    const double* __restrict row_up = old_view.data + (i - 1) * stride;
    const double* __restrict row_down = old_view.data + (i + 1) * stride;
    double* __restrict out = new_view.data + i * stride;
    //__restrict on each row for easier compiler vectorization

    //Take the chance to copy the boundary columns as well within the threaded loop
    out[0] = row_mid[0];
    out[columns - 1] = row_mid[columns - 1];

    #pragma omp simd
    for (std::size_t j = 1; j < columns - 1; j++){
      out[j] = 0.5 * row_mid[j] +
             0.125 * (row_down[j] + row_up[j] +
                     row_mid[j-1] + row_mid[j+1]);
    }
  }

  std::copy(old_view.data, old_view.data + columns, new_view.data);
  std::copy(old_view.data + (rows-1) * stride, 
            old_view.data + (rows-1) * stride + columns, 
            new_view.data + (rows-1) * stride);

};