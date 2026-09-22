#pragma once

#include <cstddef>
#include <algorithm>
#include <vector>

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
  Grid(std::size_t rows, std::size_t cols){
    rows_ = rows;
    cols_ = cols;
    stride_ = (calculate_stride(cols));
    data_matrix = std::vector<double> (rows * stride_, 0);
  }

  // I chose 1D vector as it ensures data is contiguous for later optimization.
  double& operator()(std::size_t i, std::size_t j){
    return data_matrix[i * stride_ + j];
  }
  double operator()(std::size_t i, std::size_t j) const{
    return data_matrix[i * stride_ + j];
  }

  std::size_t obtain_columns() const{
    return cols_;
  }
  std::size_t obtain_rows() const{
    return rows_;
  }

  std::size_t obtain_stride() const{
    return stride_;
  }

  // Implemented a pointer access to combat alias flags that prevent vectorization
  double* obtain_data() {
    return data_matrix.data();
  }
  const double* obtain_data() const {
    return data_matrix.data();
  }
};

void apply_stencil(const Grid& old_grid, Grid& new_grid){
  std::size_t columns = old_grid.obtain_columns();
  std::size_t rows = old_grid.obtain_rows();
  std::size_t stride = old_grid.obtain_stride();

  #pragma omp parallel for schedule(static) 
  for (std::size_t i = 1; i < rows - 1; i++){
    const double* __restrict row_mid = old_grid.obtain_data() + i * stride;
    const double* __restrict row_up = old_grid.obtain_data() + (i - 1) * stride;
    const double* __restrict row_down = old_grid.obtain_data() + (i + 1) * stride;
    double* __restrict out = new_grid.obtain_data() + i * stride;
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

  std::copy(old_grid.obtain_data(), old_grid.obtain_data() + columns, new_grid.obtain_data());
  std::copy(old_grid.obtain_data() + (rows-1) * stride, 
            old_grid.obtain_data() + (rows-1) * stride + columns, 
            new_grid.obtain_data() + (rows-1) * stride);

};