#pragma once

#include <cstddef>
#include <vector>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<double> data_matrix;

public:
  Grid(std::size_t rows, std::size_t cols){
    rows_ = rows;
    cols_ = cols;
    data_matrix = std::vector<double> (rows*cols, 0);
  }

  double& operator()(std::size_t i, std::size_t j){
    return data_matrix[i * cols_ + j];
  }

  double operator()(std::size_t i, std::size_t j) const{
    return data_matrix[i * cols_ + j];
  }

  std::size_t obtain_columns() const{
    return cols_;
  }

  std::size_t obtain_rows() const{
    return rows_;
  }

  double* obtain_data() {
    return data_matrix.data();
  }

  const double* obtain_data() const {
    return data_matrix.data();
  }
};

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid& old_grid, Grid& new_grid){
  std::size_t columns = old_grid.obtain_columns(); //j
  std::size_t rows = old_grid.obtain_rows(); //i

  #pragma omp parallel for schedule(static)
  for (std::size_t i = 1; i < rows - 1; i++){
    const double* __restrict row_mid = old_grid.obtain_data() + i * columns;
    const double* __restrict row_up = old_grid.obtain_data() + (i - 1) * columns;
    const double* __restrict row_down = old_grid.obtain_data() + (i + 1) * columns;
    double* __restrict out = new_grid.obtain_data() + i * columns;

    #pragma omp simd
    for (std::size_t j = 1; j < columns - 1; j++){
      out[j] = 0.5 * row_mid[j] +
                   0.125 * (row_down[j] + row_up[j] +
                            row_mid[j-1] + row_mid[j+1]);
    }
  }

  //copy old rows to new rows
  std::copy(old_grid.obtain_data(), old_grid.obtain_data() + columns, new_grid.obtain_data());
  std::copy(old_grid.obtain_data() + (rows-1) * columns, old_grid.obtain_data() + rows * columns, new_grid.obtain_data() + (rows-1) * columns);

  for (std::size_t i = 1; i < rows-1; i++){
    new_grid.obtain_data()[i * columns] = old_grid.obtain_data()[i * columns]; //copy over first column 
    new_grid.obtain_data()[i * columns + columns - 1] = old_grid.obtain_data()[i * columns + columns - 1]; //copy over last column
  }
};