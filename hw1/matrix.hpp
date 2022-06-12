#ifndef _MATRIX_HPP
#define _MATRIX_HPP

#include <ostream>
#include <vector>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
/*
This is the pure abstract base class specifying general set of functions for a
square matrix.

Concrete classes for specific types of matrices, like MatrixSymmetric, should
implement these functions.
*/
template <typename T>
class Matrix {
  // Returns reference to matrix element (i, j).
  virtual T& operator()(int i, int j) = 0;
  // Number of non-zero elements in matrix.
  virtual unsigned NormL0() const = 0;
  // Enables printing all matrix elements using the overloaded << operator
  virtual void Print(std::ostream& ostream) = 0;

  template <typename U>
  friend std::ostream& operator<<(std::ostream& stream, Matrix<U>& m);
};

/* TODO: Overload the insertion operator by modifying the ostream object */
template <typename T>
std::ostream& operator<<(std::ostream& stream, Matrix<T>& m) {
  m.Print(stream);
  return stream;
}

/* MatrixSymmetric Class is a subclass of the Matrix class */
template <typename T>
class MatrixSymmetric : public Matrix<T> {
 private:
  // Matrix Dimension. Equals the number of columns and the number of rows.
  unsigned int n_;
  // Elements of the matrix. You get to choose how to organize the matrix
  // elements into this vector.
  std::vector<T> data_;

 public:
  // TODO: Default constructor //not done, what to do here?
  MatrixSymmetric() {
    n_ = 0;
  }

  // TODO: Constructor that takes matrix dimension as argument
  MatrixSymmetric(const int n) {
    if (n<0)
    throw std::invalid_argument("wrong dim, shoukd be at least 1");
    n_ = n;
    data_.reserve(n*(n+1)/2);
    std::fill(data_.begin(), data_.end(), static_cast<T>(0));
    }

  // TODO: Function that returns the matrix dimension
  unsigned int size() const { return n_; }

  // TODO: Function that returns reference to matrix element (i, j).
  T& operator()(int i, int j) override { 
    if (i<0 || j<0 || i>= static_cast<int>(n_) || j>= static_cast<int>(n_))
    throw std::invalid_argument("invalid referencing arguement");

    if (i<j){
      int temp = j;
      j = i;
      i = temp;
    }
    // from now on, j<= i, i.e. lower left matrix
    // first row 1 elements, second row 2 elements, third row n elements
    // (5,5) in a 10X 10 matrix, will be 1+2+3+4+5 //sum first 4 rows, and add current col
    // (5,6) will become (6,5) will sum first 5 rows and add 5

    //correction, we assume 1 indexing but in fact it's 0 indexing
    //(5,5) will have 1+2+3+4+5+6, 
    // (5,6) will become (6,5) will sum first 6 rows, (1+...+6), 6th column add 6
    // (1+i)*i/2 +j+1

    int ele = (1+i)*i/2 +j+1;
    return data_[ele]; }

  const T& operator()(int i, int j) const { 
    if (i<0 || j<0 || i>= static_cast<int>(n_) || j>= static_cast<int>(n_))
    throw std::invalid_argument("invalid referencing arguement");
    if (i<j){
      int temp = j;
      j = i;
      i = temp;
    }
    int ele = (1+i)*i/2 +j+1;
    return data_[ele]; }

  // TODO: Function that returns number of non-zero elements in matrix.
  unsigned NormL0() const override {
    unsigned sum = 0;
    for (unsigned int i = 0; i < n_; i++)
      {
          for (unsigned int j = 0; j < this->n_; j++)
          {
            //const T temp = static_cast<const T> ((*this)(i,j));
            T temp = (*this)(i,j);
            if ( temp != 0){
              //std::cout << "count (i,j) : " << i << j <<std::endl;
              sum += 1;
            }
          }
      }   
    return sum;
      }

  // TODO: Function that modifies the ostream object so that
  // the "<<" operator can print the matrix (one row on each line).
  void Print(std::ostream& stream) override {
      for (uint i = 0; i < n_; i++)
      {
          for (uint j = 0; j < n_; j++)
          {
              // TODO: print out element (i, j). 
              stream << "    " << (*this)(i,j) << " ";
          }
          stream << std::endl;
      } 
  }
};

#endif /* MATRIX_HPP */