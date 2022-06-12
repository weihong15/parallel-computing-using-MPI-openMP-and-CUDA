#include <iostream>
#include <ostream>
#include <stdexcept>
#include <cstring>

// same idea as typedef
using uint = unsigned int;


template <typename T>
class DenseMatrix
{
public:
    DenseMatrix(uint rows, uint cols)
    {
        this->n_rows = rows;
        this->n_cols = cols;
        this->elements = new T[this->n_rows * this->n_cols];
    }

    DenseMatrix(const DenseMatrix<T>& dm)
    {
        this->n_rows = dm.n_rows;
        this->n_cols = dm.n_cols;
        this->elements = new T[this->n_rows * this->n_cols];
        std::memcpy(this->elements, dm.elements, this->n_rows * this->n_cols * sizeof(T));
    }

    ~DenseMatrix()
    {
        if (this->elements)
        {
            delete[] this->elements;
            this->elements = nullptr;
        }
    }

    DenseMatrix<T> operator +(const DenseMatrix<T>& dm)
    {
        if (this->n_rows != dm.n_rows || this->n_cols != dm.n_cols)
            throw std::invalid_argument("Invalid matrix dimension.");

        DenseMatrix<T> temp(this->n_rows, this->n_cols);
        for (uint i = 0; i < this->n_rows * this->n_cols; i++)
            temp.elements[i] = this->elements[i] + dm.elements[i];
        return temp;
    }

    const T& operator ()(uint row, uint col) const
    {
        if (row >= this->n_rows || col >= this->n_cols)
            throw std::out_of_range("Index out of range");

        return this->elements[row * this->n_cols + col];
    }

    T& operator ()(uint row, uint col)
    {
        if (row >= this->n_rows || col >= this->n_cols)
            throw std::out_of_range("Index out of range");

        return this->elements[row * this->n_cols + col];
    }

    template <typename U>
    friend std::ostream& operator <<(std::ostream& stream, DenseMatrix<U> dm)
    {
        dm.print(stream);
        return stream;
    }

private:
    uint n_rows;
    uint n_cols;
    T *elements;

    void print(std::ostream& stream)
    {
        for (uint i = 0; i < this->n_rows; i++)
        {
            for (uint j = 0; j < this->n_cols; j++)
                stream << (*this)(i, j) << " ";
            stream << std::endl;
        }   
    }
};


int main()
{
    DenseMatrix<int> a(2, 3);
    DenseMatrix<int> b(2, 3);

    // set some numbers
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            a(i, j) = i + j;
            b(i, j) = 1;
        }
    }
         

    // add them
    DenseMatrix<int> c = a + b;
    std::cout << "A: " << std::endl;
    std::cout << a << std::endl;
    std::cout << "B: " << std::endl << b << std::endl;
    std::cout << "C = A + B:" << std::endl << c << std::endl;
    return 0;
}