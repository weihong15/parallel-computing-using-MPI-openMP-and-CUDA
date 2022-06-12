#include <iostream>
#include <ostream>
#include <stdexcept>

// same idea as typedef
using uint = unsigned int;


// TODO: template this class with a type T.
// What do we need to change in the method signature or members?
template <typename T>
class DenseMatrix
{
public:
    DenseMatrix(uint rows, uint cols)
    {
        this->n_rows = rows;
        this->n_cols = cols;
        this->elements = new int[this->n_rows * this->n_cols];
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

    DenseMatrix operator +(const DenseMatrix& dm)
    {
        // TODO: check if n_rows and n_cols match dm.n_rows and dm.n_cols. 
        // if not same, throw std::invalid_argument
        if(n_rows != dm.n_rows || n_cols != dm.n_cols){
            throw std::invalid_argument("wrong d");
        }
        // TODO: create a new DenseMatrix<T> with same dimensions
        // implement the addition with a loop.
        DenseMatrix<T> temp(this->n_rows, this->n_cols);
        for (int i = 0; i<n_rows*n_cols; i++)
            temp.elements[i] = this->elements[i] + dm.elements[i];
        return temp;
    }

    const T& operator ()(uint row, uint col) const
    {
        // TODO: check if row < n_rows and cols < n_cols
        // if not throw std::out_of_range
        if(n_rows < row || n_cols < col){
            throw std::invalid_argument("wrong d..");
        }
        // TODO: return const reference to element
        // hint: you do not have to do anything special

        return this->elements[row*n_cols+col];
    }

    int& operator ()(uint row, uint col)
    {
        // TODO: check if row < n_rows and cols < n_cols
        // if not throw std::out_of_range
        if(n_rows < row || n_cols < col){
            throw std::invalid_argument("wrong d...........");
        }

        // TODO: return mutable reference to element
        // hint: you do not have to do anything special
        return this->elements[row*n_cols+col];
    }

    /**
     * TODO: Implement the output stream operator.
     * Note that it is declared a "friend" function. 
     * This allows us to call private methods if necessary. 
     * Is there a private method that could be helpful?
     */
    template <typename U>
    friend std::ostream& operator <<(std::ostream& stream, DenseMatrix<U> dm)
    {
        dm.print(stream);
        return stream;
    }

private:
    uint n_rows;
    uint n_cols;
    int *elements;

    void print(std::ostream& stream)
    {
        for (uint i = 0; i < this->n_rows; i++)
        {
            for (uint j = 0; j < this->n_cols; j++)
            {
                // TODO: print out element (i, j). 
                stream << (*this)(i,j) << " ";
            }
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
    
    std::cout << "A: " << std::endl;
    std::cout << a << std::endl;
    std::cout << "B: " << std::endl << b << std::endl;
    DenseMatrix<int> c = a + b;
    std::cout << "C = A + B:" << std::endl << c << std::endl;
    return 0;
}
