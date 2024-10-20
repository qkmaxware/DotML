using System.Data;
using System.Numerics;

namespace DotML;

/// <summary>
/// Wrapper struct around value array providing matrix like functionality. Behaves like pass-by-reference rather than pass-by-value while minimizing heap allocations and dereferences. 
/// </summary>
public struct Matrix<T> where T:INumber<T> {
    private T[,] values;

    /// <summary>
    /// Number of elements in the matrix
    /// </summary>
    public int Size     => values.Length;
    /// <summary>
    /// Number of columns
    /// </summary>
    public int Columns  => values.GetLength(1);
    /// <summary>
    /// Number of rows
    /// </summary>
    public int Rows     => values.GetLength(0);

    /// <summary>
    /// Create a matrix from the given values
    /// </summary>
    /// <param name="values">values</param>
    public Matrix(T[,] values) {
        this.values = values;
    }

    /// <summary>
    /// Create a zero matrix of the given size
    /// </summary>
    /// <param name="rows">Number of rows</param>
    /// <param name="columns">Number of columns</param>
    public Matrix(int rows, int columns) : this(rows, columns, T.Zero) {}

    /// <summary>
    /// Create a matrix of the given size filled with a default value
    /// </summary>
    /// <param name="rows">Number of rows</param>
    /// <param name="columns">Number of columns</param>
    /// <param name="value">Default value</param>
    public Matrix(int rows, int columns, T value) {
        this.values = new T[rows, columns];
        for (var r = 0; r < rows; r++) {
            for (var c = 0; c < columns; c++) {
                this.values[r, c] = value;
            }
        }
    }

    /// <summary>
    /// Zero matrix of the given size
    /// </summary>
    /// <param name="rows">Number of rows</param>
    /// <param name="columns">Number of columns</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Zeros(int rows, int columns) {
        return new Matrix<T>(rows, columns);
    }

    /// <summary>
    /// Zero matrix of the given size
    /// </summary>
    /// <param name="size">Number of rows & columns</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Zeros(int size) {
        return new Matrix<T>(size, size);
    }

    /// <summary>
    /// Matrix of the given size filled with 1's
    /// </summary>
    /// <param name="rows">Number of rows</param>
    /// <param name="columns">Number of columns</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Ones(int rows, int columns) {
        return new Matrix<T>(rows, columns, T.One);
    }

    /// <summary>
    /// Matrix of the given size filled with 1's
    /// </summary>
    /// <param name="size">Number of rows & columns</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Ones(int size) {
        return new Matrix<T>(size, size, T.One);
    }

    /// <summary>
    /// Identity matrix of the given size with 1's along the diagonal
    /// </summary>
    /// <param name="rows">Number of rows</param>
    /// <param name="columns">Number of columns</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Identity(int rows, int columns) {
        var mat = new Matrix<T>(rows, columns, T.Zero);
        for (var i = 0; i < Math.Min(rows, columns); i++) {
            mat.values[i, i] = T.One;
        }
        return mat;
    }

    /// <summary>
    /// Identity matrix of the given size with 1's along the diagonal
    /// </summary>
    /// <param name="size">Number of rows & columns</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Identity(int size) {
        var mat = new Matrix<T>(size, size, T.Zero);
        for (var i = 0; i < size; i++) {
            mat.values[i, i] = T.One;
        }
        return mat;
    }

    /// <summary>
    /// Get the value of a matrix element at the given row, column index.
    /// </summary>
    /// <param name="row">Row index</param>
    /// <param name="col">Column index</param>
    /// <returns>value at row, column or zero if out of bounds</returns>
    public T this[int row, int col] {
        get {
            if (row >= 0 && row < Rows && col >= 0 && col < Columns)
                return values[row, col];
            else
                return T.Zero;
        }
    }

    /// <summary>
    /// Implicitly convert rectangular arrays to matrices
    /// </summary>
    /// <param name="values">Values</param>
    public static implicit operator Matrix<T> (T[,] values) => new Matrix<T>(values);

    /// <summary>
    /// Explicitly unwrap a matrix to a rectangular array
    /// </summary>
    /// <param name="mat">Matrix to unwrap</param>
    public static explicit operator T[,] (Matrix<T> mat) => mat.values;

    /// <summary>
    /// Map the values in this matrix to values of another type.
    /// </summary>
    /// <typeparam name="R">Result type</typeparam>
    /// <param name="mapping">Mapping function</param>
    /// <returns>New matrix, same size as the existing one but with elements modified by the mapping function</returns>
    public Matrix<R> Map<R>(Func<T, R> mapping) where R:INumber<R> {
        Matrix<R> result = new Matrix<R>(Rows, Columns);

        for (var row = 0; row < Rows; row++)
            for (var col = 0; col < Columns; col++)
                result.values[row, col] = mapping(this.values[row, col]);
        
        return result;
    }

    /// <summary>
    /// Matrix matrix multiplication
    /// </summary>
    /// <param name="a">LHS matrix</param>
    /// <param name="b">RHS matrix</param>
    /// <returns>matrix</returns>
    /// <exception cref="ArgumentException">Incompatible dimensions</exception>
    public static Matrix<T> operator * (Matrix<T> a, Matrix<T> b) {
        if (a.Columns != b.Rows)
            throw new ArgumentException("Incompatible dimensions for matrix multiplication");

        int rows = a.Rows;
        int cols = b.Columns;

        Matrix<T> result = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                for (int k = 0; k < a.Columns; k++)
                    result.values[i, j] += a.values[i, k] * b.values[k, j];

        return result;
    }   

    /// <summary>
    /// Matrix vector multiplication
    /// </summary>
    /// <param name="a">LHS matrix</param>
    /// <param name="b">RHS vector</param>
    /// <returns>vector</returns>
    /// <exception cref="ArgumentException">Incompatible dimensions</exception>
    public static Vec<T> operator * (Matrix<T> a, Vec<T> b) {
        if (a.Rows != b.Dimensionality)
            throw new ArgumentException("Incompatible dimensions for matrix/vector multiplication");

        T[] result = new T[a.Rows];
        for (int i = 0; i < a.Rows; i++) {
            T value = T.Zero;
            for (int j = 0; j < a.Columns; j++) {
                value = value + a.values[i, j] * b[i];
            }
            result[i] = value;
        }

        return result;
    }

    /// <summary>
    /// Matrix matrix addition
    /// </summary>
    /// <param name="a">LHS matrix</param>
    /// <param name="b">RHS matrix</param>
    /// <returns>matrix</returns>
    /// <exception cref="ArgumentException">Incompatible dimensions</exception>
    public static Matrix<T> operator + (Matrix<T> a, Matrix<T> b) {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            throw new ArgumentException("Incompatible dimensions for matrix addition");

        int rows = a.Rows;
        int cols = a.Columns;

        Matrix<T> result = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.values[i, j] = a.values[i, j] + b.values[i, j];

        return result;
    }

    /// <summary>
    /// Matrix matrix subtraction
    /// </summary>
    /// <param name="a">LHS matrix</param>
    /// <param name="b">RHS matrix</param>
    /// <returns>matrix</returns>
    /// <exception cref="ArgumentException">Incompatible dimensions</exception>
    public static Matrix<T> operator - (Matrix<T> a, Matrix<T> b) {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            throw new ArgumentException("Incompatible dimensions for matrix subtraction");

        int rows = a.Rows;
        int cols = a.Columns;

        Matrix<T> result = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.values[i, j] = a.values[i, j] - b.values[i, j];

        return result;
    }


}