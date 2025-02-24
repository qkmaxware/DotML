#define MATRIX_STORAGE_ROW_MAJOR // OR MATRIX_STORAGE_COL_MAJOR

using System.Collections;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Text;
using DotML.Network;

namespace DotML;

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct Matrix<T>: 
    IEnumerable<T>,
    IEquatable<Matrix<T>>,
    IMutableTensorLike<T>,
    IHtmlable
where T:INumber<T> {

    #region Data
    private T[] values; 

    /// <summary>
    /// Number of elements in the matrix
    /// </summary>
    public int Size {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => values.Length;
    }

    /// <summary>
    /// Number of rows in the matrix
    /// </summary>
    public int Rows {get; init;}

    /// <summary>
    /// Number of columns in the matrix
    /// </summary>
    public int Columns {get; init;}

    /// <summary>
    /// Number of tensor dimensions
    /// </summary>
    public int Dimensions => 2;

    /// <summary>
    /// Matrix shape (rows & columns)
    /// </summary>
    public Shape2D Shape => new Shape2D(Rows, Columns);

    #endregion

    #region Shape Queries

    /// <summary>
    /// Check if the matrix is a column matrix (only one column)
    /// </summary> 
    public bool IsColumnMatrix => this.Columns == 1;

    /// <summary>
    /// Check if the matrix is a row matrix (only one row)
    /// </summary> 
    public bool IsRowMatrix => this.Rows == 1;

    /// <summary>
    /// Check if the matrix is square
    /// </summary> 
    public bool IsSquareMatrix => this.Rows == this.Columns;

    #endregion
    #region Accessors

    /// <summary>
    /// Get the value of a matrix element by a sequential/flattened index
    /// </summary>
    /// <param name="index">Sequential index</param>
    /// <returns>value at the given flattened index</returns>
    public T this[int index] {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => values[index];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set => values[index] = value;
    }

    /// <summary>
    /// Tests if this matrix's internal storage is in row-major order
    /// </summary>
    public bool IsRowMajor {
        #if MATRIX_STORAGE_ROW_MAJOR
        get => true;
        #else
        get => false;
        #endif
    }

    /// <summary>
    /// Tests if this matrix's internal storage is in column-major order
    /// </summary>
    public bool IsColumnMajor {
        #if MATRIX_STORAGE_COL_MAJOR
        get => true;
        #else
        get => false;
        #endif
    }

    /// <summary>
    /// Get the value of a matrix element at the given row, column index.
    /// </summary>
    /// <param name="row">Row index</param>
    /// <param name="col">Column index</param>
    /// <returns>value at row, column</returns>
    public T this[int row, int col] {
        #if MATRIX_STORAGE_ROW_MAJOR
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => values[row * Columns + col];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => values[row * Columns + col] = value;
        #elif MATRIX_STORAGE_COL_MAJOR
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => values[row + Rows * col];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => values[row + Rows * col] = value;
        #endif
    }

    /// <summary>
    /// Slice the matrix to obtain a new matrix over the given 2D region
    /// </summary>
    /// <param name="row_span">rows to capture</param>
    /// <param name="column_span">columns to capture</param>
    /// <returns>sliced matrix</returns>
    public Matrix<T> this[Range row_span, Range column_span] {
        get {
            var self_rows = this.Rows;
            var self_cols = this.Columns;
            var (row_offset, rows) = row_span.GetOffsetAndLength(self_rows);
            var (col_offset, cols) = column_span.GetOffsetAndLength(self_cols);

            var values = new Matrix<T>(rows, cols, T.Zero);
            for (var row = 0; row < rows; row++) {
                var self_row = row_offset + row;
                if (self_row < 0 || self_row >= self_rows)
                    continue;

                for (var col = 0; col < cols; col++) {
                    var self_col = col_offset + col;
                    if (self_col < 0 || self_col >= self_cols)
                        continue;

                    values[row, col] = this[self_row, self_col];
                }
            }
            return values;
        }
    }

    #endregion
    #region Constuctors

    /// <summary>
    /// Create a matrix with the given shape and elements
    /// </summary>
    /// <param name="rows">number of rows</param>
    /// <param name="columns">number of columns</param>
    /// <param name="values">elements</param>
    private Matrix(int rows, int columns, T[] values) {
        this.Rows = rows;
        this.Columns = columns;
        this.values = values;
    }

    /// <summary>
    /// Create a 0x0 empty matrix
    /// </summary>
    public Matrix() : this(0, 0, Array.Empty<T>()) {}

    /// <summary>
    /// Create a new matrix with the given shape filled with the default(T) value
    /// </summary>
    /// <param name="rows">number of rows</param>
    /// <param name="columns">number of columns</param>
    public Matrix(int rows, int columns) : this(rows, columns, new T[rows * columns]) {}

    /// <summary>
    /// Create a new matrix with the given shape
    /// </summary>
    /// <param name="shape">shape</param>
    public Matrix(Shape2D shape) : this(shape.Rows, shape.Columns) {}

    /// <summary>
    /// Create a new matrix with the given shape where every element has the same value
    /// </summary>
    /// <param name="rows">number of rows</param>
    /// <param name="columns">number of columns</param>
    /// <param name="value">Value for each element</param>
    public Matrix(int rows, int columns, T value) : this(rows, columns) {
        this.values.AsSpan().Fill(value);
    }

    /// <summary>
    /// Create a new matrix from the given 2D array by copying the values
    /// </summary>
    /// <param name="values">values</param>
    public Matrix(T[,] values) : this(values.GetLength(0), values.GetLength(1)) {
        #if MATRIX_STORAGE_ROW_MAJOR
        // Both are in row-major so just copy as is
        var vspan = MemoryMarshal.CreateSpan(ref Unsafe.As<byte, T>(ref MemoryMarshal.GetArrayDataReference(values)), values.Length);
        var tspan = this.values.AsSpan();
        vspan.CopyTo(tspan);
        #else 
        // 2d arrays in C# are in row-major order but the array is stored as column-major
        for (var row = 0; row < this.Rows; row++) {
            for (var col = 0; col < this.Columns; col++) {
                // The this indexer will correctly map to row-major or column-major order
                this[row, col] = values[row, col];
            }
        }
        #endif
    }

    /// <summary>
    /// Create a new matrix that is a copy of another matrix
    /// </summary>
    /// <param name="original">original matrix</param>
    public Matrix(Matrix<T> original) : this (original.Rows, original.Columns) {
        original.values.AsSpan().CopyTo(this.values);
    }

    /// <summary>
    /// Create a new matrix that is a copy of another matrix
    /// </summary>
    /// <returns>copy of the matrix</returns>
    public Matrix<T> Clone() => new Matrix<T>(this);

    /// <summary>
    /// Zero matrix of the given size
    /// </summary>
    /// <param name="rows">Number of rows</param>
    /// <param name="columns">Number of columns</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Zeros(int rows, int columns) {
        return new Matrix<T>(rows, columns, T.Zero);
    }

    /// <summary>
    /// Zero matrix of the given size
    /// </summary>
    /// <param name="size">Number of rows & columns</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Zeros(int size) {
        return new Matrix<T>(size, size, T.Zero);
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
            mat[i, i] = T.One;
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
            mat[i, i] = T.One;
        }
        return mat;
    }

    /// <summary>
    /// Generate a matrix with the given values provided by a generator function (such as System.Random.NextDouble)
    /// </summary>
    /// <param name="rows">number of rows</param>
    /// <param name="cols">number of columns</param>
    /// <param name="generator">generator function</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Generate(int rows, int cols, Func<T> generator) {
        var mat = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                mat[i, j] = generator();
        return mat;
    }

    /// <summary>
    /// Create a matrix whose values are the average of all matrices. Matrices should be the same size.
    /// </summary>
    /// <param name="matrices">list of matrices</param>
    /// <returns>matrix with averaged values</returns>
    public static Matrix<T> Average(IEnumerable<Matrix<T>> matrices) {
        Matrix<T> values = new Matrix<T>();
        bool has_set = false;

        // Sum across all elements
        int rows = 0;
        int columns = 0;
        T count = T.Zero;
        foreach (var matrix in matrices) {
            if (!has_set) {
                rows = matrix.Rows;
                columns =  matrix.Columns;
                values = new Matrix<T>(rows, columns);
                has_set = true;
            }

            for (var r = 0; r < rows; r++) {
                for (var c = 0; c < columns; c++) {
                    values[r, c] += matrix[r, c];
                }
            }

            count = count + T.One;
        }

        if (!has_set || count == T.Zero) {
            throw new DivideByZeroException();
        }

        // Divide by count to average it
         for (var r = 0; r < rows; r++) {
            for (var c = 0; c < columns; c++) {
                values[r, c] = values[r, c] / count;
            }
        }

        return values;
    }

    /// <summary>
    /// Create a column matrix from the given values
    /// </summary>
    /// <param name="args">matrix elements</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Column(params T[] args) {
        var values = new Matrix<T>(args.Length, 1);
        for (var i = 0; i < args.Length; i++) {
            values[i, 0] = args[i];
        }
        return values;
    }

    /// <summary>
    /// Create a column matrix from the given values
    /// </summary>
    /// <param name="args">matrix elements</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Column(Vec<T> args) {
        var values = new Matrix<T>(args.Dimensionality, 1);
        for (var i = 0; i < args.Dimensionality; i++) {
            values[i, 0] = args[i];
        }
        return values;
    }

    /// <summary>
    /// Create a row matrix from the given values
    /// </summary>
    /// <param name="args">matrix elements</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Row(params T[] args) {
        var values = new Matrix<T>(1, args.Length);
        for (var i = 0; i < args.Length; i++) {
            values[0, i] = args[i];
        }
        return values;
    }

    /// <summary>
    /// Create a row matrix from the given values
    /// </summary>
    /// <param name="args">matrix elements</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Row(Vec<T> args) {
        var values = new Matrix<T>(1, args.Dimensionality);
        for (var i = 0; i < args.Dimensionality; i++) {
            values[0, i] = args[i];
        }
        return values;
    }

    /// <summary>
    /// Create a matrix of a given size from flattened data. Flattened data should be stored in the same ordering as the matrix's internal storage.
    /// </summary>
    /// <param name="rows">matrix rows</param>
    /// <param name="columns">matrix columns</param>
    /// <param name="data">matrix elements</param>
    /// <returns>matrix</returns>  
    public static Matrix<T> FromFlattened(int rows, int columns, T[] data) {
        rows = Math.Max(0, rows);
        columns = Math.Max(0, columns);
        var size = rows * columns;
        if (size != data.Length)
            throw new ArgumentException("Flattened data length doesn't match give matrix dimensions");
        return new Matrix<T>(rows, columns, data);
    } 

    /// <summary>
    /// Create a matrix from a C# rectangular array in row-major order.
    /// </summary>
    /// <param name="array">Array of matrix values</param>
    /// <returns>matrix</returns>
    public static Matrix<T> FromRectangular(T[,] array) {
        return new Matrix<T>(array);
    }

    /// <summary>
    /// Create a matrix from a jagged array of rows.
    /// </summary>
    /// <param name="array">Array with each element representing a single row of matrix values</param>
    /// <returns>matrix</returns>
    public static Matrix<T> FromJagged(T[][] array) {
        var rows = array.Length;
        var columns = array.Select(x => x.Length).Max();

        var result = new Matrix<T>(rows, columns, T.Zero);
        for (var r = 0; r < rows; r++) {
            var row = array[r];
            for (var c = 0; c < row.Length; c++) {
                result[r, c] = row[c];
            }
        }
        return result;
    }

    #endregion
    #region Methods
    /// <summary>
    /// Pad a matrix with 0's to a given size
    /// </summary>
    /// <param name="top">top padding</param>
    /// <param name="right">right padding</param>
    /// <param name="bottom">bottom padding</param>
    /// <param name="left">left padding</param>
    /// <returns>padded matrix</returns>
    public Matrix<T> Pad(int top = 0, int right = 0, int bottom = 0, int left = 0) {
        top         = Math.Max(0, top);
        right       = Math.Max(0, right);
        bottom      = Math.Max(0, bottom);
        left        = Math.Max(0, left);

        var rows    = top + bottom + this.Rows;
        var columns = left + right + this.Columns;
        var matrix  = new Matrix<T>(rows, columns);

        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Columns; c++) {
                matrix[r + left, c + top] = this[r, c];
            }
        }

        return matrix;
    }

    /// <summary>
    /// Mirror the matrix across the selected axis
    /// </summary>
    /// <param name="x">mirror over the x axis (swap columns)</param>
    /// <param name="y">mirror over the y axis (swap rows)</param>
    /// <returns>mirrored matrix</returns>
    public Matrix<T> Mirror(bool x = false, bool y = false) {
        var rows    = this.Rows;
        var columns = this.Columns;
        var matrix  = new Matrix<T>(rows, columns);

        for (var r = 0; r < rows; r++) {
            for (var c = 0; c < columns; c++) {
                var x_index = x switch {
                    true => columns - 1 - c,
                    false => c 
                };
                var y_index = y switch {
                    true => rows - 1 - r,
                    false => r
                };
                matrix[y_index, x_index] = this[r, c];
            }
        }

        return matrix;
    }

    /// <summary>
    /// Apply a transformation to each element in this matrix and return a new matrix with the transformed elements
    /// </summary>
    /// <typeparam name="R">result type</typeparam>
    /// <param name="transformation">transformation function</param>
    /// <returns>matrix with transformed elements</returns>
    public Matrix<R> Transform<R>(Func<T,R> transformation) where R:INumber<R> {
        var result = new Matrix<R>(this.Rows, this.Columns);
        var r = result.values.AsSpan();
        var s = values.AsSpan();
        var len = s.Length;
        for (var i = 0; i < len; i++)
            r[i] = transformation(s[i]);
        return result;
    }

    /// <summary>
    /// Apply a transformation to each element in this matrix and return a new matrix with the transformed elements
    /// </summary>
    /// <typeparam name="R">result type</typeparam>
    /// <param name="transformation">transformation function</param>
    /// <returns>matrix with transformed elements</returns>
    public Matrix<R> Transform<R>(Func<(int Row, int Column), T,R> transformation) where R:INumber<R> {
        var result = new Matrix<R>(this.Rows, this.Columns);
        for (var row = 0; row < Rows; row++)
            for (var col = 0; col < Columns; col++)
                result[row, col] = transformation((row, col), this[row, col]);
        return result;
    }

    /// <summary>
    /// Perform an element-wise operation between this matrix and another matrix
    /// </summary>
    /// <typeparam name="R">result element type</typeparam>
    /// <param name="other">other matrix</param>
    /// <param name="operation">element-wise operation</param>
    /// <returns>matrix that is the result of applying the element-wise operation</returns>
    public Matrix<R> ElementWise<R>(Matrix<T> other, Func<T, T, R> operation) where R:INumber<R> {
        if (this.Rows != other.Rows || this.Columns != other.Columns)
            throw new ArithmeticException($"Invalid dimensions for element-wise operation between {this.Shape} and {other.Shape}.");

        var result = new Matrix<R>(this.Rows, this.Columns);
        var r = result.values.AsSpan();
        var s = values.AsSpan(); var o = other.values.AsSpan();
        var len = s.Length;
        for (var i = 0; i < len; i++)
            r[i] = operation(s[i], o[i]);
        return result;
    }

    /// <summary>
    /// Apply a transformation to each element in this matrix replacing the values as they are transformed
    /// </summary>
    /// <param name="transformation">transformation function</param>
    public void Apply(Func<T,T> transformation) {
        var s = values.AsSpan();
        for (var i = 0; i < s.Length; i++)
            s[i] = transformation(s[i]);
    }

    /// <summary>
    /// Apply a transformation to each element in this matrix replacing the values as they are transformed
    /// </summary>
    /// <param name="transformation">transformation function</param>
    public void Apply(Func<(int Row, int Column), T,T> transformation) {
        for (var row = 0; row < Rows; row++)
            for (var col = 0; col < Columns; col++)
                this[row, col] = transformation((row, col), this[row, col]);
    }

    /// <summary>
    /// Perform an element-wise operation between this matrix and another matrix. Results are stored in this matrix.
    /// </summary>
    /// <typeparam name="R">result element type</typeparam>
    /// <param name="other">other matrix</param>
    /// <param name="operation">element-wise operation</param>
    public void ElementWiseInplace(Matrix<T> other, Func<T, T, T> operation) {
        if (this.Rows != other.Rows || this.Columns != other.Columns)
            throw new ArithmeticException($"Invalid dimensions for element-wise operation between {this.Shape} and {other.Shape}.");

        var s = values.AsSpan(); var o = other.values.AsSpan();
        var len = s.Length;
        for (var i = 0; i < len; i++)
            s[i] = operation(s[i], o[i]);
    }

    /// <summary>
    /// Perform an element-wise operation between this matrix and another matrix. Results are stored in the result matrix.
    /// </summary>
    /// <param name="result">matrix storing the results</param>
    /// <param name="lhs">first matrix</param>
    /// <param name="rhs">second matrix</param>
    /// <param name="operation">element-wise operation</param>
    /// <exception cref="ArithmeticException">thrown when the matrix shapes are incompatible</exception>
    public static void ElementWiseInplace(Matrix<T> result, Matrix<T> lhs, Matrix<T> rhs, Func<T, T, T> operation) {
        if (result.Rows != lhs.Rows || result.Columns != lhs.Columns || result.Rows != rhs.Rows || result.Columns != rhs.Columns)
            throw new ArithmeticException($"Invalid dimensions for element-wise operation between {lhs.Shape} and {rhs.Shape}.");
        
        var r = result.values.AsSpan();
        var s = lhs.values.AsSpan(); var o = rhs.values.AsSpan();
        var len = result.Size;
        for (var i = 0; i < len; i++)
            r[i] = operation(s[i], o[i]);
    }

    /// <summary>
    /// Extract a given row from the matrix as a vector
    /// </summary>
    /// <param name="rowIndex">row index</param>
    /// <returns>vector representation of the row</returns>
    public Vec<T> ExtractRowVector(int rowIndex) {
        T[] vec = new T[this.Columns];
        for (var i = 0; i < this.Columns; i++) {
            vec[i] = this[rowIndex, i];
        }
        return Vec<T>.Wrap(vec);
    }

    #if MATRIX_STORAGE_ROW_MAJOR
    /// <summary>
    /// Extract a given row from the matrix as a Span<typeparamref name="T"/>>
    /// </summary>
    /// <param name="rowIndex">row index</param>
    /// <returns>Span<typeparamref name="T"/>> representation of the row</returns>
    public Span<T> ExtractRowSpan(int rowIndex) {
        return values.AsSpan().Slice(rowIndex * Columns, Columns);
    }
    #endif

    /// <summary>
    /// Extract a given column from the matrix as a vector
    /// </summary>
    /// <param name="rowIndex">column index</param>
    /// <returns>vector representation of the column</returns>
    public Vec<T> ExtractColumnVector(int colIndex) {
        T[] vec = new T[this.Rows];
        for (var i = 0; i < this.Rows; i++) {
            vec[i] = this[i, colIndex];
        }
        return Vec<T>.Wrap(vec);
    }

    /// <summary>
    /// Transposition of the matrix
    /// </summary>
    /// <returns>transposed matrix</returns>
    public Matrix<T> Transpose() {
        var rows = this.Rows;
        var cols = this.Columns;
        var transposed = new Matrix<T>(cols, rows);
        for (var r = 0; r < rows; r++)
            for (var c = 0; c < cols; c++)
                transposed[c, r] = this[r, c];
        return transposed;
    }

    /// <summary>
    /// Get the length of a given dimension
    /// </summary>
    /// <param name="index">dimension index</param>
    /// <returns>dimension length</returns>
    public int GetDimension(int index) => index switch {
        0 => Rows,
        1 => Columns,
        _ => 1
    };

    /// <summary>
    /// Get the value of a matrix element at the given row, column index.
    /// </summary>
    /// <param name="indices">row and column indices</param>
    /// <returns>element</returns>
    /// <exception cref="IndexOutOfRangeException">thrown when index is out of range</exception>
    public T GetElementAt(params int[] indices) {
        if (indices.Length != 2)
            throw new IndexOutOfRangeException();
        return this[indices[0], indices[1]];
    }

    /// <summary>
    /// Set a particular element in the tensor by index
    /// </summary>
    /// <param name="value">value to store</param>
    /// <param name="indices">list of indexes for each dimension, should match the number of dimensions</param>
    /// <returns>element at the given index</returns>
    public void SetElementAt(T value, params int[] indices) {
        if (indices.Length != 2)
            throw new IndexOutOfRangeException();
        this[indices[0], indices[1]] = value;
    }

    /// <summary>
    /// Perform a transpose convolution of this matrix using the provided kernel
    /// </summary>
    /// <param name="kernel">kernel</param>
    /// <param name="flip_kernel">should the kernel be flipped across both dimensions or not</param>
    /// <param name="outputStrideX">stride across the x-axis (columns) of the output matrix</param>
    /// <param name="outputStrideY">stride across the y-axis (rows) of the output matrix</param>
    /// <param name="paddingX">horizontal padding of the input matrix</param>
    /// <param name="paddingY">vertical padding of the input matrix</param>
    /// <param name="outputPaddingX">horizontal padding of the output matrix</param>
    /// <param name="outputPaddingY">vertical padding of the output matrix</param>
    /// <returns>transpose convolved matrix</returns>
    public Matrix<T> TransposeConvolve(Matrix<T> kernel, bool flip_kernel = false, int outputStrideX = 1, int outputStrideY = 1, int outputPaddingX = 0, int outputPaddingY = 0, T? bias = default(T)) {
        var in_rows = this.Rows;
        var in_cols = this.Columns;

        var kernel_rows = kernel.Rows;
        var kernel_rows_m1 = kernel_rows - 1;
        var kernel_cols = kernel.Columns;
        var kernel_cols_m1 = kernel_cols - 1;

        // TODO account for stride in output size calculation
        // https://www.digitalocean.com/community/tutorials/transpose-convolution
        // Transpose Convolution Output Size = (Input Size - 1) * Strides + Filter Size - 2 * Padding + Output Padding
        var out_cols = (this.Columns - 1) * outputStrideX + kernel.Columns - 2 * outputPaddingX; 
        var out_rows = (this.Rows - 1) * outputStrideY + kernel.Rows - 2 * outputPaddingY;

        var result = new Matrix<T>(out_rows, out_cols, bias ?? T.Zero);
        for (var r = 0; r < in_rows; r++) {
            var region_start_y = r * outputStrideY - outputPaddingY;
            var region_end_y = region_start_y + kernel_rows;

            for (var c = 0; c < in_cols; c++) {
                var region_start_x = c * outputStrideX - outputPaddingX;
                var region_end_x = region_start_x + kernel_cols;

                var i = this[r, c];

                for (int out_y = region_start_y, ky = 0; out_y < region_end_y; out_y++, ky++) {
                    if (out_y < 0 || out_y >= out_rows)
                        continue;

                    for (int out_x = region_start_x, kx = 0; out_x < region_end_x; out_x++, kx++) {
                        if (out_x < 0 || out_x >= out_cols)
                            continue;

                        if (flip_kernel) {
                            result[out_y, out_x] += i * kernel[kernel_rows_m1 - ky, kernel_cols_m1 - kx];
                        } else {
                            result[out_y, out_x] += i * kernel[ky, kx];
                        }
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Perform a convolution of this matrix using the provided kernel
    /// </summary>
    /// <param name="kernel">kernel</param>
    /// <param name="strideX">stride across the x-axis (columns)</param>
    /// <param name="strideY">stride across the y-axis (rows)</param>
    /// <param name="paddingX">horizontal padding of this matrix</param>
    /// <param name="paddingY">vertical padding of this matrix</param>
    /// <returns>convolution of this matrix</returns>
    public Matrix<T> Convolve(Matrix<T> kernel, int strideX = 1, int strideY = 1, int paddingX = 0, int paddingY = 0, T? bias = default(T)) {
        var filterRows          = kernel.Rows;   
        var filterColumns       = kernel.Columns;  
        var paddingRows         = paddingY;
        var paddingColumns      = paddingX;  

        var input               = this;
        var inputRows           = input.Rows;
        var inputColumns        = input.Columns;

        // Same math as in ConvolutionLayer.cs for output shape
        var outputRows          = (inputRows - filterRows + 2 * paddingRows) / strideY + 1; 
        var outputColumns       = (inputColumns - filterColumns + 2 * paddingColumns) / strideX + 1;  

        var result              = new Matrix<T>(outputRows, outputColumns, bias ?? T.Zero);
        for (var y = 0; y < outputRows; y++) {
        for (var x = 0; x < outputColumns; x++) {
            var startY = y * strideY - paddingRows;
            var startX = x * strideX - paddingColumns;

            var total_sum = T.Zero;
            for (int ky = 0; ky < filterRows; ky++) {
                var inY = startY + ky;
                if (inY < 0 || inY >= inputRows) continue; // Skip out-of-bounds rows

                for (int kx = 0; kx < filterColumns; kx++) {
                    var inX = startX + kx;
                    if (inX < 0 || inX >= inputColumns) continue; // Skip out-of-bounds columns
                    
                    total_sum += input[inY, inX] * kernel[ky, kx];
                }
            }

            result[y, x] += total_sum;
        }}

        return result;
    }

    /// <summary>
    /// Perform a convolution of all input matrices with their paired kernel and summing the results
    /// </summary>
    /// <param name="inputs">input matrices</param>
    /// <param name="kernels">kernels to apply to input matrices</param>
    /// <param name="strideX">stride across the x-axis (columns)</param>
    /// <param name="strideY">stride across the y-axis (rows)</param>
    /// <param name="paddingX">horizontal padding of this matrix</param>
    /// <param name="paddingY">vertical padding of this matrix</param>
    /// <returns>convolution of all input matrices with their matching kernel</returns>
    public static Matrix<T> ConvolveEach(IEnumerable<Matrix<T>> inputs, IEnumerable<Matrix<T>> kernels, int strideX = 1, int strideY = 1, int paddingX = 0, int paddingY = 0, T? bias = default(T)) {
        var first_kernel        = kernels.First();
        var filterRows          = first_kernel.Rows;   
        var filterColumns       = first_kernel.Columns;  
        var paddingRows         = paddingY;
        var paddingColumns      = paddingX;  

        var first_input         = inputs.First();
        var inputRows           = first_input.Rows;
        var inputColumns        = first_input.Columns;

        // Same math as in ConvolutionLayer.cs for output shape
        var outputRows          = (inputRows - filterRows + 2 * paddingRows) / strideY + 1; 
        var outputColumns       = (inputColumns - filterColumns + 2 * paddingColumns) / strideX + 1;  

        var result              = new Matrix<T>(outputRows, outputColumns, bias ?? T.Zero);
        foreach (var (input, kernel) in inputs.Zip(kernels)) {
            for (var y = 0; y < outputRows; y++) {
            for (var x = 0; x < outputColumns; x++) {
                var startY = y * strideY - paddingRows;
                var startX = x * strideX - paddingColumns;

                var total_sum = T.Zero;
                for (int ky = 0; ky < filterRows; ky++) {
                    var inY = startY + ky;
                    if (inY < 0 || inY >= inputRows) continue; // Skip out-of-bounds rows

                    for (int kx = 0; kx < filterColumns; kx++) {
                        var inX = startX + kx;
                        if (inX < 0 || inX >= inputColumns) continue; // Skip out-of-bounds columns
                        
                        total_sum += input[inY, inX] * kernel[ky, kx];
                    }
                }

                result[y, x] += total_sum;
            }}
        }

        return result;
    }

    /// <summary>
    /// Create a vector by aggregating the values across all rows into a single row vector
    /// </summary>
    /// <param name="aggregator">aggregation function</param>
    /// <returns>aggregated vector</returns>
    public Vec<T> AggregateOverRows(Func<T, T, T> aggregator, T initial) {
        T[] result = new T[this.Columns];
        var rows = this.Rows;
        var src = this;
        for (var x = 0; x < this.Columns; x++) {
            T aggregate = initial;
            for (var row = 0; row < rows; row++) {
                aggregate = aggregator(aggregate, src[row, x]);
            }
            result[x] = aggregate;
        };
        return Vec<T>.Wrap(result);
    }

    /// <summary>
    /// Create a vector by aggregating the values across all columns into a single column vector
    /// </summary>
    /// <param name="aggregator">aggregation function</param>
    /// <returns>aggregated vector</returns>
    public Vec<T> AggregateOverColumns(Func<T, T, T> aggregator, T initial) {
        T[] result = new T[this.Rows];
        var columns = this.Columns;
        var src = this;
        for (var x = 0; x < this.Rows; x++) {
            T aggregate = initial;
            for (var col = 0; col < columns; col++) {
                aggregate = aggregator(aggregate, src[x, col]);
            }
            result[x] = aggregate;
        };
        return Vec<T>.Wrap(result);
    }

    /// <summary>
    /// Reshape the elements of this matrix into one or more matrices of a different shape.
    /// </summary>
    /// <param name="size">shape of the matrix</param>
    /// <param name="channels">number of matrices</param>
    /// <returns>matrices</returns>
    public IEnumerable<Matrix<T>> Reshape(Shape2D size, int channels) {       
        return Reshape(Enumerable.Repeat(0, channels).Select(x => size));
    } 

    /// <summary>
    /// Reshape the elements of this matrix into one or more matrices of a different shape.
    /// </summary>
    /// <param name="shapes">list of shapes</param>
    /// <returns>matrices</returns>
    public IEnumerable<Matrix<T>> Reshape(params Shape2D[] shapes) => Reshape((IEnumerable<Shape2D>)shapes);

    /// <summary>
    /// Reshape the elements of this matrix into one or more matrices of a different shape.
    /// </summary>
    /// <param name="shapes">list of shapes</param>
    /// <returns>matrices</returns>
    public IEnumerable<Matrix<T>> Reshape(Shape3D shape) => Reshape(shape.EnumerateSubshapes());

    /// <summary>
    /// Reshape the elements of this matrix into one or more matrices of a different shape.
    /// </summary>
    /// <param name="shapes">list of shapes</param>
    /// <returns>matrices</returns>
    public IEnumerable<Matrix<T>> Reshape(IEnumerable<Shape2D> shapes) {
        var row_index = 0;
        var col_index = 0;
        var values = this.values;
        var col_count = this.Columns;
        var row_count = this.Rows;

        foreach (var shape in shapes) {
            var rows = shape.Rows;
            var cols = shape.Columns;
            var mtx = new Matrix<T>(rows, cols);
            for (int row = 0; row < rows; row++) {
                if (row_index >= row_count)
                    continue;

                for (int col = 0; col < cols; col++) {
                    mtx[row, col] = this[row_index, col_index]; // No 1D to 2D division anymore (faster?)
                    col_index++;
                    if (col_index >= col_count) {
                        row_index++;
                        col_index = 0;
                    }
                }
            }
            yield return mtx;
        }
    }

    /// <summary>
    /// String representation of the matrix in a matlab/octave syntax
    /// </summary>
    /// <returns>matrix values</returns>
    public override string ToString() {
        StringBuilder str = new StringBuilder();
        str.Append('[');
        for (int i = 0; i < Rows; i++) {
            if (i != 0)
                str.Append(';');
            for (int j = 0; j < Columns; j++) {
                if (j != 0)
                    str.Append(',');
                str.Append(this[i, j]);
            }
        }
        str.Append(']');
        return str.ToString();
    }

    /// <summary>
    /// String representation of the matrix in a HTML compatible MathML
    /// </summary>
    /// <returns>MathML HTML string</returns>
    public void ToHtml(TextWriter writer) {
        writer.Write("<math><mrow>");
            writer.Write("<mo>[</mo>");
            writer.Write("<mtable>");
            for (int i = 0; i < Rows; i++) {
                writer.Write("<mtr>");
                for (int j = 0; j < Columns; j++) {
                    writer.Write("<mtd>");
                    writer.Write("<mn>"); writer.Write(this[i, j]); writer.Write("</mn>");
                    writer.Write("</mtd>");
                }
                writer.Write("</mtr>");
            }   
            writer.Write("</mtable>");
            writer.Write("<mo>]</mo>");
        writer.Write("</mrow></math>");
    }

    #endregion
    #region Operators

    /// <summary>
    /// Perform element-wise (hadamard) multiplication between this matrix and another
    /// </summary>
    /// <param name="rhs">second matrix</param>
    /// <returns>element-wise multiplication</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]  
    public Matrix<T> HadamardWith(Matrix<T> rhs) {
        return this.ElementWise(rhs, (l, r) => l * r);
    }

    /// <summary>
    /// Perform element-wise (hadamard) multiplication between this matrix and another
    /// </summary>
    /// <param name="rhs">second matrix</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]  
    public void HadamardWithInplace(Matrix<T> rhs) {
        this.ElementWiseInplace(rhs, (l, r) => l * r);
    }

    /// <summary>
    /// Add this matrix and another matrix together
    /// </summary>
    /// <param name="rhs">rhs matrix</param>
    /// <returns>result of the matrix addition</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]  
    public Matrix<T> AddWith(Matrix<T> rhs) {
        return this.ElementWise(rhs, (l, r) => l + r);
    }

    /// <summary>
    /// Add this matrix and another matrix together
    /// </summary>
    /// <param name="rhs">rhs matrix</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]  
    public void AddWithInplace(Matrix<T> rhs) {
        this.ElementWiseInplace(rhs, (l, r) => l + r);
    }

    /// <summary>
    /// Add this matrix and another matrix together
    /// </summary>
    /// <param name="lhs">lhs matrix</param>
    /// <param name="rhs">rhs matrix</param>
    /// <returns>result of the matrix addition</returns>
    public static Matrix<T> operator + (Matrix<T> lhs, Matrix<T> rhs) => lhs.AddWith(rhs);

    /// <summary>
    /// Subtract this matrix and another matrix together
    /// </summary>
    /// <param name="rhs">rhs matrix</param>
    /// <returns>result of the matrix subtraction</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]  
    public Matrix<T> SubtractWith(Matrix<T> rhs) {
        return this.ElementWise(rhs, (l, r) => l - r);
    }

    /// <summary>
    /// Subtract this matrix and another matrix together
    /// </summary>
    /// <param name="rhs">rhs matrix</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]  
    public void SubtractWithInplace(Matrix<T> rhs) {
        this.ElementWiseInplace(rhs, (l, r) => l - r);
    }

    /// <summary>
    /// Subtract this matrix and another matrix together
    /// </summary>
    /// <param name="lhs">lhs matrix</param>
    /// <param name="rhs">rhs matrix</param>
    /// <returns>result of the matrix subtraction</returns>
    public static Matrix<T> operator - (Matrix<T> lhs, Matrix<T> rhs) => lhs.SubtractWith(rhs);

    /// <summary>
    /// Multiply a matrix by a scalar value
    /// </summary>
    /// <param name="scale">scalar</param>
    /// <returns>matrix</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]  
    public Matrix<T> ScaleBy(T scale) {
        return this.Transform((x) => x * scale);
    }

    /// <summary>
    /// Multiply a matrix by a scalar value
    /// </summary>
    /// <param name="scale">scalar</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]  
    public void ScaleByInplace(T scale) {
        this.Apply((x) => x * scale);
    }

    /// <summary>
    /// Multiply a matrix by a scalar value
    /// </summary>
    /// <param name="lhs">lhs matrix</param>
    /// <param name="rhs">rhs value</param>
    /// <returns>result of the matrix scaling</returns>
    public static Matrix<T> operator * (Matrix<T> lhs, T rhs) => lhs.ScaleBy(rhs);

    /// <summary>
    /// Multiply a matrix by a scalar value
    /// </summary>
    /// <param name="lhs">lhs matrix</param>
    /// <param name="rhs">rhs value</param>
    /// <returns>result of the matrix scaling</returns>
    public static Matrix<T> operator * (T lhs, Matrix<T> rhs) => rhs.ScaleBy(lhs);

    /// <summary>
    /// Divide a matrix by a scalar value
    /// </summary>
    /// <param name="lhs">lhs matrix</param>
    /// <param name="rhs">rhs value</param>
    /// <returns>result of the matrix scaling</returns>
    public static Matrix<T> operator / (Matrix<T> lhs, T rhs) => lhs.ScaleBy(T.One / rhs);

    /// <summary>
    /// Matrix matrix multiplication
    /// </summary>
    /// <param name="b">RHS matrix</param>
    /// <returns>matrix</returns>
    /// <exception cref="ArithmeticException">Incompatible dimensions</exception>
    public Matrix<T> MultiplyWith(Matrix<T> b) {
        var a = this;
        if (a.Columns != b.Rows)
            throw new ArithmeticException($"Incompatible dimensions for matrix multiplication {a.Rows}x{a.Columns} · {b.Rows}x{b.Columns}");

        int rows = a.Rows;
        int cols = b.Columns;
        int innerDim = a.Columns;

        var result = new Matrix<T>(rows, cols);
        Parallel.For(0, rows, i => { // This is the only thing imma leave as Parallel.For in Matrix for now
            for (int j = 0; j < cols; j++) {
                T sum = T.Zero;            
                for (int k = 0; k < innerDim; k++) {
                    sum += a[i, k] * b[k, j];
                }
                result[i, j] = sum;
            }
        });
        return result;
    }

    /// <summary>
    /// Multiply the transpose of this matrix with another
    /// </summary>
    /// <param name="b">RHS matrix</param>
    /// <returns>this transposed times RHS</returns>
    /// <exception cref="ArithmeticException">Incompatible dimensions</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Matrix<T> MultiplyTransposedWith(Matrix<T> b) {
        var a = this;
        var this_T_rows = a.Columns;
        var this_T_columns = a.Rows;
        
        if (this_T_columns != b.Rows)
            throw new ArithmeticException($"Incompatible dimensions for matrix multiplication {this_T_rows}x{this_T_columns} · {b.Rows}x{b.Columns}");

        int rows = this_T_rows;
        int cols = b.Columns;
        int innerDim = this_T_columns;

        var result = new Matrix<T>(rows, cols);
        Parallel.For(0, rows, i => {
            for (int j = 0; j < cols; j++) {
                T sum = T.Zero;            
                for (int k = 0; k < innerDim; k++) {
                    sum += a[k, i] * b[k, j];
                }
                result[i, j] = sum;
            }
        });
        return result;
    }

    /// <summary>
    /// Matrix vector multiplication
    /// </summary>
    /// <param name="b">RHS vector</param>
    /// <returns>vector</returns>
    /// <exception cref="ArithmeticException">Incompatible dimensions</exception>
    public Vec<T> MultiplyWith(Vec<T> b) {
        var a = this;
        if (a.Rows != b.Dimensionality)
            throw new ArithmeticException($"Incompatible dimensions for matrix/vector multiplication {a.Rows}x{a.Columns} · {b.Dimensionality}x1");

        T[] result = new T[a.Rows];
        Parallel.For(0, a.Rows, (i) => {
            T value = T.Zero;
            for (int j = 0; j < a.Columns; j++) {
                value = value + a[i, j] * b[j];
            }
            result[i] = value;
        });
        
        return Vec<T>.Wrap(result);
    }

    /// <summary>
    /// Matrix matrix multiplication
    /// </summary>
    /// <param name="a">LHS matrix</param>
    /// <param name="b">RHS matrix</param>
    /// <returns>matrix</returns>
    /// <exception cref="ArithmeticException">Incompatible dimensions</exception>
    public static Matrix<T> operator * (Matrix<T> a, Matrix<T> b) => a.MultiplyWith(b);

    /// <summary>
    /// Matrix vector multiplication
    /// </summary>
    /// <param name="a">LHS matrix</param>
    /// <param name="b">RHS vector</param>
    /// <returns>vector</returns>
    /// <exception cref="ArithmeticException">Incompatible dimensions</exception>
    public static Vec<T> operator * (Matrix<T> a, Vec<T> b) => a.MultiplyWith(b);

    #endregion
    #region Equality

    /// <summary>
    /// Test if this matrix is equal to another matrix
    /// </summary>
    /// <param name="other">second matrix</param>
    /// <returns>true if the matrices are equivalent</returns>
    public bool Equals(Matrix<T> other) {
        return this.SequenceEqual(other);
    }

    /// <summary>
    /// Test if this matrix is equal to another matrix
    /// </summary>
    /// <param name="other">second matrix</param>
    /// <returns>true if the matrices are equivalent</returns>
    public override bool Equals(object? other) {
        if (other is Matrix<T> otherMat)
            return this.Equals(otherMat);
        else
            return base.Equals(other);
    }   

    /// <summary>
    /// Test if this matrix is equal to another matrix
    /// </summary>
    /// <param name="rhs">right-hand-side matrix</param>
    /// <param name="lhs">left-hand-side matrix</param>
    /// <returns>true of matrices are equal</returns>
    public static bool operator == (Matrix<T> rhs, Matrix<T> lhs) {
        return rhs.Equals(lhs);
    }

    /// <summary>
    /// Test if this matrix is not equal to another matrix
    /// </summary>
    /// <param name="rhs">right-hand-side matrix</param>
    /// <param name="lhs">left-hand-side matrix</param>
    /// <returns>true of matrices are equal</returns>
    public static bool operator != (Matrix<T> rhs, Matrix<T> lhs) {
        return !rhs.Equals(lhs);
    }

    /// <summary>
    /// Returns a hash code for this matrix
    /// </summary>
    /// <returns>hash-code</returns>
    public override int GetHashCode() {
        return HashCode.Combine(Rows, Columns);
    }

    #endregion
    #region Conversion

    /// <summary>
    /// Create a span over the entire 2D matrix
    /// </summary>
    /// <returns>span over the matrix elements</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> AsSpan() => values.AsSpan();

    /// <summary>
    /// Create a read-only span over the entire 2D matrix
    /// </summary>
    /// <returns>read-only span over the matrix elements</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> AsReadOnlySpan() => values.AsSpan();

    #endregion
    #region Enumeration

    /// <summary>
    /// Enumerate over a flattened version of the matrix row-by-row.
    /// </summary>
    /// <returns>Matrix values as a single array</returns>
    public IEnumerable<T> FlattenRows() {
        for (var row = 0; row < Rows; row++) {
            for (var col = 0; col < Columns; col++) {
                yield return this[row, col];
            }
        }
    }

    /// <summary>
    /// Enumerate over a flattened version of the matrix column-by-column.
    /// </summary>
    /// <returns>Matrix values as a single array</returns>
    public IEnumerable<T> FlattenColumns() {
        for (var col = 0; col < Columns; col++) {
            for (var row = 0; row < Rows; row++) {
                yield return this[row, col];
            }
        }
    }

    /// <summary>
    /// Get an enumerator over the elements of the matrix
    /// </summary>
    /// <returns>enumerator</returns>
    public IEnumerator<T> GetEnumerator() => ((IEnumerable<T>)values).GetEnumerator();

    /// <summary>
    /// Get an enumerator over the elements of the matrix
    /// </summary>
    /// <returns>enumerator</returns>
    IEnumerator IEnumerable.GetEnumerator() => values.GetEnumerator();

    #endregion
}