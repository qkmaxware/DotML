using System.Collections;
using System.Data;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Text.Json.Serialization;
using DotML.Network;

namespace DotML;

/// <summary>
/// Wrapper struct around value array providing matrix like functionality. Behaves like pass-by-reference rather than pass-by-value while minimizing heap allocations and dereferences. 
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct Matrix<T> 
: IEnumerable<T>, ITensorLike<T>, IHtmlable
where T:INumber<T>,IExponentialFunctions<T>,IRootFunctions<T>
{
    private static T[,] NONE = new T[0,0];
    private T[,] values;

    /// <summary>
    /// Number of elements in the matrix
    /// </summary>
    [JsonIgnore] public int Size {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => values.Length;
    }
    /// <summary>
    /// Number of columns
    /// </summary>
    public int Columns {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => values.GetLength(1);
    }
    /// <summary>
    /// Number of rows
    /// </summary>
    public int Rows {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => values.GetLength(0);
    }
    /// <summary>
    /// Matrix shape (rows & columns)
    /// </summary>
    [JsonIgnore] public Shape2D Shape => new Shape2D(Rows, Columns);
    /// <summary>
    /// Check if the matrix is a column matrix (only one column)
    /// </summary> 
    [JsonIgnore] public bool IsColumn => this.Columns == 1;
    /// <summary>
    /// Check if the matrix is a row matrix (only one row)
    /// </summary> 
    [JsonIgnore] public bool IsRow => this.Rows == 1;
    /// <summary>
    /// Check if the matrix is square
    /// </summary> 
    [JsonIgnore] public bool IsSquare => this.Rows == this.Columns;

    /// <summary>
    /// Number of dimensions of this tensor
    /// </summary>
    [JsonIgnore] public int Dimensions => 2;

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
    /// <param name="row">Row index</param>
    /// <param name="col">Column index</param>
    /// <returns>value at row, column or zero if out of bounds</returns>
    public T this[int row, int col] {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => values[row, col];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set {
            if (row >= 0 && row < Rows && col >= 0 && col < Columns)
                values[row, col] = value;
        }
    }
    
    /// <summary>
    /// Get the value of a matrix element by a sequential index
    /// </summary>
    /// <param name="index">Sequential index</param>
    /// <returns>value at row, column or zero if out of bounds</returns>
    public T this[int index] { 
        [MethodImpl(MethodImplOptions.AggressiveInlining)] 
        get {
            var row = index / Columns;
            var col = index % Columns;
            var val = this[row, col];
            return val;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set {
            var row = index / Columns;
            var col = index % Columns;
            this[row, col] = value;
        }
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

            T[,] self = this.AsArray();
            T[,] values = new T[rows, cols];
            for (var row = 0; row < rows; row++) {
                var self_row = row_offset + row;
                if (self_row < 0 || self_row >= self_rows)
                    continue;

                for (var col = 0; col < cols; col++) {
                    var self_col = col_offset + col;
                    if (self_col < 0 || self_col >= self_cols)
                        continue;

                    values[row, col] = self[self_row, self_col];
                }
            }
            return Matrix<T>.Wrap(values);
        }
    }

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
    /// Create an empty 0x0 matrix
    /// </summary>
    public Matrix() {
        this.values = NONE;
    }

    /// <summary>
    /// Create a matrix with the given shape
    /// </summary>
    public Matrix(Shape2D shape) : this(shape.Rows, shape.Columns) {}

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
        var values = new T[rows, columns];
        this.values = values;
        var span = MemoryMarshal.CreateSpan(ref Unsafe.As<byte, T>(ref MemoryMarshal.GetArrayDataReference(values)), values.Length);
        span.Fill(value);
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
    /// Generate a matrix with the given values provided by a generator function
    /// </summary>
    /// <param name="rows">number of rows</param>
    /// <param name="cols">number of columns</param>
    /// <param name="generator">generator function</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Generate(int rows, int cols, Func<T> generator) {
        T[,] values = new T[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                values[i, j] = generator();
        return Matrix<T>.Wrap(values);
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
    /// Transform the values in this matrix to values of another type.
    /// </summary>
    /// <typeparam name="R">Result type</typeparam>
    /// <param name="mapping">Mapping function</param>
    /// <returns>New matrix, same size as the existing one but with elements modified by the mapping function</returns>
    public Matrix<R> Transform<R>(Func<T, R> mapping) where R:INumber<R>,IExponentialFunctions<R>,IRootFunctions<R> {
        var self = this;
        R[,] values = new R[Rows, Columns];
        ParallelUtils.ForEach(self, (job) => {
            values[job.Row, job.Column] = mapping(self.values[job.Row, job.Column]);
        });
        return Matrix<R>.Wrap(values);
    }

    /// <summary>
    /// Transform the values in this matrix to values of another type.
    /// </summary>
    /// <typeparam name="R">Result type</typeparam>
    /// <param name="mapping">Mapping function</param>
    /// <returns>New matrix, same size as the existing one but with elements modified by the mapping function</returns>
    public Matrix<R> Transform<R>(Func<(int Row, int Column), T, R> mapping) where R:INumber<R>,IExponentialFunctions<R>,IRootFunctions<R> {
        var self = this;
        R[,] values = new R[Rows, Columns];
        ParallelUtils.ForEach(self, (job) => {
            values[job.Row, job.Column] = mapping((job.Row, job.Column), self.values[job.Row, job.Column]);
        });
        return Matrix<R>.Wrap(values);
    }

    /// <summary>
    /// Enumerate over a flattened version of the matrix row-by-row.
    /// </summary>
    /// <returns>Matrix values as a single array</returns>
    public IEnumerable<T> FlattenRows() {
        for (var row = 0; row < Rows; row++) {
            for (var col = 0; col < Columns; col++) {
                yield return values[row, col];
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
                yield return values[row, col];
            }
        }
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

    /// <summary>
    /// Extract a given row from the matrix as a Span
    /// </summary>
    /// <param name="rowIndex">row index</param>
    /// <returns>span over the row</returns>
    public Span<T> ExtractRowSpan(int rowIndex) {
        return MemoryMarshal.CreateSpan(ref this.values[rowIndex, 0], this.Columns);
    }

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
        T[,] transposed = new T[cols,rows];
        for (var r = 0; r < rows; r++)
            for (var c = 0; c < cols; c++)
                transposed[c, r] = this[r, c];
        return Matrix<T>.Wrap(transposed);
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
    public Matrix<T> Convolve(Matrix<T> kernel, int strideX = 1, int strideY = 1, int paddingX = 0, int paddingY = 0) {
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

        var result              = new Matrix<T>(outputRows, outputColumns, T.Zero);
        var result_array        = result.AsArray();
        ParallelUtils.ForEach(result, (job) => {
            var startY = job.Y * strideY - paddingRows;
            var startX = job.X * strideX - paddingColumns;

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

            result_array[job.Row, job.Column] += total_sum;
        });

        return result;
    }
    /// <summary>
    /// Perform a convolution of this matrix by performing a convolution and summation with all provided kernels
    /// </summary>
    /// <param name="kernels">kernels</param>
    /// <param name="strideX">stride across the x-axis (columns)</param>
    /// <param name="strideY">stride across the y-axis (rows)</param>
    /// <param name="paddingX">horizontal padding of this matrix</param>
    /// <param name="paddingY">vertical padding of this matrix</param>
    /// <returns>convolution of this matrix with all kernels</returns>
    public Matrix<T> ConvolveAll(IEnumerable<Matrix<T>> kernels, int strideX = 1, int strideY = 1, int paddingX = 0, int paddingY = 0) {
        var filterRows          = kernels.First().Rows;   
        var filterColumns       = kernels.First().Columns;  
        var paddingRows         = paddingY;
        var paddingColumns      = paddingX;  

        var input               = this;
        var inputRows           = input.Rows;
        var inputColumns        = input.Columns;

        // Same math as in ConvolutionLayer.cs for output shape
        var outputRows          = (inputRows - filterRows + 2 * paddingRows) / strideY + 1; 
        var outputColumns       = (inputColumns - filterColumns + 2 * paddingColumns) / strideX + 1;  

        var result              = new Matrix<T>(outputRows, outputColumns, T.Zero);
        var result_array        = result.AsArray();
        foreach (var kernel in kernels) {
            ParallelUtils.ForEach(result, (job) => {
                var startY = job.Y * strideY - paddingRows;
                var startX = job.X * strideX - paddingColumns;

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

                result_array[job.Row, job.Column] += total_sum;
            });
        }

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
    public static Matrix<T> ConvolveAll(IEnumerable<Matrix<T>> inputs, IEnumerable<Matrix<T>> kernels, int strideX = 1, int strideY = 1, int paddingX = 0, int paddingY = 0) {
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

        var result              = new Matrix<T>(outputRows, outputColumns, T.Zero);
        var result_array        = result.AsArray();
        foreach (var (input, kernel) in inputs.Zip(kernels)) {
            ParallelUtils.ForEach(result, (job) => {
                var startY = job.Y * strideY - paddingRows;
                var startX = job.X * strideX - paddingColumns;

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

                result_array[job.Row, job.Column] += total_sum;
            });
        }

        return result;
    }

    /// <summary>
    /// Wrap an existing rectangular array as a matrix without copying it's elements
    /// </summary>
    /// <param name="values">matrix elements</param>
    /// <returns>matrix</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix<T> Wrap(T[,] values) {
        return new Matrix<T>(values);
    }

    /// <summary>
    /// Clone the values of the given rectangular array into the matrix copying each element
    /// </summary>
    /// <param name="values">matrix elements</param>
    /// <returns>matrix</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix<T> FromCopy(T[,] values) {
        var matrix = new T[values.GetLength(0),values.GetLength(1)];
        Array.Copy(values, matrix, values.Length);
        return new Matrix<T>(matrix);
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
        ParallelUtils.ForX(0, this.Columns, (job) => {
            T aggregate = initial;
            for (var row = 0; row < rows; row++) {
                aggregate = aggregator(aggregate, src[row, job.X]);
            }
            result[job.X] = aggregate;
        });
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
        ParallelUtils.ForX(0, this.Columns, (job) => {
            T aggregate = initial;
            for (var col = 0; col < columns; col++) {
                aggregate = aggregator(aggregate, src[job.X, col]);
            }
            result[job.X] = aggregate;
        });
        return Vec<T>.Wrap(result);
    }

    /// <summary>
    /// Hadamard or element-wise multiplication of two matrices
    /// </summary>
    /// <param name="other">other matrix</param>
    /// <returns>element-wise product</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Matrix<T> Hadamard(Matrix<T> other) => ElementWise(other, (a, b) => a * b);

    /// <summary>
    /// Perform an element-wise operation between two matrices
    /// </summary>
    /// <example>
    /// <code>
    /// var result = As.ElementWise(Bs, (a,b) => a + b);
    /// </code>
    /// </example>
    /// <param name="other">second matrix</param>
    /// <param name="operator">element-wise operation</param>
    /// <returns>matrix</returns>
    /// <exception cref="ArithmeticException">Matrix dimensions must match</exception>
    public Matrix<T> ElementWise(Matrix<T> other, Func<T, T, T> @operator) {
        if (this.Columns != other.Columns || this.Rows != other.Rows)
            throw new ArithmeticException("Incompatible dimensions for element-wise operations");

        int rows = Rows;
        int cols = Columns;

        var self = this;
        T[,] values = new T[rows, cols];
        ParallelUtils.ForEach(this, (job) => {
            values[job.Row, job.Column] = @operator(self[job.Row, job.Column], other[job.Row, job.Column]);
        });
        return Matrix<T>.Wrap(values);
    }

    /// <summary>
    /// Perform an element-wise operation between two matrices
    /// </summary>
    /// <example>
    /// <code>
    /// var result = As.ElementWise<double>(Bs, (a,b) => (double)(a + b));
    /// </code>
    /// </example>
    /// <param name="other">second matrix</param>
    /// <param name="operator">element-wise operation</param>
    /// <returns>matrix</returns>
    /// <exception cref="ArithmeticException">Matrix dimensions must match</exception>
    public Matrix<R> ElementWise<R>(Matrix<T> other, Func<T, T, R> @operator) where R:INumber<R>,IExponentialFunctions<R>,IRootFunctions<R> {
        if (this.Columns != other.Columns || this.Rows != other.Rows)
            throw new ArithmeticException("Incompatible dimensions for element-wise operations");

        int rows = Rows;
        int cols = Columns;

        var self = this;
        R[,] values = new R[rows, cols];
        ParallelUtils.ForEach(this, (job) => {
            values[job.Row, job.Column] = @operator(self[job.Row, job.Column], other[job.Row, job.Column]);
        });
        return Matrix<R>.Wrap(values);
    }   

    /// <summary>
    /// Reshape the elements of this matrix into one or more matrices of a different shape.
    /// </summary>
    /// <param name="shapes">list of shapes</param>
    /// <returns>matrices</returns>
    public IEnumerable<Matrix<T>> Reshape(params Shape2D[] shapes) {
        var index = 0;

        foreach (var shape in shapes) {
            var mtx = new Matrix<T>(shape.Rows, shape.Columns);
            for (int row = 0; row < mtx.Rows; row++) {
                for (int col = 0; col < mtx.Columns; col++) {
                    if (index < Size)
                        mtx.values[row, col] = this[index++];
                    else 
                        mtx.values[row, col] = T.Zero;
                }
            }
            yield return mtx;
        }
    }

    /// <summary>
    /// Reshape the elements of this matrix into one or more matrices of a different shape.
    /// </summary>
    /// <param name="size">shape of the matrix</param>
    /// <param name="channels">number of matrices</param>
    /// <returns>matrices</returns>
    public IEnumerable<Matrix<T>> Reshape(Shape2D size, int channels) {       
        return Reshape(Enumerable.Repeat(0, channels).Select(x => size).ToArray());
    }   

    /// <summary>
    /// Deep clone a matrix
    /// </summary>
    /// <returns></returns>
    public Matrix<T> Clone() {
        T[,] values = new T[Rows, Columns];
        for (var r = 0; r < Rows; r++)
            for (var c = 0; c < Columns; c++)
                values[r, c] = this[r,c];
        return Matrix<T>.Wrap(values);
    }

    /// <summary>
    /// Create a matrix whose values are the average of all matrices. Matrices should be the same size.
    /// </summary>
    /// <param name="matrices">list of matrices</param>
    /// <returns>matrix with averaged values</returns>
    public static Matrix<T> Average(IEnumerable<Matrix<T>> matrices) {
        T[,]? values = null;

        // Sum across all elements
        int rows = 0;
        int columns = 0;
        T count = T.Zero;
        foreach (var matrix in matrices) {
            if (values == null) {
                rows = matrix.Rows;
                columns =  matrix.Columns;
                values = new T[rows, columns];
            }

            for (var r = 0; r < rows; r++) {
                for (var c = 0; c < columns; c++) {
                    values[r, c] += matrix[r, c];
                }
            }

            count = count + T.One;
        }

        if (values is null || count == T.Zero) {
            throw new DivideByZeroException();
        }

        // Divide by count to average it
         for (var r = 0; r < rows; r++) {
            for (var c = 0; c < columns; c++) {
                values[r, c] = values[r, c] / count;
            }
        }

        return Matrix<T>.Wrap(values);
    }

    /// <summary>
    /// Create a column matrix from the given values
    /// </summary>
    /// <param name="args">matrix elements</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Column(params T[] args) {
        T[,] values = new T[args.Length, 1];
        for (var i = 0; i < args.Length; i++) {
            values[i, 0] = args[i];
        }
        return Matrix<T>.Wrap(values);
    }

    /// <summary>
    /// Create a column matrix from the given values
    /// </summary>
    /// <param name="args">matrix elements</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Column(Vec<T> args) {
        T[,] values = new T[args.Dimensionality, 1];
        for (var i = 0; i < args.Dimensionality; i++) {
            values[i, 0] = args[i];
        }
        return Matrix<T>.Wrap(values);
    }

    /// <summary>
    /// Create a row matrix from the given values
    /// </summary>
    /// <param name="args">matrix elements</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Row(params T[] args) {
        T[,] values = new T[1, args.Length];
        for (var i = 0; i < args.Length; i++) {
            values[0, i] = args[i];
        }
        return Matrix<T>.Wrap(values);
    }

    /// <summary>
    /// Create a row matrix from the given values
    /// </summary>
    /// <param name="args">matrix elements</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Row(Vec<T> args) {
        T[,] values = new T[1, args.Dimensionality];
        for (var i = 0; i < args.Dimensionality; i++) {
            values[0, i] = args[i];
        }
        return Matrix<T>.Wrap(values);
    }

    /// <summary>
    /// Multiply a matrix by a scalar value
    /// </summary>
    /// <param name="a">matrix</param>
    /// <param name="b">scalar</param>
    /// <returns>matrix</returns>
    public static Matrix<T> operator * (T a, Matrix<T> b) => b.ScaleBy(a);

    /// <summary>
    /// Multiply a matrix by a scalar value
    /// </summary>
    /// <param name="a">matrix</param>
    /// <param name="b">scalar</param>
    /// <returns>matrix</returns>
    public static Matrix<T> operator * (Matrix<T> a, T b) => a.ScaleBy(b);

    /// <summary>
    /// Multiply a matrix by a scalar value
    /// </summary>
    /// <param name="scale">scalar</param>
    /// <returns>matrix</returns>
    public Matrix<T> ScaleBy(T scale) {
        int rows = this.Rows;
        int cols = this.Columns;

        T[,] values = new T[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                values[i, j] = this[i,j] * scale;

        return Matrix<T>.Wrap(values);
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

        T[,] result = new T[rows, cols];
        ParallelUtils.ForX(0, rows, i => {
            for (int j = 0; j < cols; j++) {
                T sum = T.Zero;            
                for (int k = 0; k < innerDim; k++) {
                    sum += a[i.X, k] * b[k, j];
                }
                result[i.X, j] = sum;
            }
        });
        return Matrix<T>.Wrap(result);
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

        T[,] result = new T[rows, cols];
        ParallelUtils.ForX(0, rows, i => {
            for (int j = 0; j < cols; j++) {
                T sum = T.Zero;            
                for (int k = 0; k < innerDim; k++) {
                    sum += a[k, i.X] * b[k, j];
                }
                result[i.X, j] = sum;
            }
        });
        return Matrix<T>.Wrap(result);
    }

    /// <summary>
    /// Matrix vector multiplication
    /// </summary>
    /// <param name="a">LHS matrix</param>
    /// <param name="b">RHS vector</param>
    /// <returns>vector</returns>
    /// <exception cref="ArithmeticException">Incompatible dimensions</exception>
    public static Vec<T> operator * (Matrix<T> a, Vec<T> b) {
        if (a.Rows != b.Dimensionality)
            throw new ArithmeticException($"Incompatible dimensions for matrix/vector multiplication {a.Rows}x{a.Columns} · {b.Dimensionality}x1");

        T[] result = new T[a.Rows];
        ParallelUtils.ForX(0, a.Rows, (job) => {
            T value = T.Zero;
            for (int j = 0; j < a.Columns; j++) {
                value = value + a[job.X, j] * b[j];
            }
            result[job.X] = value;
        });
        
        return Vec<T>.Wrap(result);
    }

    /// <summary>
    /// Matrix matrix addition
    /// </summary>
    /// <param name="a">LHS matrix</param>
    /// <param name="b">RHS matrix</param>
    /// <returns>matrix</returns>
    /// <exception cref="ArithmeticException">Incompatible dimensions</exception>
    public static Matrix<T> operator + (Matrix<T> a, Matrix<T> b) {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            throw new ArithmeticException("Incompatible dimensions for matrix addition");

        int rows = a.Rows;
        int cols = a.Columns;

        T[,] values = new T[rows, cols];
        ParallelUtils.ForXY(0, cols, 0, rows, (job) => {
            values[job.Row, job.Column] = a[job.Row, job.Column] + b[job.Row, job.Column];
        });

        return Matrix<T>.Wrap(values);
    }

    /// <summary>
    /// Matrix matrix subtraction
    /// </summary>
    /// <param name="a">LHS matrix</param>
    /// <param name="b">RHS matrix</param>
    /// <returns>matrix</returns>
    /// <exception cref="ArithmeticException">Incompatible dimensions</exception>
    public static Matrix<T> operator - (Matrix<T> a, Matrix<T> b) {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            throw new ArithmeticException("Incompatible dimensions for matrix subtraction");

        int rows = a.Rows;
        int cols = a.Columns;

        T[,] values = new T[rows, cols];
        ParallelUtils.ForXY(0, cols, 0, rows, (job) => {
            values[job.Row, job.Column] = a[job.Row, job.Column] - b[job.Row, job.Column];
        });

        return Matrix<T>.Wrap(values);
    }

    #region In-place Operations
    /// <summary>
    /// Transform a matrix using an existing matrix as in-place storage without allocating new memory
    /// </summary>
    /// <typeparam name="R">result type</typeparam>
    /// <param name="target">matrix to store results</param>
    /// <param name="src">matrix with original values</param>
    /// <param name="mapping">mapping function</param>
    /// <exception cref="ArithmeticException">thrown if matrices are of incompatible dimensions</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void TransformInplace<R>(Matrix<R> target, Matrix<T> src, Func<T, R> mapping) where R:INumber<R>,IExponentialFunctions<R>,IRootFunctions<R> {
        var result = (R[,])target;
        var rows = target.Rows;
        var cols = target.Columns;
        if (rows != src.Rows || cols != src.Columns) {
            throw new ArithmeticException("Incompatible dimensions for storing matrix transformation result");
        }
        
        ParallelUtils.ForEach(target, (job) => {
            result[job.Row, job.Column] =  mapping(src[job.Row, job.Column]);
        });
    } 
    /// <summary>
    /// Transform a matrix using an existing matrix as in-place storage without allocating new memory
    /// </summary>
    /// <typeparam name="R">result type</typeparam>
    /// <param name="target">matrix to store results</param>
    /// <param name="src">matrix with original values</param>
    /// <param name="mapping">mapping function</param>
    /// <exception cref="ArithmeticException">thrown if matrices are of incompatible dimensions</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void TransformInplace<R>(Matrix<R> target, Matrix<T> src, Func<(int Row, int Column), T, R> mapping) where R:INumber<R>,IExponentialFunctions<R>,IRootFunctions<R> {
        var result = (R[,])target;
        var rows = target.Rows;
        var cols = target.Columns;
        if (rows != src.Rows || cols != src.Columns) {
            throw new ArithmeticException("Incompatible dimensions for storing matrix transformation result");
        }
        
        ParallelUtils.ForEach(target, (job) => {
            result[job.Row, job.Column] =  mapping((job.Row, job.Column), src[job.Row, job.Column]);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void HadamardInplace(Matrix<T> target, Matrix<T> a, Matrix<T> b) {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            throw new ArithmeticException("Incompatible dimensions for element-wise multiplication");

        int rows = a.Rows;
        int cols = a.Columns;
        var result = (T[,])target;

        if (target.Rows != rows || target.Columns != cols) {
            throw new ArithmeticException("Incompatible dimensions for storing element-wise multiplication result");
        }

        ParallelUtils.ForEach(target, (job) => {
            result[job.Row, job.Column] = a[job.Row, job.Column] * b[job.Row, job.Column];
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ElementWiseInplace(Matrix<T> target, Matrix<T> a, Matrix<T> b, Func<T, T, T> mapping) {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            throw new ArithmeticException("Incompatible dimensions for element-wise multiplication");

        int rows = a.Rows;
        int cols = a.Columns;
        var result = (T[,])target;

        if (target.Rows != rows || target.Columns != cols) {
            throw new ArithmeticException("Incompatible dimensions for storing element-wise multiplication result");
        }

        ParallelUtils.ForEach(target, (job) => {
            result[job.Row, job.Column] =  mapping(a[job.Row, job.Column], b[job.Row, job.Column]);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddInplace(Matrix<T> target, Matrix<T> a, Matrix<T> b) {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            throw new ArithmeticException("Incompatible dimensions for matrix addition");

        int rows = a.Rows;
        int cols = a.Columns;
        var result = (T[,])target;

        if (target.Rows != rows || target.Columns != cols) {
            throw new ArithmeticException("Incompatible dimensions for storing matrix addition result");
        }

        ParallelUtils.ForEach(target, (job) => {
            result[job.Row, job.Column] =  a[job.Row, job.Column] + b[job.Row, job.Column];
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SubInplace(Matrix<T> target, Matrix<T> a, Matrix<T> b) {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            throw new ArithmeticException("Incompatible dimensions for matrix subtraction");

        int rows = a.Rows;
        int cols = a.Columns;
        var result = (T[,])target;

        if (target.Rows != rows || target.Columns != cols) {
            throw new ArithmeticException("Incompatible dimensions for storing matrix subtraction result");
        }

        ParallelUtils.ForEach(target, (job) => {
            result[job.Row, job.Column] =  a[job.Row, job.Column] - b[job.Row, job.Column];
        });
    }
    #endregion

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
    public string ToHtml() {
        StringBuilder sb = new StringBuilder();
        sb.Append("<math><mrow>");
            sb.Append("<mo>[</mo>");
            sb.Append("<mtable>");
            for (int i = 0; i < Rows; i++) {
                sb.Append("<mtr>");
                for (int j = 0; j < Columns; j++) {
                    sb.Append("<mtd>");
                    sb.Append("<mn>"); sb.Append(this[i, j]); sb.Append("</mn>");
                    sb.Append("</mtd>");
                }
                sb.Append("</mtr>");
            }   
            sb.Append("</mtable>");
            sb.Append("<mo>]</mo>");
        sb.Append("</mrow></math>");
        return sb.ToString();
    }

    /// <summary>
    /// Create a span over the entire 2D matrix
    /// </summary>
    /// <returns>span over the matrix elements indexed in row-major order</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> AsSpan() {
        return MemoryMarshal.CreateSpan(ref Unsafe.As<byte, T>(ref MemoryMarshal.GetArrayDataReference(values)), values.Length);
    }

    /// <summary>
    /// Create a read-only span over the entire 2D matrix
    /// </summary>
    /// <returns>span over the matrix elements indexed in row-major order</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> AsReadOnlySpan() {
        return MemoryMarshal.CreateSpan(ref Unsafe.As<byte, T>(ref MemoryMarshal.GetArrayDataReference(values)), values.Length);
    }

    /// <summary>
    /// Get the underlying array of values
    /// </summary>
    /// <returns>underlying array of elements indexed in row-major order</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T[,] AsArray() {
        return this.values;
    }

    public IEnumerator<T> GetEnumerator() {
        for (var row = 0; row < this.Rows; row++) {
            for (var col = 0; col < this.Columns; col++)
                yield return this[row, col];
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => this.GetEnumerator();
}