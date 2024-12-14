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

namespace DotML;

/// <summary>
/// Wrapper struct around value array providing matrix like functionality. Behaves like pass-by-reference rather than pass-by-value while minimizing heap allocations and dereferences. 
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct Matrix<T> 
: IEnumerable<T>
where T:INumber<T>,IExponentialFunctions<T>,IRootFunctions<T>
{
    private static T[,] NONE = new T[0,0];
    private T[,] values;

    /// <summary>
    /// Number of elements in the matrix
    /// </summary>
    [JsonIgnore] public int Size     => values.Length;
    /// <summary>
    /// Number of columns
    /// </summary>
    public int Columns  => values.GetLength(1);
    private int value_columns => values.GetLength(1); // Same as Columns, here in case I swap transposition to just be a boolean flag
    /// <summary>
    /// Number of rows
    /// </summary>
    public int Rows     => values.GetLength(0);
    private int value_rows => values.GetLength(0); // Same as Rows, here in case I swap transposition to just be a boolean flag
    /// <summary>
    /// Matrix shape (rows & columns)
    /// </summary>
    [JsonIgnore] public Shape2D Shape => new Shape2D(value_rows, value_columns);
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
    /// Create an empty 0x0 matrix
    /// </summary>
    public Matrix() {
        this.values = NONE;
    }

    /// <summary>
    /// Create a matrix from the given values
    /// </summary>
    /// <param name="values">values</param>
    public Matrix(T[,] values) {
        this.values = values;
    }

    /// <summary>
    /// Create a matrix from the given values
    /// </summary>
    /// <param name="values">values</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)] // Hoping this will allow the internal IFs to be optimized out at calling sites, tbh no idea if it will
    private Matrix(T[,] values, bool shared) {
        //this.values = values;
        if (!shared) {
            this.values = new T[values.GetLength(0),values.GetLength(1)];
            Array.Copy(values, this.values, values.Length);
        } else {
            this.values = values;
        }
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get {
            if (row >= 0 && row < value_rows && col >= 0 && col < value_columns)
                return values[row, col];
            else
                return T.Zero;
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
        Matrix<R> result = new Matrix<R>(Rows, Columns);

        for (var row = 0; row < value_rows; row++)
            for (var col = 0; col < value_columns; col++)
                result.values[row, col] = mapping(this.values[row, col]);
        
        return result;
    }

    /// <summary>
    /// Transform the values in this matrix to values of another type.
    /// </summary>
    /// <typeparam name="R">Result type</typeparam>
    /// <param name="mapping">Mapping function</param>
    /// <returns>New matrix, same size as the existing one but with elements modified by the mapping function</returns>
    public Matrix<R> Transform<R>(Func<(int Row, int Column), T, R> mapping) where R:INumber<R>,IExponentialFunctions<R>,IRootFunctions<R> {
        Matrix<R> result = new Matrix<R>(Rows, Columns);

        for (var row = 0; row < value_rows; row++)
            for (var col = 0; col < value_columns; col++)
                result.values[row, col] = mapping((row, col), this.values[row, col]);
        
        return result;
    }

    /// <summary>
    /// Enumerate over a flattened version of the matrix row-by-row.
    /// </summary>
    /// <returns>Matrix values as a single array</returns>
    public IEnumerable<T> FlattenRows() {
        for (var row = 0; row < value_rows; row++) {
            for (var col = 0; col < value_columns; col++) {
                yield return values[row, col];
            }
        }
    }

    /// <summary>
    /// Enumerate over a flattened version of the matrix column-by-column.
    /// </summary>
    /// <returns>Matrix values as a single array</returns>
    public IEnumerable<T> FlattenColumns() {
        for (var col = 0; col < value_columns; col++) {
            for (var row = 0; row < value_rows; row++) {
                yield return values[row, col];
            }
        }
    }

    /// <summary>
    /// Extract a given row from the matrix
    /// </summary>
    /// <param name="rowIndex">row index</param>
    /// <returns>vector representation of the row</returns>
    public Vec<T> ExtractRow(int rowIndex) {
        T[] vec = new T[this.Columns];
        for (var i = 0; i < this.Columns; i++) {
            vec[i] = this[rowIndex, i];
        }
        return Vec<T>.Wrap(vec);
    }

    /// <summary>
    /// Extract a given column from the matrix
    /// </summary>
    /// <param name="rowIndex">column index</param>
    /// <returns>vector representation of the column</returns>
    public Vec<T> ExtractColumn(int colIndex) {
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
    /// Wrap an existing rectangular array as a matrix without copying it's elements
    /// </summary>
    /// <param name="values">matrix elements</param>
    /// <returns>matrix</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix<T> Wrap(T[,] values) {
        return new Matrix<T>(values, shared: true);
    }

    /// <summary>
    /// Clone the values of the given rectangular array into the matrix copying each element
    /// </summary>
    /// <param name="values">matrix elements</param>
    /// <returns>matrix</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix<T> FromCopy(T[,] values) {
        return new Matrix<T>(values, shared: false);
    }

    // TODO below this point I need to clarify the difference between Rows/Columns and value_rows, value_columns

    /// <summary>
    /// Hadamard or element-wise multiplication of two matrices
    /// </summary>
    /// <param name="other">other matrix</param>
    /// <returns>element-wise product</returns>
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

        Matrix<T> result = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.values[i, j] = @operator(this[i, j], other[i, j]);

        return result;
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

        Matrix<R> result = new Matrix<R>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.values[i, j] = @operator(this[i, j], other[i, j]);

        return result;
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
        Matrix<T> res = new Matrix<T>(this.Rows, this.Columns);
        for (var r = 0; r < res.Rows; r++)
            for (var c = 0; c < res.Columns; c++)
                res.values[r, c] = this[r,c];
        return res;
    }

    /// <summary>
    /// Generate a matrix with the given values provided by a generator function
    /// </summary>
    /// <param name="rows">number of rows</param>
    /// <param name="cols">number of columns</param>
    /// <param name="generator">generator function</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Generate(int rows, int cols, Func<T> generator) {
        Matrix<T> result = new Matrix<T>(rows, cols);
        for (int i = 0; i < result.Rows; i++)
            for (int j = 0; j < result.Columns; j++)
                result.values[i, j] = generator();
        return result;
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
        Matrix<T> result = new Matrix<T>(args.Length, 1);
        for (var i = 0; i < args.Length; i++) {
            result.values[i, 0] = args[i];
        }
        return result;
    }

    /// <summary>
    /// Create a column matrix from the given values
    /// </summary>
    /// <param name="args">matrix elements</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Column(Vec<T> args) {
        Matrix<T> result = new Matrix<T>(args.Dimensionality, 1);
        for (var i = 0; i < args.Dimensionality; i++) {
            result.values[i, 0] = args[i];
        }
        return result;
    }

    /// <summary>
    /// Create a row matrix from the given values
    /// </summary>
    /// <param name="args">matrix elements</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Row(params T[] args) {
        Matrix<T> result = new Matrix<T>(1, args.Length);
        for (var i = 0; i < args.Length; i++) {
            result.values[0, i] = args[i];
        }
        return result;
    }

    /// <summary>
    /// Create a row matrix from the given values
    /// </summary>
    /// <param name="args">matrix elements</param>
    /// <returns>matrix</returns>
    public static Matrix<T> Row(Vec<T> args) {
        Matrix<T> result = new Matrix<T>(1, args.Dimensionality);
        for (var i = 0; i < args.Dimensionality; i++) {
            result.values[0, i] = args[i];
        }
        return result;
    }

    /// <summary>
    /// Multiply a matrix by a scalar value
    /// </summary>
    /// <param name="a">matrix</param>
    /// <param name="b">scalar</param>
    /// <returns>matrix</returns>
    public static Matrix<T> operator * (T a, Matrix<T> b) {
        int rows = b.Rows;
        int cols = b.Columns;

        Matrix<T> result = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.values[i, j] = a * b[i,j];

        return result;
    }

    /// <summary>
    /// Multiply a matrix by a scalar value
    /// </summary>
    /// <param name="a">matrix</param>
    /// <param name="b">scalar</param>
    /// <returns>matrix</returns>
    public static Matrix<T> operator * (Matrix<T> a, T b) {
        int rows = a.Rows;
        int cols = a.Columns;

        Matrix<T> result = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.values[i, j] = a[i,j] * b;

        return result;
    }

    /// <summary>
    /// Matrix matrix multiplication
    /// </summary>
    /// <param name="a">LHS matrix</param>
    /// <param name="b">RHS matrix</param>
    /// <returns>matrix</returns>
    /// <exception cref="ArithmeticException">Incompatible dimensions</exception>
    public static Matrix<T> operator * (Matrix<T> a, Matrix<T> b) {
        return MatrixHelper<T>.MultiplyInParallel(a, b); // I hate that this is the fastest. Seems like it works great in a single threaded app. Will have to see how it works in multi-threaded apps where threads are not as readily available.
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
        for (int i = 0; i < a.Rows; i++) {
            T value = T.Zero;
            for (int j = 0; j < a.Columns; j++) {
                value = value + a[i, j] * b[i];
            }
            result[i] = value;
        }

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

        Matrix<T> result = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.values[i, j] = a[i, j] + b[i, j];

        return result;
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

        Matrix<T> result = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.values[i, j] = a[i, j] - b[i, j];

        return result;
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
        
        for (var r = 0; r < rows; r++) {
            for (var c = 0; c < cols; c++) {
                result[r,c] =  mapping(src[r, c]);
            }
        }
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
        
        for (var r = 0; r < rows; r++) {
            for (var c = 0; c < cols; c++) {
                result[r,c] =  mapping((r, c), src[r, c]);
            }
        }
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

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = a[i, j] * b[i, j];
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

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = a[i, j] + b[i, j];
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

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = a[i, j] - b[i, j];
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

    public IEnumerator<T> GetEnumerator() {
        for (var row = 0; row < this.Rows; row++) {
            for (var col = 0; col < this.Columns; col++)
                yield return this[row, col];
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => this.GetEnumerator();
}