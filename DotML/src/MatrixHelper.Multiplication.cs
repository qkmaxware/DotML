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

internal static partial class MatrixHelper<T> where T:INumber<T>,IExponentialFunctions<T>,IRootFunctions<T> {

    // Based on the paper https://www.researchgate.net/publication/356268171_EFFICIENT_MATRIX_MULTIPLICATION_USING_HARDWARE_INTRINSICS_AND_PARALLELISM_WITH_C

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix<T> MultiplySIMDInParallelTranspose(Matrix<T> a, Matrix<T> b) {
        if (a.Columns != b.Rows)
            throw new ArithmeticException($"Incompatible dimensions for matrix multiplication {a.Rows}x{a.Columns} · {b.Rows}x{b.Columns}");
        
        var bt = b.Transpose();
        int rows = a.Rows;
        int cols = b.Columns;
        int innerDim = a.Columns;
        int vector_size = Vector<T>.Count;

        T[,] result = new T[rows, cols];
        
        Parallel.For(0, rows, i => {
            T[] va = new T[vector_size];
            T[] vb = new T[vector_size];
            for (int j = 0; j < cols; j++) {
                T sum = T.Zero;
                int k = 0;
                // Handle processing elements in chunks of vector_size
                for (; k < innerDim - vector_size; k+= vector_size) {
                    for (var w = 0; w < vector_size; w++) {
                        va[w] = a[i, k+w]; 
                        vb[w] = bt[j, k+w]; 
                    }

                    var product = Vector.Dot(new Vector<T>(va), new Vector<T>(vb));
                    sum += product;
                }

                // Handle any remaining elements not processed in chunks of vector_size
                for (; k < innerDim; k++) {
                    sum += a[i, k] * bt[j, k];
                }

                // Store the results
                result[i, j] = sum;
            }
        });

        return Matrix<T>.Wrap(result);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix<T> MultiplySIMDInParallel(Matrix<T> a, Matrix<T> b) {
        if (a.Columns != b.Rows)
            throw new ArithmeticException($"Incompatible dimensions for matrix multiplication {a.Rows}x{a.Columns} · {b.Rows}x{b.Columns}");
        
        int rows = a.Rows;
        int cols = b.Columns;
        int innerDim = a.Columns;
        int vector_size = Vector<T>.Count;

        T[,] result = new T[rows, cols];
        
        Parallel.For(0, rows, i => {
            T[] va = new T[vector_size];
            T[] vb = new T[vector_size];
            for (int j = 0; j < cols; j++) {
                T sum = T.Zero;
                int k = 0;
                // Handle processing elements in chunks of vector_size
                for (; k < innerDim - vector_size; k+= vector_size) {
                    for (var w = 0; w < vector_size; w++) {
                        va[w] = a[i, k+w]; 
                        vb[w] = b[k+w, j]; 
                    }

                    var product = Vector.Dot(new Vector<T>(va), new Vector<T>(vb));
                    sum += product;
                }

                // Handle any remaining elements not processed in chunks of vector_size
                for (; k < innerDim; k++) {
                    sum += a[i, k] * b[k, j];
                }

                // Store the results
                result[i, j] = sum;
            }
        });

        return Matrix<T>.Wrap(result);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix<T> MultiplySIMDTranspose(Matrix<T> a, Matrix<T> b) {
        if (a.Columns != b.Rows)
            throw new ArithmeticException($"Incompatible dimensions for matrix multiplication {a.Rows}x{a.Columns} · {b.Rows}x{b.Columns}");
        
        var bt = b.Transpose();
        int rows = a.Rows;
        int cols = b.Columns;
        int innerDim = a.Columns;
        int vector_size = Vector<T>.Count;

        T[,] result = new T[rows, cols];
        T[] va = new T[vector_size];
        T[] vb = new T[vector_size];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                T sum = T.Zero;
                int k = 0;
                // Handle processing elements in chunks of vector_size
                for (; k < innerDim; k+= vector_size) {
                    for (var w = 0; w < vector_size; w++) {
                        va[w] = a[i, k+w]; 
                        vb[w] = bt[j, k+w]; 
                    }

                    var product = Vector.Dot(new Vector<T>(va), new Vector<T>(vb));
                    sum += product;
                }

                // Handle any remaining elements not processed in chunks of vector_size
                for (; k < innerDim; k++) {
                    sum += a[i, k] * bt[j, k];
                }

                // Store the results
                result[i, j] = sum;
            }
        }

        return Matrix<T>.Wrap(result);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix<T> MultiplySIMD(Matrix<T> a, Matrix<T> b) {
        if (a.Columns != b.Rows)
            throw new ArithmeticException($"Incompatible dimensions for matrix multiplication {a.Rows}x{a.Columns} · {b.Rows}x{b.Columns}");
        
        int rows = a.Rows;
        int cols = b.Columns;
        int innerDim = a.Columns;
        int vector_size = Vector<T>.Count;

        T[,] result = new T[rows, cols];
        T[] va = new T[vector_size];
        T[] vb = new T[vector_size];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                T sum = T.Zero;
                int k = 0;
                // Handle processing elements in chunks of vector_size
                for (; k < innerDim; k+= vector_size) {
                    for (var w = 0; w < vector_size; w++) {
                        va[w] = a[i, k+w]; 
                        vb[w] = b[k+w, j]; 
                    }

                    var product = Vector.Dot(new Vector<T>(va), new Vector<T>(vb));
                    sum += product;
                }

                // Handle any remaining elements not processed in chunks of vector_size
                for (; k < innerDim; k++) {
                    sum += a[i, k] * b[k, j];
                }

                // Store the results
                result[i, j] = sum;
            }
        }

        return Matrix<T>.Wrap(result);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix<T> MultiplyInParallelTransposed(Matrix<T> a, Matrix<T> b) {
        // ~3x faster than MultiplyNaive when threads are available
        if (a.Columns != b.Rows)
            throw new ArithmeticException($"Incompatible dimensions for matrix multiplication {a.Rows}x{a.Columns} · {b.Rows}x{b.Columns}");

        var bt = b.Transpose(); // Better use of hardware caching but increases time complexity due to this transposition which is O(n^2)
        int rows = a.Rows;
        int cols = b.Columns;
        int innerDim = a.Columns;

        T[,] result = new T[rows, cols];
        Parallel.For(0, rows, i => {
            for (int j = 0; j < cols; j++) {
                T sum = T.Zero;            
                for (int k = 0; k < innerDim; k++) {
                    sum += a[i, k] * bt[j, k];
                }
                result[i, j] = sum;
            }
        });
        return Matrix<T>.Wrap(result);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix<T> MultiplyInParallel(Matrix<T> a, Matrix<T> b) {
        // ~3x faster than MultiplyNaive when threads are available
        if (a.Columns != b.Rows)
            throw new ArithmeticException($"Incompatible dimensions for matrix multiplication {a.Rows}x{a.Columns} · {b.Rows}x{b.Columns}");

        int rows = a.Rows;
        int cols = b.Columns;
        int innerDim = a.Columns;

        T[,] result = new T[rows, cols];
        Parallel.For(0, rows, i => {
            for (int j = 0; j < cols; j++) {
                T sum = T.Zero;            
                for (int k = 0; k < innerDim; k++) {
                    sum += a[i, k] * b[k, j];
                }
                result[i, j] = sum;
            }
        });
        return Matrix<T>.Wrap(result);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix<T> MultiplyNaive(Matrix<T> a, Matrix<T> b) {
        // Baseline matrix multiplication algorithm
        if (a.Columns != b.Rows)
            throw new ArithmeticException($"Incompatible dimensions for matrix multiplication {a.Rows}x{a.Columns} · {b.Rows}x{b.Columns}");

        int rows = a.Rows;
        int cols = b.Columns;
        int innerDim = a.Columns;

        T[,] result = new T[rows, cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                T sum = T.Zero;            
                for (int k = 0; k < innerDim; k++) {
                    sum += a[i, k] * b[k, j];
                }
                result[i, j] = sum;
            }
        }

        return Matrix<T>.Wrap(result);
    }
}