using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace DotML;

/// <summary>
/// Cooley-Tukey Fast Fourier Transform implementation
/// </summary>
public static class CooleyTukey {

    /// <summary>
    /// Performs an in-place 2D FFT on a matrix of complex numbers using the Cooley-Tukey algorithm.
    /// </summary>
    /// <param name="data">A 2D array of Complex numbers. The transform is performed in-place.</param>
    /// <param name="inverse">Set to true to perform the inverse FFT.</param>
    public static void FFT2D(Complex[,] data, bool inverse = false)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);

        // Temporary buffer for row/column operations to minimize allocations
        Span<Complex> buffer = stackalloc Complex[Math.Max(rows, cols)];

        // 1. FFT on rows
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
                buffer[c] = data[r, c];

            FFT1D(buffer, cols, inverse);

            for (int c = 0; c < cols; c++)
                data[r, c] = buffer[c];
        }

        // 2. FFT on columns
        for (int c = 0; c < cols; c++)
        {
            for (int r = 0; r < rows; r++)
                buffer[r] = data[r, c];

            FFT1D(buffer, rows, inverse);

            for (int r = 0; r < rows; r++)
                data[r, c] = buffer[r];
        }
    }

    /// <summary>
    /// Performs an in-place 1D FFT on a span of complex numbers using the Cooley-Tukey algorithm.
    /// </summary>
    /// <param name="data">A span of Complex numbers. The transform is performed in-place.</param>
    /// <param name="n">The number of elements in the span to transform. Must be a power of two.</param>
    /// <param name="inverse">Set to true to perform the inverse FFT.</param>
    private static void FFT1D(Span<Complex> data, int n, bool inverse)
    {
        // Bit reversal permutation
        int j = 0;
        for (int i = 0; i < n; i++)
        {
            if (i < j)
            {
                var temp = data[i];
                data[i] = data[j];
                data[j] = temp;
            }
            int m = n >> 1;
            while (m >= 1 && j >= m)
            {
                j -= m;
                m >>= 1;
            }
            j += m;
        }

        // Cooley-Tukey
        for (int len = 2; len <= n; len <<= 1)
        {
            double angle = 2 * Math.PI / len * (inverse ? 1 : -1);
            Complex wlen = new Complex(Math.Cos(angle), Math.Sin(angle));
            for (int i = 0; i < n; i += len)
            {
                Complex w = Complex.One;
                for (int j2 = 0; j2 < len / 2; j2++)
                {
                    Complex u = data[i + j2];
                    Complex v = data[i + j2 + len / 2] * w;
                    data[i + j2] = u + v;
                    data[i + j2 + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }

        // Normalize if inverse
        if (inverse)
        {
            double invN = 1.0 / n;
            for (int i = 0; i < n; i++)
                data[i] *= invN;
        }
    }

    /// <summary>
    /// Performs an in-place 2D FFT on a matrix of complex numbers using the Cooley-Tukey algorithm.
    /// </summary>
    /// <param name="data">A 2D array of ComplexF numbers. The transform is performed in-place.</param>
    /// <param name="inverse">Set to true to perform the inverse FFT.</param>
    public static void FFT2D(ComplexF[,] data, bool inverse = false)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);

        // Temporary buffer for row/column operations to minimize allocations
        Span<ComplexF> buffer = stackalloc ComplexF[Math.Max(rows, cols)];

        // 1. FFT on rows
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
                buffer[c] = data[r, c];

            FFT1D(buffer, cols, inverse);

            for (int c = 0; c < cols; c++)
                data[r, c] = buffer[c];
        }

        // 2. FFT on columns
        for (int c = 0; c < cols; c++)
        {
            for (int r = 0; r < rows; r++)
                buffer[r] = data[r, c];

            FFT1D(buffer, rows, inverse);

            for (int r = 0; r < rows; r++)
                data[r, c] = buffer[r];
        }
    }

    /// <summary>
    /// Performs an in-place 1D FFT on a span of complex numbers using the Cooley-Tukey algorithm.
    /// </summary>
    /// <param name="data">A span of ComplexF numbers. The transform is performed in-place.</param>
    /// <param name="n">The number of elements in the span to transform. Must be a power of two.</param>
    /// <param name="inverse">Set to true to perform the inverse FFT.</param>
    private static void FFT1D(Span<ComplexF> data, int n, bool inverse)
    {
        // Bit reversal permutation
        int j = 0;
        for (int i = 0; i < n; i++)
        {
            if (i < j)
            {
                var temp = data[i];
                data[i] = data[j];
                data[j] = temp;
            }
            int m = n >> 1;
            while (m >= 1 && j >= m)
            {
                j -= m;
                m >>= 1;
            }
            j += m;
        }

        // Cooley-Tukey
        for (int len = 2; len <= n; len <<= 1)
        {
            float angle = 2 * MathF.PI / len * (inverse ? 1 : -1);
            ComplexF wlen = new ComplexF(MathF.Cos(angle), MathF.Sin(angle));
            for (int i = 0; i < n; i += len)
            {
                ComplexF w = ComplexF.One;
                for (int j2 = 0; j2 < len / 2; j2++)
                {
                    ComplexF u = data[i + j2];
                    ComplexF v = data[i + j2 + len / 2] * w;
                    data[i + j2] = u + v;
                    data[i + j2 + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }

        // Normalize if inverse
        if (inverse)
        {
            float invN = 1.0f / n;
            for (int i = 0; i < n; i++)
                data[i] *= invN;
        }
    }

}