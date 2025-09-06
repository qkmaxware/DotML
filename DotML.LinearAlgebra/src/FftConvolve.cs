using System.Numerics;

namespace DotML;

/// <summary>
/// Helper class to perform convolutions using fast fourier transforms
/// </summary>
public class FftConvolve {

    public static Matrix<T> Convolve2D<T>(
        Matrix<T> source,
        Matrix<T> kernel, 
        int strideX = 1, int strideY = 1, 
        int padTop = 0, int padRight = 0, int padBottom = 0, int padLeft = 0,
        T? bias = default(T)
    ) where T:INumber<T> {
        int outRows = (source.Rows + padTop + padBottom - kernel.Rows) / strideY + 1;
        int outCols = (source.Columns + padLeft + padRight - kernel.Columns) / strideX + 1;

        // Find size for FFT (next power of two for each dimension)
        int fftRows = 1;
        while (fftRows < source.Rows + padTop + padBottom || fftRows < kernel.Rows) fftRows <<= 1;
        int fftCols = 1;
        while (fftCols < source.Columns + padLeft + padRight || fftCols < kernel.Columns) fftCols <<= 1;

        // Pad source
        var srcPadded = new Complex[fftRows, fftCols];
        for (int i = 0; i < source.Rows; i++)
            for (int j = 0; j < source.Columns; j++)
                srcPadded[i + padTop, j + padLeft] = new Complex(Convert.ToDouble(source[i, j]), 0);

        // Pad and flip kernel
        var kerPadded = new Complex[fftRows, fftCols];
        for (int i = 0; i < kernel.Rows; i++)
            for (int j = 0; j < kernel.Columns; j++)
                kerPadded[i, j] = new Complex(Convert.ToDouble(kernel[kernel.Rows - 1 - i, kernel.Columns - 1 - j]), 0);

        // FFT both
        CooleyTukey.FFT2D(srcPadded, inverse: false);
        CooleyTukey.FFT2D(kerPadded, inverse: false);

        // Multiply in frequency domain
        int len = fftRows * fftCols;
        var srcSpan = System.Runtime.InteropServices.MemoryMarshal.CreateSpan(
            ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(srcPadded), len);
        var kerSpan = System.Runtime.InteropServices.MemoryMarshal.CreateSpan(
            ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(kerPadded), len);
        for (int idx = 0; idx < len; idx++)
            srcSpan[idx] *= kerSpan[idx];
        var freq = srcPadded;

        // Inverse FFT
        CooleyTukey.FFT2D(freq, inverse: true);

        // Extract result
        var result = new Matrix<T>(outRows, outCols);
        var bias_v = bias ?? T.Zero;
        for (int i = 0; i < outRows; i++)
        {
            for (int j = 0; j < outCols; j++)
            {
                int y = i * strideY;
                int x = j * strideX;
                double val = freq[y + kernel.Rows - 1, x + kernel.Columns - 1].Real / (fftRows * fftCols);
                result[i, j] = T.CreateChecked(val) + bias_v;
            }
        }
        return result;
    }
}