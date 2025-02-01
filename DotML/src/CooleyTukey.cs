using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using DotML;

/// <summary>
/// Cooly-Tukey Fast Fourier Transform implementation
/// </summary>
public static class CooleyTukey {

    private static void BitReverse(Span<Complex> x) {
        var N = x.Length;
        int j = 0;
        for (int i = 0; i < N; i++) {
            if (j > i) {
                var temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
            int m = N >> 1;
            while (m >= 1 && j >= m) {
                j -= m;
                m >>= 1;
            }
            j += m;
        }
    }

    /// <summary>
    /// Perform a fast fourier transform
    /// </summary>
    /// <param name="x">Series</param>
    public static void FFT(Span<Complex> x) {
        var N = x.Length;
        if (N <= 1) return;

        BitReverse(x);

        for (int len = 2; len <= N; len <<= 1) {
            double angle = -2 * Math.PI / len;
            var len_2 = len/2;
            var wlen = Complex.FromPolarCoordinates(1.0, angle);
            
            for (int i = 0; i < N; i += len) {
                var w = Complex.One;
                for (int j = 0; j < len_2; j++) {
                    var ij = i + j;
                    var u = x[ij];
                    var t = w * x[ij + len_2];
                    
                    x[ij] = u + t;
                    x[ij + len_2] = u - t;
                    
                    w *= wlen;
                }
            }
        }
    }

    /// <summary>
    /// Perform an inverse fast fourier transform
    /// </summary>
    /// <param name="x">Series</param>
    public static void iFFT(Span<Complex> x) {
        var N = x.Length;
        if (N <= 1)
            return;

        // Conjugate the complex numbers
        for (var i = 0; i < N; i++) {
            x[i] = Complex.Conjugate(x[i]);
        }

        // Forward FFT
        FFT(x);

        // Conjugate again & Rescale
        for (var i = 0; i < N; i++) {
            x[i] = Complex.Conjugate(x[i]) / N;
        }
    }

    private static int GetNextPowerOfTwo(int n) {
        return (int)Math.Pow(2, Math.Ceiling(Math.Log(n) / Math.Log(2)));
    }

    public static Matrix<double> ConvolveEachFFT(IEnumerable<Matrix<double>> inputs, IEnumerable<Matrix<double>> kernels, int strideX = 1, int strideY = 1, int paddingX = 0, int paddingY = 0, double bias = 0.0) {
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

        var result              = new Matrix<double>(outputRows, outputColumns, bias);
        
        // This is a sum of each input with each kernel
        foreach (var pair in inputs.Zip(kernels)) {
            var input = pair.First;
            var kernel = pair.Second;

            var convolution = ConvolveFFT(input, kernel, strideX, strideY, paddingX, paddingY);
            result.AddWithInplace(convolution);
        }

        return result;
    }

    public static Matrix<double> ConvolveFFT(Matrix<double> input, Matrix<double> kernel, int strideX = 1, int strideY = 1, int paddingX = 0, int paddingY = 0) {
        var filterRows          = kernel.Rows;   
        var filterColumns       = kernel.Columns;  
        var paddingRows         = paddingY;
        var paddingColumns      = paddingX;  

        var inputRows           = input.Rows;
        var inputColumns        = input.Columns;

        // Same math as in ConvolutionLayer.cs for output shape
        var outputRows          = (inputRows - filterRows + 2 * paddingRows) / strideY + 1; 
        var outputColumns       = (inputColumns - filterColumns + 2 * paddingColumns) / strideX + 1;  

        var result              = new Matrix<double>(outputRows, outputColumns, 0.0);

        // Different from this point onwards
        // 1. Pad matrices with 0s to nearest power of 2
        int desired_width = input.Columns + 2 * paddingX;
        int padded_width = GetNextPowerOfTwo(desired_width);
        var offset_x = (padded_width - desired_width) >> 1;
        var output_offset_x = offset_x + ((desired_width - outputColumns) >> 1);

        int desired_height = input.Rows + 2 * paddingY;
        int padded_height = GetNextPowerOfTwo(desired_height);
        var offset_y = (padded_height - desired_height) >> 1;
        var output_offset_y = offset_y + ((padded_height - outputRows) >> 1);

        var padded_length = padded_width * padded_height;

        Complex[,] padded_input = new Complex[padded_height, padded_width];
        var padded_input_span = MemoryMarshal.CreateSpan(ref Unsafe.As<byte, Complex>(ref MemoryMarshal.GetArrayDataReference(padded_input)), padded_input.Length);
        Complex[,] padded_kernel = new Complex[padded_height, padded_width];
        var padded_kernel_span = MemoryMarshal.CreateSpan(ref Unsafe.As<byte, Complex>(ref MemoryMarshal.GetArrayDataReference(padded_kernel)), padded_kernel.Length);
        // TODO copy values from arrays into the padded ones
        for (var row = 0; row < inputRows; row++) {
            for (var col = 0; col < inputColumns; col++) {
                padded_input[offset_x + row, offset_y + col] = input[row, col];
            }
        }
        for (var row = 0; row < filterRows; row++) {
            for (var col = 0; col < filterColumns; col++) {
                padded_kernel[offset_x + row,offset_x + col] = kernel[row, col];
            }
        }

        // 2. Call FFT
        CooleyTukey.FFT(padded_input_span);
        CooleyTukey.FFT(padded_kernel_span);

        // 3. Multiply in the frequency domain (element-wise)
        for (var i = 0; i < padded_length; i++) {
            padded_input_span[i] = padded_input_span[i] * padded_kernel_span[i];
        }

        // 4. Inverse (results can just be stored in the padded-input span)
        CooleyTukey.iFFT(padded_input_span);

        // 5. Extract the relevant portion based on stride and padding
        for (var row = 0; row < outputRows; row++) {
            var region_row_index = output_offset_y + row * strideY; // Grab every Y element
            if (region_row_index >= padded_height)
                continue;

            for (var col = 0; col < outputColumns; col++) {
                var region_col_index = output_offset_x + col * strideX; // Grab every X element
                if (region_col_index >= padded_width)
                    continue;

                result[row, col] = padded_input[region_row_index, region_col_index].Real;
            }
        }

        return result;
    }
}