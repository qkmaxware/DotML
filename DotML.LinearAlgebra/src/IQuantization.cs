namespace DotML;

/// <summary>
/// Interface for defining the behaviour of quantization methods.
/// <see href="https://towardsdatascience.com/introduction-to-weight-quantization-2494701b9c0c/"/>
/// </summary>
public interface IQuantization<TIn, TOut> {
    /// <summary>
    /// Quantizes the input tensor.
    /// The quantization process converts the input tensor of type TIn to a quantized tensor of type TOut.
    /// </summary>
    /// <param name="X">Input tensor</param>
    /// <param name="Y">Quantized tensor</param>
    /// <param name="scale">Quantiziation scale</param>
    /// <param name="zeroPoint">Quantiziation zero-point</param>
    /// <returns>Output tensor</returns>
    public void Quantize(TIn[] X, out TOut[] Y, out double scale, out double zeroPoint);

    /// <summary>
    /// Quantizes the input tensor.
    /// The quantization process converts the input tensor of type TIn to a quantized tensor of type TOut.
    /// </summary>
    /// <param name="X">Input tensor</param>
    /// <param name="Y">Quantized tensor</param>
    /// <param name="scale">Quantiziation scale</param>
    /// <param name="zeroPoint">Quantiziation zero-point</param>
    /// <returns>Output tensor</returns>
    public void Quantize(ITensorLike<TIn> X, out ITensorLike<TOut> Y, out double scale, out double zeroPoint) {
        var shape = Enumerable.Range(0, X.Rank).Select(i => X.GetDimension(i)).ToArray();

        // TODO, don't like this for the additional memory allocation, but I need arrays for length computation
        Quantize(X.EnumerateElements().ToArray(), out var data, out var s, out var z);
        
        Y = new GenericTensor<TOut>(shape, data);
        scale = s;
        zeroPoint = z;
    }

    /// <summary>
    /// Dequantizes the input tensor.
    /// The dequantization process converts the quantized tensor of type TOut back to the original tensor type TIn.
    /// </summary>
    /// <param name="X">Input tensor</param>
    /// <param name="Y">Quantized tensor</param>
    /// <param name="scale">Quantiziation scale</param>
    /// <param name="zeroPoint">Quantiziation zero-point</param>
    /// <returns>Input tensor</returns>
    public void Dequantize(out TIn[] X, TOut[] Y, double scale, double zeroPoint);

    /// <summary>
    /// Dequantizes the input tensor.
    /// The dequantization process converts the quantized tensor of type TOut back to the original tensor type TIn.
    /// </summary>
    /// <param name="X">Input tensor</param>
    /// <param name="Y">Quantized tensor</param>
    /// <param name="scale">Quantiziation scale</param>
    /// <param name="zeroPoint">Quantiziation zero-point</param>
    /// <returns>Input tensor</returns>
    public void Dequantize(out ITensorLike<TIn> X, ITensorLike<TOut> Y, double scale, double zeroPoint) {
        var shape = Enumerable.Range(0, Y.Rank).Select(i => Y.GetDimension(i)).ToArray();

        // TODO, don't like this for the additional memory allocation, but I need arrays for length computation
        Dequantize(out var data, Y.EnumerateElements().ToArray(), scale, zeroPoint);

        X = new GenericTensor<TIn>(shape, data);
    }
}

/// <summary>
/// Naïve Absolute Maximum (absmax) symmetric 8bit quantization method.
/// </summary>
public class AbsmaxQuantization : IQuantization<double, byte>, IQuantization<float, byte> {

    public void Quantize(double[] X, out byte[] Y, out double scale, out double zeroPoint) {
        Y = new byte[X.Length];

        // Calculate the scale factor
        // Max(Abs(X))
        var maxX = X.Select(Math.Abs).Max();
        scale = 127 / maxX;

        // Quantization step
        for (var i = 0; i < X.Length; i++) {
            var x_quant = (byte)Math.Round(scale * X[i]);
            Y[i] = x_quant;
        }

        zeroPoint = 0;
    }

    public void Quantize(float[] X, out byte[] Y, out double scale, out double zeroPoint) {
        Y = new byte[X.Length];

        // Calculate the scale factor
        // Max(Abs(X))
        var maxX = X.Select(MathF.Abs).Max();
        scale = 127 / maxX;

        // Quantization step
        var scaleF = (float)scale;
        for (var i = 0; i < X.Length; i++) {
            var x_quant = (byte)MathF.Round(scaleF * X[i]);
            Y[i] = x_quant;
        }

        zeroPoint = 0;
    }

    public void Dequantize(out double[] X, byte[] Y, double scale, double zeroPoint) {
        X = new double[Y.Length];

        // Dequantization step
        for (var i = 0; i < X.Length; i++) {
            var x_dequant = Y[i] / scale;
            X[i] = x_dequant;
        }
    }

    public void Dequantize(out float[] X, byte[] Y, double scale, double zeroPoint) {
        X = new float[Y.Length];

        // Dequantization step
        var scaleF = (float)scale;
        for (var i = 0; i < X.Length; i++) {
            var x_dequant = Y[i] / scaleF;
            X[i] = x_dequant;
        }
    }

}

/// <summary>
/// Naïve Zero-Point asymmetric 8bit quantization method.
/// </summary>
public class ZeroPointQuantization : IQuantization<double, byte>, IQuantization<float, byte> {

    public void Quantize(double[] X, out byte[] Y, out double scale, out double zeroPoint) {
        Y = new byte[X.Length];

        // Calculate the scale factor
        // Max(Abs(X))
        var maxX = X.Max();
        var minX = X.Min();
        scale = 255 / (maxX - minX);
        zeroPoint = -Math.Round(scale * minX) - 128;

        // Quantization step
        for (var i = 0; i < X.Length; i++) {
            var x = X[i];
            var x_quant = (byte)Math.Round(scale * x + zeroPoint);
            Y[i] = x_quant;
        }
    }

    public void Quantize(float[] X, out byte[] Y, out double scale, out double zeroPoint) {
        Y = new byte[X.Length];

        // Calculate the scale factor
        // Max(Abs(X))
        var maxX = X.Max();
        var minX = X.Min();
        var scaleF = 255 / (maxX - minX);
        scale = scaleF;
        var zeroPointF = -MathF.Round(scaleF * minX) - 128;
        zeroPoint = zeroPointF;

        // Quantization step
        for (var i = 0; i < X.Length; i++) {
            var x = X[i];
            var x_quant = (byte)MathF.Round(scaleF * x + zeroPointF);
            Y[i] = x_quant;
        }
    }

    public void Dequantize(out double[] X, byte[] Y, double scale, double zeroPoint) {
        X = new double[Y.Length];

        // Dequantization step
        for (var i = 0; i < X.Length; i++) {
            var x_quant = Y[i];
            var x_dequant = (x_quant - zeroPoint) / scale;
            X[i] = x_dequant;
        }
    }

    public void Dequantize(out float[] X, byte[] Y, double scale, double zeroPoint) {
        X = new float[Y.Length];

        // Dequantization step
        var scaleF = (float)scale;
        var zeroPointF = (float)zeroPoint;
        for (var i = 0; i < X.Length; i++) {
            var x_quant = Y[i];
            var x_dequant = (x_quant - zeroPointF) / scaleF;
            X[i] = x_dequant;
        }
    }

}