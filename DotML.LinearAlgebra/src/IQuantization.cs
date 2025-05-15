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
    /// <returns>Output tensor</returns>
    public (TOut[] Tensor, double Scale, double ZeroPoint) Quantize(TIn[] X);

    /// <summary>
    /// Dequantizes the input tensor.
    /// The dequantization process converts the quantized tensor of type TOut back to the original tensor type TIn.
    /// </summary>
    /// <param name="Y">Output tensor</param>
    /// <returns>Input tensor</returns>
    public TIn[] Dequantize(TOut[] Y, double scale, double zeroPoint);

    /// <summary>
    /// Quantizes the input tensor.
    /// The quantization process converts the input tensor of type TIn to a quantized tensor of type TOut.
    /// </summary>
    /// <param name="X">Input tensor</param>
    /// <returns>Output tensor</returns>
    public (ITensorLike<TOut> Tensor, double Scale, double ZeroPoint) Quantize(ITensorLike<TIn> X) {
        var shape = Enumerable.Range(0, X.Rank).Select(i => X.GetDimension(i)).ToArray();

        // TODO, don't like this for the additional memory allocation, but I need arrays for length computation
        var (data, scale, zero) = Quantize(X.EnumerateElements().ToArray());

        return (new GenericTensor<TOut>(shape, data), scale, zero);
    }

    /// <summary>
    /// Dequantizes the input tensor.
    /// The dequantization process converts the quantized tensor of type TOut back to the original tensor type TIn.
    /// </summary>
    /// <param name="Y">Output tensor</param>
    /// <returns>Input tensor</returns>
    public ITensorLike<TIn> Dequantize(ITensorLike<TOut> Y, double scale, double zeroPoint) {
        var shape = Enumerable.Range(0, Y.Rank).Select(i => Y.GetDimension(i)).ToArray();

        // TODO, don't like this for the additional memory allocation, but I need arrays for length computation
        var data = Dequantize(Y.EnumerateElements().ToArray(), scale, zeroPoint);

        return new GenericTensor<TIn>(shape, data);
    }
}

/// <summary>
/// Naïve Absolute Maximum (absmax) symmetric 8bit quantization method.
/// </summary>
public class AbsmaxQuantization : IQuantization<double, byte> {

    public (byte[] Tensor, double Scale, double ZeroPoint) Quantize(double[] X) {
        byte[] Y = new byte[X.Length];

        // Calculate the scale factor
        // Max(Abs(X))
        var maxX = X.Select(Math.Abs).Max();
        var scale = 127 / maxX;

        // Quantization step
        for (var i = 0; i < X.Length; i++) {
            var x_quant = (byte)Math.Round(scale * X[i]);
            Y[i] = x_quant;
        }

        return (Y, scale, 0);
    }

    public double[] Dequantize(byte[] Y, double scale, double zeroPoint) {
        double[] X = new double[Y.Length];

        // Dequantization step
        for (var i = 0; i < X.Length; i++) {
            var x_dequant = Y[i] / scale;
            X[i] = x_dequant;
        }

        return X;
    }

}

/// <summary>
/// Naïve Zero-Point asymmetric 8bit quantization method.
/// </summary>
public class ZeroPointQuantization : IQuantization<double, byte> {


    public (byte[] Tensor, double Scale, double ZeroPoint) Quantize(double[] X) {
        byte[] Y = new byte[X.Length];

        // Calculate the scale factor
        // Max(Abs(X))
        var maxX = X.Max();
        var minX = X.Min();
        var scale = 255 / (maxX - minX);
        var zeroPoint = -Math.Round(scale * minX) - 128;

        // Quantization step
        for (var i = 0; i < X.Length; i++) {
            var x = X[i];
            var x_quant = (byte)Math.Round(scale * x + zeroPoint);
            Y[i] = x_quant;
        }

        return (Y, scale, zeroPoint);
    }

    public double[] Dequantize(byte[] Y, double scale, double zeroPoint) {
        double[] X = new double[Y.Length];

        // Dequantization step
        for (var i = 0; i < X.Length; i++) {
            var x_quant = Y[i];
            var x_dequant = (x_quant - zeroPoint) / scale;
            X[i] = x_dequant;
        }

        return X;
    }
}