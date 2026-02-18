using Colour = System.Drawing.Color;

namespace DotML.Network.Embedding;

/// <summary>
/// Generic base class for all gradient-based image embeddings.
/// </summary>
public abstract class PixelGradient:  ImageEmbedding<Colour> {

    protected float ToGreyscale(Colour colour) => (0.2126f * colour.R + 0.7152f * colour.G + 0.0722f * colour.B);
    protected float ToNormalizedGreyscale(Colour colour) => (0.2126f * colour.R + 0.7152f * colour.G + 0.0722f * colour.B) / 255.0f;

    protected abstract float GetTensorValue(int row, int column, IImage<Colour> image);

    public override Tensor<float> ToTensor(IImage<Colour> image) {
        Tensor<float> encoding = Tensor<float>.Defaults(new TensorShape(1, image.Height, image.Width));

        for (var r = 0; r < image.Height; r++) {
            for (var c = 0; c < image.Width; c++) {
                encoding[0, r, c] = GetTensorValue(r, c, image);
            }
        }

        return encoding;
    }
}

/// <summary>
/// Generic base class for all convolution-based gradient image embeddings.
/// </summary>
public abstract class ConvolutionGradient: PixelGradient {
    private Matrix<float> KernelX;
    private Matrix<float> KernelY;

    public bool NormalizePixelValues {get; set;} = false;

    public ConvolutionGradient(Matrix<float> kernelX, Matrix<float> kernelY): base() {
        if (kernelX.Rows % 2 == 0 || kernelX.Columns % 2 == 0) {
            throw new ArgumentException(nameof(kernelX), "KernelX must have odd dimensions");
        }   
        if (kernelY.Rows % 2 == 0 || kernelY.Columns % 2 == 0) {
            throw new ArgumentException(nameof(kernelY), "KernelY must have odd dimensions");
        }

        this.KernelX = kernelX;
        this.KernelY = kernelY;        
    }

    protected float GetPixelValue(IImage<Colour> image, int x, int y)
    {
        // Simple edge handling: clamp
        x = Math.Clamp(x, 0, image.Width - 1);
        y = Math.Clamp(y, 0, image.Height - 1);
        return NormalizePixelValues ? ToNormalizedGreyscale(image[x, y]) : ToGreyscale(image[x, y]);
    }

    protected float ConvolveKernelX(
        IImage<Colour> image,
        int x,
        int y
    ) {
        float sum = 0f;

        var halfKernelWidth = KernelX.Columns / 2;
        var halfKernelHeight = KernelX.Rows / 2;

        for (int ky = -halfKernelHeight; ky <= halfKernelHeight; ky++)
        {
            for (int kx = -halfKernelWidth; kx <= halfKernelWidth; kx++)
            {
                sum += KernelX[ky + halfKernelHeight, kx + halfKernelWidth] * GetPixelValue(image, x + kx, y + ky);
            }
        }

        return sum;
    }

    protected float ConvolveKernelY(
        IImage<Colour> image,
        int x,
        int y
    ) {
        float sum = 0f;

        var halfKernelWidth = KernelY.Columns / 2;
        var halfKernelHeight = KernelY.Rows / 2;

        for (int ky = -halfKernelHeight; ky <= halfKernelHeight; ky++)
        {
            for (int kx = -halfKernelWidth; kx <= halfKernelWidth; kx++)
            {
                sum += KernelY[ky + halfKernelHeight, kx + halfKernelWidth] * GetPixelValue(image, x + kx, y + ky);
            }
        }

        return sum;
    }
}

#region Sobel
/// <summary>
/// Embedding base class for Sobel based operators
/// </summary>
public abstract class SobelGradient: ConvolutionGradient {
    public SobelGradient(): base(Kernels.SobelXKernel(), Kernels.SobelYKernel()) { }
}

/// <summary>
/// Embedding creator using Sobel operator in the X direction
/// </summary>
public class SobelGradientX: SobelGradient {
    protected override float GetTensorValue(int row, int column, IImage<Colour> image) {
        return ConvolveKernelX(image, column, row);
    }
}

/// <summary>
/// Embedding creator using Sobel operator in the Y direction
/// </summary>
public class SobelGradientY: SobelGradient {
    protected override float GetTensorValue(int row, int column, IImage<Colour> image) {
        return ConvolveKernelY(image, column, row);
    }
}

/// <summary>
/// Embedding creator using Sobel gradient magnitude
/// </summary>
public class SobelGradientMagnitude: SobelGradient {
    protected override float GetTensorValue(int row, int column, IImage<Colour> image) {
        float gx = ConvolveKernelX(image, column, row);
        float gy = ConvolveKernelY(image, column, row);
        return MathF.Sqrt(gx * gx + gy * gy);
    }
}

/// <summary>
/// Embedding creator using Sobel gradient orientation
/// </summary>
public class SobelGradientOrientation: SobelGradient {
    protected override float GetTensorValue(int row, int column, IImage<Colour> image) {
        float gx = ConvolveKernelX(image, column, row);
        float gy = ConvolveKernelY(image, column, row);
        return MathF.Atan2(gy, gx);
    }
}

#endregion

#region Prewitt
/// <summary>
/// Embedding base class for Prewitt based operators
/// </summary>
public abstract class PrewittGradient: ConvolutionGradient {
    public PrewittGradient(): base(Kernels.PrewittXKernel(), Kernels.PrewittYKernel()) { }
}

/// <summary>
/// Embedding creator using Prewitt operator in the X direction
/// </summary>
public class PrewittGradientX: PrewittGradient {
    protected override float GetTensorValue(int row, int column, IImage<Colour> image) {
        return ConvolveKernelX(image, column, row);
    }
}

/// <summary>
/// Embedding creator using Prewitt operator in the Y direction
/// </summary>
public class PrewittGradientY: PrewittGradient {
    protected override float GetTensorValue(int row, int column, IImage<Colour> image) {
        return ConvolveKernelY(image, column, row);
    }
}

/// <summary>
/// Embedding creator using Prewitt gradient magnitude
/// </summary>
public class PrewittGradientMagnitude: PrewittGradient {
    protected override float GetTensorValue(int row, int column, IImage<Colour> image) {
        float gx = ConvolveKernelX(image, column, row);
        float gy = ConvolveKernelY(image, column, row);
        return MathF.Sqrt(gx * gx + gy * gy);
    }
}

/// <summary>
/// Embedding creator using Prewitt gradient orientation
/// </summary>
public class PrewittGradientOrientation: PrewittGradient {
    protected override float GetTensorValue(int row, int column, IImage<Colour> image) {
        float gx = ConvolveKernelX(image, column, row);
        float gy = ConvolveKernelY(image, column, row);
        return MathF.Atan2(gy, gx);
    }
}

#endregion

#region Scharr
/// <summary>
/// Embedding base class for Scharr based operators
/// </summary>
public abstract class ScharrGradient: ConvolutionGradient {
    public ScharrGradient(): base(Kernels.ScharrXKernel(), Kernels.ScharrYKernel()) { }
}

/// <summary>
/// Embedding creator using Scharr operator in the X direction
/// </summary>
public class ScharrGradientX: ScharrGradient {
    protected override float GetTensorValue(int row, int column, IImage<Colour> image) {
        return ConvolveKernelX(image, column, row);
    }
}

/// <summary>
/// Embedding creator using Scharr operator in the Y direction
/// </summary>
public class ScharrGradientY: ScharrGradient {
    protected override float GetTensorValue(int row, int column, IImage<Colour> image) {
        return ConvolveKernelY(image, column, row);
    }
}

/// <summary>
/// Embedding creator using Scharr gradient magnitude
/// </summary>
public class ScharrGradientMagnitude: ScharrGradient {
    protected override float GetTensorValue(int row, int column, IImage<Colour> image) {
        float gx = ConvolveKernelX(image, column, row);
        float gy = ConvolveKernelY(image, column, row);
        return MathF.Sqrt(gx * gx + gy * gy);
    }
}

/// <summary>
/// Embedding creator using Scharr gradient orientation
/// </summary>
public class ScharrGradientOrientation: ScharrGradient {
    protected override float GetTensorValue(int row, int column, IImage<Colour> image) {
        float gx = ConvolveKernelX(image, column, row);
        float gy = ConvolveKernelY(image, column, row);
        return MathF.Atan2(gy, gx);
    }
}

#endregion