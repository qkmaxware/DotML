using Colour = System.Drawing.Color;

namespace DotML.Network.Embedding;

/// <summary>
/// Generic base class for all classes that create embeddings from images.
/// </summary>
public abstract class ImageEmbedding<TPixel>: IEmbedding<IImage<TPixel>, float> {
    public abstract Tensor<float> ToTensor(IImage<TPixel> image);
}

/// <summary>
/// Creates a single channel greyscale pixel embedding from an image.
/// </summary>
public class GreyscalePixels:  ImageEmbedding<Colour> {

    private float Offset = 0.0f;

    public bool NormalizePixelValues {get; set;} = false;

    public GreyscalePixels(float offset = 0, bool normalize = false) {
        this.SetOffset(offset);
        this.NormalizePixelValues = normalize;
    }

    public void SetOffset(float offset) {
        Offset = offset;
    }

    protected float GetGreyscaleValue(Colour colour) {
        var grey = 0.2126f * colour.R + 0.7152f * colour.G + 0.0722f * colour.B;
        return NormalizePixelValues ? grey / 255.0f : grey;
    }

    public override Tensor<float> ToTensor(IImage<Colour> image) {
        Tensor<float> encoding = Tensor<float>.Defaults(new Shape(1, image.Height, image.Width));

        for (var r = 0; r < image.Height; r++) {
            for (var c = 0; c < image.Width; c++) {
                Colour colour = image[c, r];
                encoding[0, r, c] = GetGreyscaleValue(colour) - Offset;
            }
        }

        return encoding;
    }
}

/// <summary>
/// Creates a raw RGB 3 channel pixel embedding from an image.
/// </summary>
public class RgbPixels:  ImageEmbedding<Colour> {

    public const int RedChannel = 0;
    public const int GreenChannel = 1;
    public const int BlueChannel = 2;

    private float RedOffset = 0.0f;
    private float GreenOffset = 0.0f;
    private float BlueOffset = 0.0f;

    public bool NormalizePixelValues {get; set;} = false;

    public RgbPixels(float redOffset = 0, float greenOffset = 0, float blueOffset = 0, bool normalize = false) {
        this.SetChannelOffsets(redOffset, greenOffset, blueOffset);
        this.NormalizePixelValues = normalize;
    }

    public void SetRedOffset(float offset) {
        RedOffset = offset;
    }

    public void SetGreenOffset(float offset) {
        GreenOffset = offset;
    }

    public void SetBlueOffset(float offset) {
        BlueOffset = offset;
    }

    public void SetChannelOffsets(float redOffset, float greenOffset, float blueOffset) {
        RedOffset = redOffset;
        GreenOffset = greenOffset;
        BlueOffset = blueOffset;
    }

    protected float GetRed(Colour colour) => NormalizePixelValues ? colour.R / 255.0f : colour.R;
    protected float GetGreen(Colour colour) => NormalizePixelValues ? colour.G / 255.0f : colour.G;
    protected float GetBlue(Colour colour) => NormalizePixelValues ? colour.B / 255.0f : colour.B;

    public override Tensor<float> ToTensor(IImage<Colour> image) {
        Tensor<float> encoding = Tensor<float>.Defaults(new Shape(3, image.Height, image.Width));

        for (var r = 0; r < image.Height; r++) {
            for (var c = 0; c < image.Width; c++) {
                Colour colour = image[c, r];
                encoding[RedChannel, r, c] = GetRed(colour) - RedOffset;
                encoding[GreenChannel, r, c] = GetGreen(colour) - GreenOffset;
                encoding[BlueChannel, r, c] = GetBlue(colour) - BlueOffset;
            }
        }

        return encoding;
    }
}

/// <summary>
/// Creates a YCrCb 3 channel pixel embedding from an image.
/// </summary>
public class YCrCbPixels:  ImageEmbedding<Colour> {

    public const int YChannel = 0;
    public const int CrChannel = 1;
    public const int CbChannel = 2;

    private float YOffset = 0.0f;
    private float CrOffset = 0.0f;
    private float CbOffset = 0.0f;

    public bool NormalizePixelValues {get; set;} = false;

    public YCrCbPixels(float yOffset = 0, float crOffset = 0, float cbOffset = 0, bool normalize = false) {
        this.SetChannelOffsets(yOffset, crOffset, cbOffset);
        this.NormalizePixelValues = normalize;
    }

    public void SetYOffset(float offset) {
        YOffset = offset;
    }

    public void SetCrOffset(float offset) {
        CrOffset = offset;
    }

    public void SetCbOffset(float offset) {
        CbOffset = offset;
    }

    public void SetChannelOffsets(float yOffset, float crOffset, float cbOffset) {
        YOffset = yOffset;
        CrOffset = crOffset;
        CbOffset = cbOffset;
    }

    private void GetYCbCr(Colour colour, out float Y, out float Cb, out float Cr) {
        float y  =  16f  + (65.481f * colour.R + 128.553f * colour.G + 24.966f * colour.B) / 255.0f;
        float cb = 128f  + (-37.797f * colour.R - 74.203f * colour.G + 112.0f * colour.B) / 255.0f;
        float cr = 128f  + (112.0f * colour.R - 93.786f * colour.G - 18.214f * colour.B) / 255.0f;

        Y  = (MathF.Min(MathF.Max(y, 0), 255));
        Cb = (MathF.Min(MathF.Max(cb, 0), 255));
        Cr = (MathF.Min(MathF.Max(cr, 0), 255));

        if (this.NormalizePixelValues) {
            Y  /= 255.0f;
            Cb /= 255.0f;
            Cr /= 255.0f;
        }
    }

    public override Tensor<float> ToTensor(IImage<Colour> image) {
        Tensor<float> encoding = Tensor<float>.Defaults(new Shape(3, image.Height, image.Width));

        for (var r = 0; r < image.Height; r++) {
            for (var c = 0; c < image.Width; c++) {
                Colour colour = image[c, r];
                GetYCbCr(colour, out float Y, out float Cb, out float Cr);
                encoding[YChannel, r, c] = Y - YOffset;
                encoding[CrChannel, r, c] = Cr - CrOffset;
                encoding[CbChannel, r, c] = Cb - CbOffset;
            }
        }

        return encoding;
    }
}