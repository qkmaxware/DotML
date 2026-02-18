using SkiaSharp;

namespace DotML.Examples;

public struct YCrCb
{
    public byte Y;
    public byte Cr;
    public byte Cb;

    public byte Luminance => Y;
    public byte BlueDifferenceChroma => Cb;
    public byte RedDifferenceChroma => Cr;
}

public static class SKBitmapExtensions
{
    /// <summary>
    /// Slice an image into multiple sub-images ie for spritesheets
    /// </summary>
    /// <param name="original">original image</param>
    /// <param name="rows">number of rows</param>
    /// <param name="columns">number of columns</param>
    /// <returns>set of sliced images in row-major order</returns>
    /// <exception cref="ArgumentNullException"></exception>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public static SKBitmapSet Slice(this SKBitmap original, int rows, int columns)
    {
        if (original == null)
            throw new ArgumentNullException(nameof(original));
        if (rows <= 0)
            throw new ArgumentOutOfRangeException(nameof(rows));
        if (columns <= 0)
            throw new ArgumentOutOfRangeException(nameof(columns));

        var result = new SKBitmapSet();

        int tileWidth = original.Width / columns;
        int tileHeight = original.Height / rows;

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                int x = col * tileWidth;
                int y = row * tileHeight;

                // Ensure the last tile extends to the edge if not evenly divisible
                //int width = (col == columns - 1) ? original.Width - x : tileWidth;
                //int height = (row == rows - 1) ? original.Height - y : tileHeight;
                int width = tileWidth;
                int height = tileHeight;

                var rect = new SKRectI(x, y, x + width, y + height);

                var slice = new SKBitmap(width, height);
                using (var canvas = new SKCanvas(slice))
                {
                    var srcRect = rect;
                    var destRect = new SKRect(0, 0, width, height);
                    canvas.DrawBitmap(original, srcRect, destRect);
                }

                result.Add(slice);
            }
        }

        return result;
    }

    public static YCrCb[,] ToYCrCb(this SKBitmap original)
    {
        YCrCb[,] values = new YCrCb[original.Height, original.Width];

        for (var r = 0; r < original.Height; r++)
        {
            for (var c = 0; c < original.Width; c++)
            {
                var colour = original.GetPixel(c, r);
                var value = ToYCrCb(colour);
                values[r, c] = value;
            }
        }
        return values;
    }

    private static YCrCb ToYCrCb(SKColor color)
    {
        var R = color.Red;
        var G = color.Green;
        var B = color.Blue;

        double y  =  16  + (65.481 * R + 128.553 * G + 24.966 * B) / 255.0;
        double cb = 128  + (-37.797 * R - 74.203 * G + 112.0 * B) / 255.0;
        double cr = 128  + (112.0 * R - 93.786 * G - 18.214 * B) / 255.0;

        byte Y  = (byte)Math.Round(Math.Min(Math.Max(y, 0), 255));
        byte Cb = (byte)Math.Round(Math.Min(Math.Max(cb, 0), 255));
        byte Cr = (byte)Math.Round(Math.Min(Math.Max(cr, 0), 255));

        return new YCrCb
        {
            Y  = Y,
            Cb = Cb,
            Cr = Cr
        };
    }

    public static Tensor<float> ToTensor<T>(this T[,] image, Func<T, float> extract)
    {
        var tensor = Tensor<float>.Defaults(new TensorShape(1, image.GetLength(0), image.GetLength(1)));

        for (var r = 0; r < tensor.Shape[NCHW.Rows]; r++)
        {
            for (var c = 0; c < tensor.Shape[NCHW.Columns]; c++)
            {
                tensor[0, r, c] = extract(image[r, c]);
            }
        }

        return tensor;
    }

    /// <summary>
    /// Convert a bitmap to an greyscale tensor with pixel values between 0 and 255
    /// </summary>
    /// <param name="bitmap"bitmap></param>
    /// <param name="region">region to use for tensor</param>
    /// <returns>1 channel tensor</returns>
    public static Tensor<float> ToGreyscaleTensor(this SKBitmap bitmap, SKRectI? region = null)
    {
        var crop = region ?? new SKRectI(0, 0, bitmap.Width, bitmap.Height);

        var width = crop.Width;
        var height = crop.Height;
        var tensor = Tensor<float>.Defaults(new TensorShape(1, height, width)); // NCHW

        for (var r = 0; r < height; r++)
        {
            for (var c = 0; c < width; c++)
            {
                var pixel = bitmap.GetPixel(crop.Left + c, crop.Top + r);
                var R = pixel.Red;
                var G = pixel.Green;
                var B = pixel.Blue;
                tensor[0, r, c] = (byte)Math.Clamp(0.2126f * R + 0.7152f * G + 0.0722f * B, 0f, 255f);
            }
        }

        return tensor;
    }

    public static SKBitmapSet ToGreyscaleBitmaps(this Tensor<float> tensor)
    {
        tensor = tensor.ReshapeShared(tensor.Shape.NormalizeRank(4)); // NCHW
        
        var N = tensor.Shape[0];
        var C = tensor.Shape[1];
        var H = tensor.Shape[2];
        var W = tensor.Shape[3];

        var set = new SKBitmapSet(N);

        for (var n = 0; n < N; n++)
        {
            var bitmap = new SKBitmap(width: W, height: H);

            for (var y = 0; y < H; y++)
            {
                for (var x = 0; x < W; x++)
                {
                    var lumos = (byte)Math.Clamp((C >= 1 ? tensor[n, 0, y, x] : 0) * 255.0f, 0, 255);
                    bitmap.SetPixel(x, y, new SKColor(red: lumos, lumos, lumos));
                }
            }

            set.Add(bitmap);
        }

        return set;
    }


    /// <summary>
    /// Convert a bitmap to an RGB colour tensor with pixel values between 0 and 255
    /// </summary>
    /// <param name="bitmap"bitmap></param>
    /// <param name="region">region to use for tensor</param>
    /// <returns>3 channel tensor</returns>
    public static Tensor<float> ToColourTensor(this SKBitmap bitmap, SKRectI? region = null)
    {
        var crop = region ?? new SKRectI(0, 0, bitmap.Width, bitmap.Height);

        var width = crop.Width;
        var height = crop.Height;
        var tensor = Tensor<float>.Defaults(new TensorShape(3, height, width)); // NCHW

        const int R = 0;
        const int G = 1;
        const int B = 2;

        for (var r = 0; r < height; r++)
        {
            for (var c = 0; c < width; c++)
            {
                var pixel = bitmap.GetPixel(crop.Left + c, crop.Top + r);
                tensor[R, r, c] = pixel.Red;
                tensor[G, r, c] = pixel.Green;
                tensor[B, r, c] = pixel.Blue;
            }
        }

        return tensor;
    }

    public static SKBitmapSet ToColourBitmaps(this Tensor<float> tensor)
    {
        tensor = tensor.ReshapeShared(tensor.Shape.NormalizeRank(4)); // NCHW
        
        var N = tensor.Shape[0];
        var C = tensor.Shape[1];
        var H = tensor.Shape[2];
        var W = tensor.Shape[3];

        var set = new SKBitmapSet(N);

        for (var n = 0; n < N; n++)
        {
            var bitmap = new SKBitmap(width: W, height: H);

            for (var y = 0; y < H; y++)
            {
                for (var x = 0; x < W; x++)
                {
                    var red = (byte)Math.Clamp((C >= 1 ? tensor[n, 0, y, x] : 0) * 255.0f, 0, 255);
                    var grn = (byte)Math.Clamp((C >= 2 ? tensor[n, 1, y, x] : 0) * 255.0f, 0, 255);
                    var blu = (byte)Math.Clamp((C >= 3 ? tensor[n, 2, y, x] : 0) * 255.0f, 0, 255);
                    bitmap.SetPixel(x, y, new SKColor(red: red, grn, blu));
                }
            }

            set.Add(bitmap);
        }

        return set;
    }

    public static SKBitmap ScaleToCover(this SKBitmap bitmap, int width, int height)
    {
        var targetAspect = width / (float)height;
        var srcW = bitmap.Width;
        var srcH = bitmap.Height;
        SKRectI srcRect;
        var srcAspect = srcW / (float)srcH;
        if (srcAspect > targetAspect)
        {
            // image is wider than target -> crop width
            var newW = Math.Max(1, (int)Math.Round(targetAspect * srcH));
            var x = (srcW - newW) / 2;
            srcRect = new SKRectI(x, 0, x + newW, srcH);
        }
        else
        {
            // image is taller than target -> crop height
            var newH = Math.Max(1, (int)Math.Round(srcW / targetAspect));
            var y = (srcH - newH) / 2;
            srcRect = new SKRectI(0, y, srcW, y + newH);
        }

        using var cropped = new SKBitmap(srcRect.Width, srcRect.Height, isOpaque: true);
        using (var canvas = new SKCanvas(cropped))
        {
            canvas.Clear(SKColors.Transparent);
            canvas.DrawBitmap(bitmap, srcRect, new SKRect(0, 0, srcRect.Width, srcRect.Height));
        }

        var scaled = new SKBitmap(width: width, height: height, isOpaque: true);
        cropped.ScalePixels(scaled, SKSamplingOptions.Default);
        return scaled;
    }
}