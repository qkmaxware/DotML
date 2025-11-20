using SkiaSharp;

namespace DotML.Examples;

public class SKBitmapSet : List<SKBitmap>, IDisposable
{
    public SKBitmapSet(): base() {}

    public SKBitmapSet(int capacity): base(capacity) {}

    public void Dispose()
    {
        foreach (var bmp in this)
            bmp.Dispose();
    }
}

/// <summary>
/// Class to apply data augmentations to an SKBitmap
/// </summary>
public class SKAugmentor
{
    private Random rng = new Random();

    /// <summary>
    /// Dimensions to crop/resize to for final images (cover scaling with center cropping)
    /// </summary>
    public (float Width, float Height)? ForcedOutputDimensions = null;

    /// <summary>
    /// Amount of the image to select before applying transformations (default: select the whole image)
    /// </summary>
    public IProbabilityDistribution<float> SelectionPercent = Distributions.Uniform<float>(1.0, 1.0);

    /// <summary>
    /// Max range to use for rotations (default: rotate by 25deg)
    /// </summary>
    public IProbabilityDistribution<float> RotationDegrees = Distributions.Uniform<float>(-25, 25);

    /// <summary>
    /// Min and max range to use for scaling (default: scale down and up slighly)
    /// </summary>
    public IProbabilityDistribution<float> ScalingFactors = Distributions.Uniform<float>(0.8f, 1.2f);

    /// <summary>
    /// Min and max range to use for adjusting brightness (default: slighly dim and brighten)
    /// </summary>
    public IProbabilityDistribution<float> Brightness = Distributions.Uniform<float>(0.9f, 1.1f);

    /// <summary>
    /// Min and max range to use for adjusting contrast (default: slightly increase and decrease contrast)
    /// </summary>
    public IProbabilityDistribution<float> Contrast = Distributions.Uniform<float>(0.9f, 1.1f);

    /// <summary>
    /// Allow inverting the image colours (default: false)
    /// </summary>
    public bool AllowInverting = false;

    /// <summary>
    /// Allow flipping the image across the x-axis (default: false)
    /// </summary>
    public bool AllowHorizontalFlip = false;

    /// <summary>
    /// Allow flipping the image across the y-axis (default: false)
    /// </summary>
    public bool AllowVerticalFlip = false;

    public SKBitmapSet Augment(SKBitmap original, int augmentations)
    {
        var set = new SKBitmapSet();

        var backgroundColour = GetMedianColour(original);

        for (var i = 0; i < augmentations; i++)
        {
            float keepFactor = SelectionPercent.Sample();
            float angle = RotationDegrees.Sample();
            float scale = ScalingFactors.Sample();
            bool flipH = AllowHorizontalFlip ? rng.NextDouble() > 0.5 : false;
            bool flipV = AllowVerticalFlip ? rng.NextDouble() > 0.5 : false;
            bool invert = AllowInverting ? rng.NextDouble() > 0.5 : false;
            float brightness = Brightness.Sample();
            float contrast = Contrast.Sample();

            using var cropped = RandomCropping(original, keepFactor);
            using var rotated = RotateAndScale(cropped, backgroundColour, angle, scale);
            using var flipped = Flip(rotated, flipH, flipV);
            using var luminos = AdjustBrightness(flipped, brightness, contrast);
            using var inverted = Invert(luminos, invert);

            var transformed = inverted.Copy();
            if (ForcedOutputDimensions.HasValue)
            {
                var (targetW, targetH) = ForcedOutputDimensions.Value;
                var final = ResizeAndCrop(transformed, (int)targetW, (int)targetH);
                transformed.Dispose();
                transformed = final;
            }

            set.Add(transformed);
        }

        return set;
    }

    private SKBitmap RandomCropping(SKBitmap src, float keepFactor)
    {
        keepFactor = Math.Clamp(keepFactor, 0, 1);
        var width = (int)Math.Floor(src.Width * keepFactor);
        var height =(int) Math.Floor(src.Height * keepFactor);

        var offsetX = width < src.Width ? Random.Shared.Next(0, src.Width - width): 0;
        var offsetY = height < src.Height ? Random.Shared.Next(0, src.Height - height) : 0;

        var cropped = new SKBitmap(width, height);
        src.ExtractSubset(cropped, new SKRectI(left: offsetX, top: offsetY, right: offsetX + width, bottom: offsetY + height));
        return cropped;
    }

    private SKBitmap RotateAndScale(SKBitmap src, SKColor backgroundColour, float rotationDegrees, float scale)
    {
        int width = src.Width;
        int height = src.Height;

        // Create a new bitmap to draw into
        var result = new SKBitmap(width, height);
        using var canvas = new SKCanvas(result);

        canvas.Clear(backgroundColour);

        canvas.Translate(width / 2f, height / 2f);
        canvas.Scale(scale);
        canvas.RotateDegrees(rotationDegrees);
        canvas.Translate(-width / 2f, -height / 2f);

        canvas.DrawBitmap(src, 0, 0);
        canvas.Flush();

        return result;
    }

    private SKBitmap Flip(SKBitmap src, bool x, bool y)
    {
        int width = src.Width;
        int height = src.Height;

        // Create a new bitmap to draw into
        var result = new SKBitmap(width, height);
        using var canvas = new SKCanvas(result);

        canvas.Translate(width / 2f, height / 2f);
        canvas.Scale(x ? -1 : 1, y ? -1 : 1);
        canvas.Translate(-width / 2f, -height / 2f);

        canvas.DrawBitmap(src, 0, 0);
        canvas.Flush();

        return result;
    }

    private static SKBitmap AdjustBrightness(SKBitmap src, float brightness, float contrast)
    {
        var adjusted = new SKBitmap(src.Width, src.Height);
        using var canvas = new SKCanvas(adjusted);

        float translate = (brightness - 1f) + (0.5f * (1f - contrast));

        var colorFilter = SKColorFilter.CreateColorMatrix(new float[]
        {
            contrast, 0, 0, 0, translate,
            0, contrast, 0, 0, translate,
            0, 0, contrast, 0, translate,
            0, 0, 0, 1, 0
        });

        using var paint = new SKPaint { ColorFilter = colorFilter };
        canvas.DrawBitmap(src, 0, 0, paint);
        canvas.Flush();

        return adjusted;
    }

    private static SKBitmap Invert(SKBitmap src, bool invert)
    {
        if (!invert)
            return src;

        SKBitmap inverted = new SKBitmap(src.Width, src.Height, src.ColorType, src.AlphaType);
    
        using (var canvas = new SKCanvas(inverted))
        {
            var paint = new SKPaint
            {
                ColorFilter = SKColorFilter.CreateColorMatrix(new float[]
                {
                    -1,  0,  0,  0, 255,  // Red channel: invert and shift
                    0, -1,  0,  0, 255,  // Green channel: invert and shift
                    0,  0, -1,  0, 255,  // Blue channel: invert and shift
                    0,  0,  0,  1,   0   // Alpha channel: unchanged
                })
            };
            
            canvas.DrawBitmap(src, 0, 0, paint);
        }
        
        return inverted;
    }

    private SKBitmap ApplyGaussianNoise(SKBitmap src, float sigma)
    {
        if (sigma <= 0) return src;
        
        SKBitmap noisy = new SKBitmap(src.Width, src.Height, src.ColorType, src.AlphaType);
        var random = new System.Random();
        
        for (int y = 0; y < src.Height; y++)
        {
            for (int x = 0; x < src.Width; x++)
            {
                SKColor pixel = src.GetPixel(x, y);
                float noise = (float)GaussianRandom(random, 0, sigma);
                
                SKColor noisyPixel = new SKColor(
                    (byte)Math.Clamp(pixel.Red + noise, 0, 255),
                    (byte)Math.Clamp(pixel.Green + noise, 0, 255),
                    (byte)Math.Clamp(pixel.Blue + noise, 0, 255),
                    (byte)pixel.Alpha
                );
                noisy.SetPixel(x, y, noisyPixel);
            }
        }
        return noisy;
    }

    private double GaussianRandom(Random rng, double mean, double sigma)
    {
        double u1 = rng.NextDouble();
        double u2 = rng.NextDouble();
        return mean + sigma * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }

    private SKBitmap ApplyGaussianBlur(SKBitmap src, float sigma)
    {
        if (sigma <= 0) return src;
        
        SKBitmap blurred = new SKBitmap(src.Width, src.Height, src.ColorType, src.AlphaType);
        using (var canvas = new SKCanvas(blurred))
        {
            var paint = new SKPaint
            {
                ImageFilter = SKImageFilter.CreateBlur(sigma, sigma)
            };
            canvas.DrawBitmap(src, 0, 0, paint);
        }
        return blurred;
    }

    private SKBitmap ApplyHueShift(SKBitmap src, float hueShift)
    {
        if (Math.Abs(hueShift) < 0.01f) return src;
        
        SKBitmap shifted = new SKBitmap(src.Width, src.Height, src.ColorType, src.AlphaType);
        using (var canvas = new SKCanvas(shifted))
        {
            var paint = new SKPaint
            {
                ColorFilter = SKColorFilter.CreateColorMatrix(new float[]
                {
                    1, 0, 0, 0, hueShift,  // Simplified HSV rotation (approximation)
                    0, 1, 0, 0, 0,
                    0, 0, 1, 0, 0,
                    0, 0, 0, 1, 0
                })
            };
            canvas.DrawBitmap(src, 0, 0, paint);
        }
        return shifted;
    }

    private SKBitmap ApplyCutout(SKBitmap src, int patchSize, int numPatches)
    {
        SKBitmap cutout = new SKBitmap(src.Width, src.Height);
        src.CopyTo(cutout);
        
        for (int p = 0; p < numPatches; p++)
        {
            int x = rng.Next(0, src.Width - patchSize);
            int y = rng.Next(0, src.Height - patchSize);
            
            for (int dy = 0; dy < patchSize; dy++)
                for (int dx = 0; dx < patchSize; dx++)
                    cutout.SetPixel(x + dx, y + dy, SKColors.Black);
        }
        return cutout;
    }

    private SKColor GetAverageColour(SKBitmap bmp)
    {
        long r = 0, g = 0, b = 0;
        int width = bmp.Width;
        int height = bmp.Height;
        int count = 0;
        int step = Math.Max(1, (int)Math.Sqrt((width * height) / 10000f));

        for (int y = 0; y < height; y += step)
        {
            for (int x = 0; x < width; x += step)
            {
                var c = bmp.GetPixel(x, y);
                r += c.Red;
                g += c.Green;
                b += c.Blue;
                count++;
            }
        }

        return new SKColor((byte)(r / count), (byte)(g / count), (byte)(b / count));
    }

    private SKColor GetMedianColour(SKBitmap bmp)
    {
        int width = bmp.Width;
        int height = bmp.Height;
        int step = Math.Max(1, (int)Math.Sqrt((width * height) / 10000f)); 
        // sample up to ~10k pixels for speed

        List<byte> rs = new();
        List<byte> gs = new();
        List<byte> bs = new();

        for (int y = 0; y < height; y += step)
        {
            for (int x = 0; x < width; x += step)
            {
                var color = bmp.GetPixel(x, y);
                rs.Add(color.Red);
                gs.Add(color.Green);
                bs.Add(color.Blue);
            }
        }

        rs.Sort();
        gs.Sort();
        bs.Sort();

        byte r = rs[rs.Count / 2];
        byte g = gs[gs.Count / 2];
        byte b = bs[bs.Count / 2];

        return new SKColor(r, g, b);
    }

    private SKBitmap ResizeAndCrop(SKBitmap src, int targetWidth, int targetHeight)
    {
        float scale = Math.Max((float)targetWidth / src.Width, (float)targetHeight / src.Height);
        int scaledW = (int)(src.Width * scale);
        int scaledH = (int)(src.Height * scale);

        using var scaled = src.Resize(new SKImageInfo(scaledW, scaledH), SKSamplingOptions.Default);

        // Compute center crop
        int x = (scaledW - targetWidth) / 2;
        int y = (scaledH - targetHeight) / 2;
        var rect = new SKRectI(x, y, x + targetWidth, y + targetHeight);

        var cropped = new SKBitmap(targetWidth, targetHeight);
        using var canvas = new SKCanvas(cropped);
        canvas.DrawBitmap(scaled, rect, new SKRect(0, 0, targetWidth, targetHeight));

        return cropped;
    }
}