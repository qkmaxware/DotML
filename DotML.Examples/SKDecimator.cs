using SkiaSharp;

namespace DotML.Examples;

/// <summary>
/// Utility class to destroy images over a sequence of steps of increasing noise
/// </summary>
public class SKDecimator
{
    /// <summary>
    /// Number of steps until the image is basically all noise
    /// </summary>
    public int Steps {get; init;}

    public SKDecimator(int steps)
    {
        this.Steps = Math.Max(1, steps);
    }

    public SKBitmapSet Decimate(SKBitmap original)
    {
        var steps = new SKBitmapSet(this.Steps);

        // Add the 0th step (original image)
        steps.Add(original.Copy());

        // For each subsequent step, decimate it
        for (var i = 1; i < Steps; i++)
        {
            // Calculate noise amount: 0 at step 0, 1.0 at final step
            float noiseAmount = (float)i / (Steps - 1);
            SKBitmap decimated = Decimate(steps[i - 1], noiseAmount);
            steps.Add(decimated);


        }

        return steps;
    }

    private SKBitmap Decimate(SKBitmap src, float amt)
    {
        amt = Math.Clamp(amt, 0, 1);
        
        SKBitmap noisy = new SKBitmap(src.Width, src.Height, src.ColorType, src.AlphaType);
        var random = Random.Shared;

        for (int y = 0; y < src.Height; y++)
        {
            for (int x = 0; x < src.Width; x++)
            {
                SKColor original = src.GetPixel(x, y);
                
                // Generate random noise pixel
                byte noiseRed = (byte)random.Next(0, 256);
                byte noiseGreen = (byte)random.Next(0, 256);
                byte noiseBlue = (byte)random.Next(0, 256);

                // Blend: (1 - amt) * original + amt * noise
                byte finalRed = (byte)(original.Red * (1 - amt) + noiseRed * amt);
                byte finalGreen = (byte)(original.Green * (1 - amt) + noiseGreen * amt);
                byte finalBlue = (byte)(original.Blue * (1 - amt) + noiseBlue * amt);

                SKColor blended = new SKColor(finalRed, finalGreen, finalBlue, original.Alpha);
                noisy.SetPixel(x, y, blended);
            }
        }

        return noisy;
    }
}