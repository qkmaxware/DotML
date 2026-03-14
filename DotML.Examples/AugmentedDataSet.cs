using DotML.Network.Training;

namespace DotML.Examples;

public class Augmented2DClassificationDataSource
: GenerativeVariationDataSource<float>
{
    private Random rng = new Random();

    public bool FlipX {get; set;}
    public bool FlipY {get; set;}
    public int MaxShift {get; set;}
    private IProbabilityDistribution<float>? Noise {get; set;}

    public Augmented2DClassificationDataSource(
        ITrainingDataSource<float> baseDataSource, 
        int variations = 0, 
        bool flipX = false, 
        bool flipY = false, 
        int maxShift = 0,
        IProbabilityDistribution<float>? noise = null
    ) : base(baseDataSource, variations)
    {
        this.FlipX = flipX;
        this.FlipY = flipY;
        this.MaxShift = Math.Max(0, maxShift);
        this.Noise = noise;
    }

    // Create a random variation of the image image. Output is unchanged, for use with classification tasks.
    public override (Tensor<float> Input, Tensor<float> Output) Vary(Tensor<float> input, Tensor<float> output, int variationIndex)
    {
        // TODO Generates tonnes of garbage, make it only generate a single modified tensor rather than a different one each step

        var variation = input;

        // X-axis flip
        if (this.FlipX && rng.NextDouble() < 0.5)
        {
            variation = variation.Mirror(NCHW.Columns);
        }

        // Y-axis flip
        if (this.FlipY && rng.NextDouble() < 0.5)
        {
            variation = variation.Mirror(NCHW.Rows);
        }  

        // Shift (functionally similar to padding with random crop) 
        if (MaxShift > 0)
        {
            int shiftX = rng.Next(-MaxShift, MaxShift + 1);
            int shiftY = rng.Next(-MaxShift, MaxShift + 1);
            if (shiftX == 0)
            {
                shiftY = rng.Next(-MaxShift, MaxShift);
                if (shiftY > 0) 
                    shiftY += 1; // avoid no-shift
            } 

            var shifted = Tensor<float>.ZerosLike(variation);
            for (int c = 0; c < variation.Shape[NCHW.Channels]; c++)
            {
                for (int y = 0; y < variation.Shape[NCHW.Rows]; y++)
                {
                    for (int x = 0; x < variation.Shape[NCHW.Columns]; x++)
                    {
                        int srcX = x - shiftX;
                        int srcY = y - shiftY;
                        if (srcX >= 0 && srcX < variation.Shape[NCHW.Columns] &&
                            srcY >= 0 && srcY < variation.Shape[NCHW.Rows])
                        {
                            shifted[c, y, x] = variation[c, srcY, srcX];
                        }
                    }
                }
            }
            variation = shifted;
        }

        // Apply noise
        if (Noise is not null)
        {
            variation = variation.Clone();
            foreach (ref float v in variation.AsSpan())
            {
                v += Noise.Sample();
            }
        }

        return (variation, output);
    }
}