using System.Drawing;
using System.Numerics;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// A layer that acts as a normalization layer
/// </summary>
public abstract class NormalizationLayer : NetworkLayer
{
    /// <summary>
    /// Compute the mean and variance of the given values
    /// </summary>
    /// <param name="values">values</param>
    /// <param name="mean">mean of the values</param>
    /// <param name="variance">variance of the values</param>
    protected void MeanAndVariance(Span<float> values, out float mean, out float variance)
    {
        var length = values.Length;
        if (length == 0)
        {
            mean = 0;
            variance = 0;
            return;
        }

        // Vectorized elements
        float sum = 0, sumSq = 0;
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<float>.IsSupported)
        {
            int simdLength = Vector<float>.Count;
            int simdLimit = length - simdLength + 1;

            Vector<float> vecSum = Vector<float>.Zero;
            Vector<float> vecSumSq = Vector<float>.Zero;

            for (; i < simdLimit; i += simdLength)
            {
                var v = new Vector<float>(values.Slice(i, simdLength));
                vecSum += v;
                vecSumSq += v * v;
            }

            // Sum vector lanes
            sum += Vector.Sum(vecSum);
            sumSq += Vector.Sum(vecSumSq);
        }
        // Remaining elements
        for (; i < length; i++)
        {
            float val = values[i];
            sum += val;
            sumSq += val * val;
        }

        mean = sum / length;
        variance = (sumSq / length) - (mean * mean);
    }
}