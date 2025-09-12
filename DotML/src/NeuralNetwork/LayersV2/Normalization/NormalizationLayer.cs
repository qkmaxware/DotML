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
    /// <param name="offset">offset to start computing from</param>
    /// <param name="stride">amount to increment by (useful for non-sequential values)</param>
    protected void MeanAndVariance(ReadOnlySpan<float> values, out float mean, out float variance, int offset = 0, int stride = 1)
    {
        var length = values.Length;
        if (length == 0)
        {
            mean = 0;
            variance = 0;
            return;
        }

        // Vectorized elements
        int count = 0;                  // Since we can have stride, count may not = length
        float sum = 0, sumSq = 0;       // Sum and sum of squares
        int i = offset;                 // Starting index 0 or offset
        if (1 == stride && Vector.IsHardwareAccelerated && Vector<float>.IsSupported)
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
                count += simdLength;
            }

            // Sum vector lanes
            sum += Vector.Sum(vecSum);
            sumSq += Vector.Sum(vecSumSq);
        }
        // Remaining elements
        for (; i < length; i += stride)
        {
            float val = values[i];
            sum += val;
            sumSq += val * val;
            count++;
        }

        mean = sum / count;
        variance = (sumSq / count) - (mean * mean);
    }

    /// <summary>
    /// A reference to a region of an array which can be converted to a span but isn't as restrictive as a ref struct
    /// </summary>
    /// <typeparam name="T">element type</typeparam>
    protected struct SpanSurrogate<T>
    {
        public int Offset { get; init; }
        public int Count { get; init; }
        private T[] underlying;

        public SpanSurrogate(T[] underlying, int offset, int count)
        {
            this.underlying = underlying;
            Offset = offset;
            Count = count;
        }

        public Span<T> AsSpan() => underlying.AsSpan(Offset, Count);
    }

    /// <summary>
    /// Compute the mean and variance of the given values
    /// </summary>
    /// <param name="sets">sets of values</param>
    /// <param name="mean">mean of the values</param>
    /// <param name="variance">variance of the values</param>
    /// <param name="offset">offset to start computing from</param>
    /// <param name="stride">amount to increment by (useful for non-sequential values)</param>
    protected void MeanAndVariance(IList<SpanSurrogate<float>> sets, out float mean, out float variance, int offset = 0, int stride = 1)
    {
        var length = sets.Count;
        if (length == 0)
        {
            mean = 0;
            variance = 0;
            return;
        }

        // Vectorized elements
        int count = 0;                  // Since we can have stride, count may not = length
        float sum = 0, sumSq = 0;       // Sum and sum of squares
        foreach (var set in sets)
        {
            var values = set.AsSpan();
            if (values.Length == 0)
                continue;

            int i = offset;             // Starting index 0 or offset
            if (1 == stride && Vector.IsHardwareAccelerated && Vector<float>.IsSupported)
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
                    count += simdLength;
                }

                // Sum vector lanes
                sum += Vector.Sum(vecSum);
                sumSq += Vector.Sum(vecSumSq);
            }
            // Remaining elements
            for (; i < length; i += stride)
            {
                float val = values[i];
                sum += val;
                sumSq += val * val;
                count++;
            }
        }

        mean = sum / count;
        variance = (sumSq / count) - (mean * mean);
    }
}