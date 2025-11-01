using System.Numerics;

namespace DotML;

/// <summary>
/// A simple metric that records statistical information about the metric collected over time
/// </summary>
/// <typeparam name="T">metric type</typeparam>
public class Metric<T>
where T : INumber<T>
{
    /// <summary>
    /// Sum of all samples of this metric
    /// </summary>
    public T Sum;
    /// <summary>
    /// Number of samples taken for this metric
    /// </summary>
    private int Count;
    /// <summary>
    /// Average of all samples of this metric: <see cref="Mean"/>
    /// </summary>
    public T Average => Mean;
    /// <summary>
    /// Maximum value of this metric
    /// </summary>
    public T Max;
    /// <summary>
    /// Minimum value of this metric
    /// </summary>
    public T Min;

    /// <summary>
    /// First recorded value of this metric
    /// </summary> 
    public T First;
    /// <summary>
    /// Last/most recent recorded value of this metric
    /// </summary>
    public T Last;

    /// <summary>
    /// Running mean used by Welford's algorithm
    /// </summary>
    public T Mean;
    /// <summary>
    /// Sum of squared differences from the running mean (M2 in Welford)
    /// </summary>
    private T M2;

    /// <summary>
    /// Sample variance (uses sample variance: M2 / (n - 1)). Returns zero if fewer than 2 samples.
    /// </summary>
    public T Variance => Count < 2 ? T.Zero : M2 / T.CreateChecked(Count - 1);

    /// <summary>
    /// Standard deviation (sqrt of Variance). Returns zero if fewer than 2 samples.
    /// </summary>
    public T StandardDeviation
    {
        get
        {
            if (Count < 2) 
                return T.Zero;
            // convert to double for sqrt, clamp small negative rounding errors
            var varDouble = Math.Max(0.0, Convert.ToDouble(Variance));
            return T.CreateChecked(Math.Sqrt(varDouble));
        }
    }

    public Metric()
    {
        this.Sum = T.Zero;
        this.Max = T.Zero;
        this.Min = T.Zero;
        this.Count = 0;

        First = T.Zero;
        Last = T.Zero;

        Mean = T.Zero;
        M2 = T.Zero;
    }

    /// <summary>
    /// Reset the metric's statistics
    /// </summary>
    public void Reset()
    {
        this.Sum = T.Zero;
        this.Max = T.Zero;
        this.Min = T.Zero;
        this.Count = 0;

        First = T.Zero;
        Last = T.Zero;

        Mean = T.Zero;
        M2 = T.Zero;
    }   

    /// <summary>
    /// Add a new sample to this metric
    /// </summary>
    /// <param name="sample">sample value</param>
    public void Add(T sample)
    {
        if (Count == 0)
        {
            Min = Max = sample;
            First = sample;

            // initialize running values
            Mean = sample;
            M2 = T.Zero;
        }
        else
        {
            Min = T.Min(Min, sample);
            Max = T.Max(Max, sample);
        }

        Last = sample;

        Sum += sample;
        Count++;

        // Welford's online algorithm for mean
        if (Count == 1)
        {
            Mean = sample;
            M2 = T.Zero;
        }
        else
        {
            // delta = x - mean_old
            T delta = sample - Mean;
            // mean_new = mean_old + delta / n
            Mean += delta / T.CreateChecked(Count);
            // M2 += delta * (x - mean_new)
            T delta2 = sample - Mean;
            M2 += delta * delta2;
        }
    }
}