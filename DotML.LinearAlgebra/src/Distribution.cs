using System.Numerics;

namespace DotML;

/// <summary>
/// Represents a probability distribution for random number sampling.
/// </summary>
/// <typeparam name="T">The type of value produced by the distribution.</typeparam>
public interface IProbabilityDistribution<T>
{
    /// <summary>
    /// Samples a random value from the distribution.
    /// </summary>
    /// <returns>A random value following the distribution.</returns>
    T Sample();

    /// <summary>
    /// The expected value (mean) of the distribution.
    /// </summary>
    public T Mean { get; }
}

/// <summary>
/// Static class containing some common probability distributions
/// </summary>
public static class Distributions
{
    /// <summary>
    /// Create a uniform probability distribution over the provided range
    /// </summary>
    /// <typeparam name="T">numeric type</typeparam>
    /// <param name="min">range minimum</param>
    /// <param name="max">range maximum</param>
    /// <returns>UniformDistribution</returns>
    public static UniformDistribution<T> Uniform<T>(double min, double max)
    where T : INumber<T>
    => new UniformDistribution<T>(min, max);

    /// <summary>
    /// Create a normal probability distribution with the given mean and standard deviation
    /// </summary>
    /// <typeparam name="T">numeric type</typeparam>
    /// <param name="mean">mean value</param>
    /// <param name="stdDev">standard deviation</param>
    /// <returns>NormalDistribution</returns>
    public static NormalDistribution<T> Normal<T>(double mean, double stdDev)
    where T : INumber<T>
    => new NormalDistribution<T>(mean, stdDev);
}

/// <summary>
/// A uniform distribution of values between a range defined by a minimum and maximum value
/// </summary>
/// <typeparam name="T">numeric type</typeparam>
public class UniformDistribution<T> : IProbabilityDistribution<T>
where T : INumber<T>
{
    private readonly double _min;
    private readonly double _max;
    private readonly Random _random;

    public UniformDistribution(double min, double max, Random? random = null)
    {
        this._min = Math.Min(min, max);
        this._max = Math.Max(min, max);

        this._random = random ?? new Random();
    }

    public T Sample()
    {
        return T.CreateSaturating(_min + (_max - _min) * _random.NextDouble());
    }

    public T Mean => T.CreateSaturating((_min + _max) / 2.0);
}

/// <summary>
/// A normal distribution of values defined by a mean and standard deviation
/// </summary>
/// <typeparam name="T">numeric type</typeparam>
public class NormalDistribution<T> : IProbabilityDistribution<T>
where T : INumber<T>
{
    private readonly double _mean;
    private readonly T _tmean;
    private readonly double _stdDev;
    private readonly Random _random;

    public NormalDistribution(double mean, double stdDev, Random? random = null)
    {
        if (stdDev <= 0)
            throw new ArgumentException("Standard deviation must be positive.");

        _mean = mean;
        _tmean = T.CreateSaturating(_mean);
        _stdDev = stdDev;
        _random = random ?? new Random();
    }

    public T Sample()
    {
        // Box-Muller transform
        double u1 = _random.NextDouble();
        double u2 = _random.NextDouble();

        double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        double value = _mean + z0 * _stdDev;

        return T.CreateSaturating(value);
    }

    public T Mean => _tmean;
}

/// <summary>
/// A bernoulli distribution with values either of 0 or 1 depending on the probability of sampling
/// </summary>
/// <typeparam name="T">numeric type</typeparam>
public class BernoulliDistribution<T> : IProbabilityDistribution<T>
where T : INumber<T>
{
    private readonly double _p;
    private readonly Random _random;

    public BernoulliDistribution(double p, Random? random = null)
    {
        _p = Math.Clamp(p, 0.0, 1.0);
        _random = random ?? new Random();
    }

    public T Sample() => _random.NextDouble() < _p ? T.One : T.Zero;

    public T Mean => T.CreateSaturating(_p);
}

/// <summary>
/// A poisson distribution over number of events in a fixed interval
/// </summary>
/// <typeparam name="T">numeric type</typeparam>
public class PoissonDistribution<T> : IProbabilityDistribution<T>
where T : INumber<T>
{
    private readonly double _lambda;
    private readonly Random _random;

    public PoissonDistribution(double lambda, Random? random = null)
    {
        if (lambda <= 0)
            throw new ArgumentOutOfRangeException(nameof(lambda), "Lambda must be positive.");

        _lambda = lambda;
        _random = random ?? new Random();
    }

    public T Sample()
    {
        double L = Math.Exp(-_lambda);
        int k = 0;
        double p = 1.0;

        do
        {
            k++;
            p *= _random.NextDouble();
        } while (p > L);

        return T.CreateSaturating(k - 1);
    }

    public T Mean => T.CreateSaturating(_lambda);
}


