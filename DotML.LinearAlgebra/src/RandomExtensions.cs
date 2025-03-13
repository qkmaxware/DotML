using System.Runtime.CompilerServices;

namespace System;

/// <summary>
/// Utility extensions for System.Random
/// </summary>
public static class RandomExtensions {
    /// <summary>
    /// Generate a random number in a normal distribution
    /// </summary>
    /// <param name="random">random generator</param>
    /// <param name="mean">distribution mean</param>
    /// <param name="stddev">distribution standard deviation</param>
    /// <returns>random value</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double NormalDistribution(this Random random, double mean, double stddev) {
        double u1 = random.NextDouble();
        double u2 = random.NextDouble();
        double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return Math.Abs(mean) + z0 * Math.Abs(stddev);
    }

    /// <summary>
    /// Generate a random number in a uniform distribution
    /// </summary>
    /// <param name="random">random generator</param>
    /// <param name="range">distribution range</param>
    /// <returns>random value</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double UniformDistribution(this Random random, double range) {
        range = Math.Abs(range);
        return (random.NextDouble() * 2 * range) - range;
    }
}