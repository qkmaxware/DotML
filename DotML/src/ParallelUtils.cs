using System.Numerics;
using System.Runtime.CompilerServices;
using DotML;

/// <summary>
/// Id of a job 
/// </summary>
internal record struct JobId {
    /// <summary>
    /// Job index in the X axis
    /// </summary>
    public readonly int X; 
    /// <summary>
    /// Job index in the Y axis
    /// </summary>
    public readonly int Y;

    /// <summary>
    /// Alias for Y
    /// </summary>
    public readonly int Row {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Y;
    }

    /// <summary>
    /// Alias for X
    /// </summary>
    public readonly int Column {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => X;
    }

    public JobId(int x, int y) {
        this.X = x;
        this.Y = y;
    }
}

/// <summary>
/// A job that can be run in parallel
/// </summary>
/// <param name="id"job id</param>
internal delegate void ParallelJob(JobId id);

/// <summary>
/// Utilities for running in parallel (wrapper for System.Threading.Tasks.Parallel)
/// </summary>
internal static class ParallelUtils {

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ForX(int xFromInclusive, int xToExclusive, ParallelJob job) {
        Parallel.For(xFromInclusive, xToExclusive, (index) => {
            // Create jobid
            var id = new JobId(index, 0);

            // Run job
            job(id);
        });
    }  

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ForEach<T>(Vec<T> vector, ParallelJob job) where T:INumber<T>,IExponentialFunctions<T>,IRootFunctions<T> {
        ForX(0, vector.Dimensionality, job);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ForXY(int xFromInclusive, int xToExclusive, int yFromInclusive, int yToExclusive, ParallelJob job) {
        var yrange = yToExclusive - yFromInclusive;
        var xrange = xToExclusive - xFromInclusive;
        var count = xrange * yrange;

        // By iterating only over one axis, there are no divisions for computing the index, though it's not really "parallel" for both axis
        Parallel.For(0, yrange, (yPrime) => {
            var y = yPrime + yFromInclusive;
            for (var xPrime = 0; xPrime < xrange; xPrime++) {
                var x = xPrime + xFromInclusive;

                var id = new JobId(x, y);

                job(id);
            }
        });

        /*Parallel.For(0, count, (index) => {
            // Create jobid
            var y = index / xrange + yFromInclusive;
            var x = index % xrange + xFromInclusive;

            var id = new JobId(x, y);

            // Run job
            job(id);
        });*/
    }     

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ForEach<T>(Matrix<T> matrix, ParallelJob job) where T:INumber<T>,IExponentialFunctions<T>,IRootFunctions<T> {
        ForXY(0, matrix.Columns, 0, matrix.Rows, job);
    }

}