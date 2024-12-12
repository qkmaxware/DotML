namespace DotML.Network;

/// <summary>
/// A single filter for a use in a ConvolutionLayer
/// </summary>
public class ConvolutionFilter : List<Matrix<double>> {
    /// <summary>
    /// Filter bias value
    /// </summary>
    public double Bias {get; set;}

    /// <summary>
    /// Width of the filter, max column count of all kernels
    /// </summary>
    public int Width {
        get {
            var max = 0;
            for (var i = 0; i < this.Count; i++) {
                var val = this[i].Columns;
                if (val > max)
                    max = val;
            }
            return max;
        }
    }

    /// <summary>
    /// Height of the filter, max row count of all kernels
    /// </summary>
    public int Height {
        get {
            var max = 0;
            for (var i = 0; i < this.Count; i++) {
                var val = this[i].Rows;
                if (val > max)
                    max = val;
            }
            return max;
        }
    }

    /// <summary>
    /// Create a filter with no kernels
    /// </summary>
    public ConvolutionFilter() { }

    /// <summary>
    /// Create a filter with no kernels, but with a preset kernel capacity
    /// </summary>
    /// <param name="capacity">number of kernels expected</param>
    public ConvolutionFilter(int capacity) : base(capacity) { }
    
    /// <summary>
    /// Create a filter from the given kernels
    /// </summary>
    /// <param name="kernels">kernels</param>
    public ConvolutionFilter(params Matrix<double>[] kernels) : base(kernels) {}

    /// <summary>
    /// Make a bunch of filters with the given number of kernels per filter and kernel size
    /// </summary>
    /// <param name="filters">number of filters</param>
    /// <param name="kernels_per_filter">number of kernels per filter</param>
    /// <param name="kernel_size">size of each kernel (width & height)</param>
    /// <returns>filter list</returns>
    public static ConvolutionFilter[] Make(int filters, int kernels_per_filter, int kernel_size) {
        return Enumerable
            .Range(0, filters)
            .Select(i => 
                new ConvolutionFilter(
                    Enumerable.Range(0, kernels_per_filter)
                    .Select(j => 
                        Kernels.HeKernel(kernel_size)
                    )
                    .ToArray()
                )
            ).ToArray();
    }

}