using System.Collections;

namespace DotML.Network;

/// <summary>
/// A single filter for a use in a ConvolutionLayer
/// </summary>
public class ConvolutionFilter : IEnumerable<Matrix<double>> {

    private Matrix<double>[] kernels;

    /// <summary>
    /// Shape of the filter
    /// </summary>
    public Shape3D Shape => new Shape3D(Count, Height, Width);

    /// <summary>
    /// Number of kernels in this filter
    /// </summary>
    public int Count => kernels.Length;

    /// <summary>
    /// Filter bias value
    /// </summary>
    public double Bias {get; set;}

    /// <summary>
    /// Get or set a kernel value, dimensions must match filter size
    /// </summary>
    /// <param name="index">kernel index</param>
    /// <returns>kernel matrix</returns>
    /// <exception cref="ArgumentException">Thrown when kernel dimensions are mismatched</exception>
    public Matrix<double> this[int index] {
        get => kernels[index];
        set {
            if (value.Rows != this.Height || value.Columns != this.Width)
                throw new ArgumentException("Invalid kernel dimensions for filter");
            kernels[index] = value;
        }
    }

    /// <summary>
    /// Width of the filter, max column count of all kernels
    /// </summary>
    public int Width {
        get; private set;
    }

    /// <summary>
    /// Height of the filter, max row count of all kernels
    /// </summary>
    public int Height {
        get; private set;
    }

    
    /// <summary>
    /// Create a filter from the given kernels
    /// </summary>
    /// <param name="kernels">kernels</param>
    public ConvolutionFilter(params Matrix<double>[] kernels) {
        this.Width = kernels.Select(x => x.Columns).Max();
        this.Height = kernels.Select(x => x.Rows).Max();

        this.kernels = kernels;
    }

    /// <summary>
    /// Make a bunch of filters with the given number of kernels per filter and kernel size
    /// </summary>
    /// <param name="filters">number of filters</param>
    /// <param name="kernels_per_filter">number of kernels per filter</param>
    /// <param name="kernel_size">size of each kernel (width & height)</param>
    /// <returns>filter list</returns>
    public static ConvolutionFilter[] Make(int filters, int kernels_per_filter, int kernel_size) {
        kernels_per_filter = Math.Max(0, kernels_per_filter);
        filters = Math.Max(0, filters);

        var objs = new ConvolutionFilter[filters]; 
        for (var i = 0; i < objs.Length; i++) {
            var kernels = new Matrix<double>[kernels_per_filter];
            for (var j = 0; j < kernels_per_filter; j++) {
                kernels[j] = Kernels.HeKernel(kernel_size);
            }
            var filter = new ConvolutionFilter(kernels);
            objs[i] = filter;
        }
        return objs;
    }

    public IEnumerator<Matrix<double>> GetEnumerator() {
        return ((IEnumerable<Matrix<double>>)kernels).GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator() {
        return kernels.GetEnumerator();
    }
}