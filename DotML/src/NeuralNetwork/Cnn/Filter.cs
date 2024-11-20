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
    public int Width => this.Select(x => x.Columns).Max();
    /// <summary>
    /// Height of the filter, max row count of all kernels
    /// </summary>
    public int Height => this.Select(x => x.Rows).Max();

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

}