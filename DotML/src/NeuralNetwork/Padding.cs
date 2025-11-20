namespace DotML.Network;

/// <summary>
/// ConvolutionLayer padding option
/// </summary>
public enum Padding
{
    /// <summary>
    /// Image is kept the same size, padding is used
    /// </summary>
    Same,
    /// <summary>
    /// Image size is reduced, padding is not used
    /// </summary>
    Valid
}

/// <summary>
/// Utility methods for dealing with padding settings
/// </summary>
public static class PaddingExtensions
{
    /// <summary>
    /// Computes the padding values for each side of the input based on the specified padding mode, kernel size, stride, and dilation.
    /// </summary>
    /// <param name="padding">padding kind</param>
    /// <param name="kernel">kernel size</param>
    /// <param name="stride">kernel stride</param>
    /// <param name="dilation">kernel dilation</param>
    /// <returns>left, top, right, bottom padding tuple</returns>
    /// <exception cref="ArgumentOutOfRangeException">when an invalid padding is provided</exception>
    public static (int Left, int Top, int Right, int Bottom) ToTuple(this Padding padding, Size2D kernel, Stride2D stride, Dilation2D dilation)
    {
        return padding switch
        {
            Padding.Same => (
                ((stride.X - 1) + dilation.X * (kernel.Width - 1)) / 2,
                ((stride.Y - 1) + dilation.Y * (kernel.Height - 1)) / 2,
                ((stride.X - 1) + dilation.X * (kernel.Width - 1)) - ((stride.X - 1) + dilation.X * (kernel.Width - 1)) / 2,
                ((stride.Y - 1) + dilation.Y * (kernel.Height - 1)) - ((stride.Y - 1) + dilation.Y * (kernel.Height - 1)) / 2
            ),
            Padding.Valid => (0, 0, 0, 0),
            _ => throw new ArgumentOutOfRangeException(nameof(padding), padding, null)
        };
    }

    /// <summary>
    /// Computes the padding values for each side of the input based on the specified padding mode, kernel size, stride, and dilation.
    /// </summary>
    /// <param name="padding">padding kind</param>
    /// <param name="kernel">kernel size</param>
    /// <param name="stride">kernel stride</param>
    /// <param name="dilation">kernel dilation</param>
    /// <returns>left, top, right, bottom padding tuple</returns>
    /// <exception cref="ArgumentOutOfRangeException">when an invalid padding is provided</exception>
    public static Padding2D ToPadding(this Padding padding, Size2D kernel, Stride2D stride, Dilation2D dilation)
    {
        return padding.ToTuple(kernel, stride, dilation);
    }
}

/// <summary>
/// Convolution layer inverse padding option
/// </summary>
public enum Expansion
{
    /// <summary>
    /// Image is kept the same size, result is not padded
    /// </summary>
    Same,
    /// <summary>
    /// Image size is increased, result is padded
    /// </summary>
    Expand
}