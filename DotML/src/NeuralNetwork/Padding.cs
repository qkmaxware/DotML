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
    public static (int Left, int Top, int Right, int Bottom) ToTuple(this Padding padding, (int height, int width) kernel, (int height, int width) stride, (int height, int width) dilation)
    {
        return padding switch
        {
            Padding.Same => (
                ((stride.width - 1) + dilation.width * (kernel.width - 1)) / 2,
                ((stride.height - 1) + dilation.height * (kernel.height - 1)) / 2,
                ((stride.width - 1) + dilation.width * (kernel.width - 1)) - ((stride.width - 1) + dilation.width * (kernel.width - 1)) / 2,
                ((stride.height - 1) + dilation.height * (kernel.height - 1)) - ((stride.height - 1) + dilation.height * (kernel.height - 1)) / 2
            ),
            Padding.Valid => (0, 0, 0, 0),
            _ => throw new ArgumentOutOfRangeException(nameof(padding), padding, null)
        };
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