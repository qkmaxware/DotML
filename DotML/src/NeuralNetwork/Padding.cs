namespace DotML.Network;

/// <summary>
/// ConvolutionLayer padding option
/// </summary>
public enum Padding {
    /// <summary>
    /// Image is kept the same size, padding is used
    /// </summary>
    Same,
    /// <summary>
    /// Image size is reduced, padding is not used
    /// </summary>
    Valid
}