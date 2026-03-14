namespace DotML.Network.Embedding;

/// <summary>
/// Interface representing the behaviour of an image
/// </summary>
/// <typeparam name="TPixel">pixel type</typeparam>
public interface IImage<TPixel> 
{
    public int Width {get;}
    public int Height {get;}
    public TPixel this[int x, int y] {get;}
}

/// <summary>
/// Behaviour of an object that can decode an image like structure from a stream
/// </summary>
/// <typeparam name="TPixel">pixel type</typeparam>
public interface IImageDecoder<TPixel>
{
    /// <summary>
    /// Decode an image from the provided stream
    /// </summary>
    /// <param name="imageData">
    /// Stream containing encoded image data. The stream is read from its current position.
    /// The stream is not closed by the decoder.
    /// </param>
    public IImage<TPixel> Decode(Stream imageData);
}