using System.Collections;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// A function that can generate a single network layer with the given input size
/// </summary>
/// <param name="input_shape">input shape</param>
/// <returns>network layer</returns>
public delegate IFeedforwardNetworkLayer NetworkLayerGenerator(Shape3D input_shape);

/// <summary>
/// A function that can generate a block of several network layers with the given input size
/// </summary>
/// <param name="input_shape">input shape</param>
/// <returns>enumerable of layers</returns>
public delegate IEnumerable<IFeedforwardNetworkLayer> NetworkBlockGenerator(Shape3D input_shape);

/// <summary>
/// Layer sequencer to make creating sequences of layers easier by passing input shaped on from one layer to another
/// </summary>
public class LayerSequencer : IEnumerable<IFeedforwardNetworkLayer> {
    private Shape3D ishape;
    private List<IFeedforwardNetworkLayer> layers = new List<IFeedforwardNetworkLayer>();

    /// <summary>
    /// Sequence input shape
    /// </summary>
    public Shape3D InputShape => layers.Count > 0 ? layers[0].InputShape : ishape;
    
    /// <summary>
    /// Sequence output shape
    /// </summary>
    public Shape3D OutputShape => layers.Count > 0 ? layers[^1].OutputShape : ishape;

    /// <summary>
    /// Create a new layer sequencer with the given initial shape
    /// </summary>
    /// <param name="input_size">initial shape</param>
    public LayerSequencer(Shape3D input_size) {
        this.ishape = input_size;
    } 

    /// <summary>
    /// Create a new layer sequencer with the given first layer
    /// </summary>
    /// <param name="first">first layer</param>
    public LayerSequencer(IFeedforwardNetworkLayer first) {
        this.layers.Add(first);
    }

    /// <summary>
    /// Append another layer to this sequence
    /// </summary>
    /// <param name="constructor">A method that creates a layer with the given input size</param>
    /// <returns>this</returns>
    public LayerSequencer Then(NetworkLayerGenerator constructor) {
        this.layers.Add(constructor(OutputShape));
        return this;
    }

    /// <summary>
    /// Append a block of layers to this sequence
    /// </summary>
    /// <param name="generator">A method that creates many layers with the given input size</param>
    /// <returns>this</returns>
    public LayerSequencer Then(NetworkBlockGenerator generator) {
        this.layers.AddRange(generator(OutputShape));
        return this;
    }

    public IEnumerator<IFeedforwardNetworkLayer> GetEnumerator() => layers.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => layers.GetEnumerator();
}