using System.Collections;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Alias for a sequencer of layers
/// </summary>
using LayerSequence = IEnumerable<IFeedforwardNetworkLayer>;

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
/// A function that generates a skip connection
/// </summary>
/// <param name="input_shape">input shape</param>
/// <returns>skip connection</returns>
public delegate SkipConnection SkipConnectionGenerator(Shape3D input_shape);

/// <summary>
/// Layer sequencer to make creating sequences of layers easier by passing input shaped on from one layer to another
/// </summary>
public static class LayerSequencingExtensions {

    /// <summary>
    /// Sequence input shape
    /// </summary>
    public static Shape3D InputShape(this LayerSequence seq) => seq.First().InputShape;
    
    /// <summary>
    /// Sequence output shape
    /// </summary>
    public static Shape3D OutputShape(this LayerSequence seq) => seq.Last().OutputShape;

    #region Initial Layer
    /// <summary>
    /// Begin a sequence of layers starting from this layer
    /// </summary>
    /// <param name="layer">Starting layer</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence BeginSequence(this IFeedforwardNetworkLayer layer) {
        yield return layer;
    }

    /// <summary>
    /// Create a sequence of layers starting with this layer and continuing with the next layer
    /// </summary>
    /// <param name="layer">Starting layer</param>
    /// <param name="next">Next layer</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence Then(this IFeedforwardNetworkLayer layer, IFeedforwardNetworkLayer next) {
        yield return layer;
        yield return next;
    }

    /// <summary>
    /// Create a sequence of layers starting with this layer and continuing with the next layer if the condition is true
    /// </summary>
    /// <param name="layer">Starting layer</param>
    /// <param name="next">Next layer if the condition is true</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence ThenIf(this IFeedforwardNetworkLayer layer, bool condition, IFeedforwardNetworkLayer next) {
        if (condition)
            return Then(layer, next);
        return BeginSequence(layer);
    }

    /// <summary>
    /// Create a sequence of layers starting with this layer and continuing with the next layer
    /// </summary>
    /// <param name="layer">Starting layer</param>
    /// <param name="next">Generator that produces the next layer given a specific input size</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence Then(this IFeedforwardNetworkLayer layer, NetworkLayerGenerator next) {
        yield return layer;
        yield return next(layer.OutputShape);
    }

    /// <summary>
    /// Create a sequence of layers starting with this layer and continuing with the next layer if the condition is true
    /// </summary>
    /// <param name="layer">Starting layer</param>
    /// <param name="next">Generator that produces the next layer given a specific input size</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence ThenIf(this IFeedforwardNetworkLayer layer, bool condition, NetworkLayerGenerator next) {
        yield return layer;
        if (condition)
            yield return next(layer.OutputShape);
    }

    /// <summary>
    /// Create a sequence of layers starting with this layer and continuing with the next layers
    /// </summary>
    /// <param name="seq">initial sequence</param>
    /// <param name="next">next layers</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence Then(this IFeedforwardNetworkLayer layer, LayerSequence next) {
        yield return layer;
        foreach (var l in next) {
            yield return l;
        }
    }

    /// <summary>
    /// Create a sequence of layers starting with this layer and continuing with the next layers if the condition is true
    /// </summary>
    /// <param name="seq">initial sequence</param>
    /// <param name="next">next layers</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence ThenIf(this IFeedforwardNetworkLayer layer, bool condition, LayerSequence next) {
        yield return layer;
        if (condition) {
            foreach (var l in next) {
                yield return l;
            }
        }
    }

    /// <summary>
    /// Create a sequence of layers starting with this layer and continuing with the next layers
    /// </summary>
    /// <param name="layer">Starting layer</param>
    /// <param name="next">Generator that produces subsequent layers given a specific input size for the next layer</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence Then(this IFeedforwardNetworkLayer layer, NetworkBlockGenerator next) {
        yield return layer;
        foreach (var nextLayer in next(layer.OutputShape)) {
            yield return nextLayer;
        }
    }

    /// <summary>
    /// Create a sequence of layers starting with this layer and continuing with the next layers if the condition is true
    /// </summary>
    /// <param name="layer">Starting layer</param>
    /// <param name="next">Generator that produces subsequent layers given a specific input size for the next layer</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence ThenIf(this IFeedforwardNetworkLayer layer, bool condition, NetworkBlockGenerator next) {
        yield return layer;
        if (condition) {
            foreach (var nextLayer in next(layer.OutputShape)) {
                yield return nextLayer;
            }
        }
    }

    /// <summary>
    /// Create a sequence of layers followed by a block of layers repeated the given number of times
    /// </summary>
    /// <param name="layer">Starting layer</param>
    /// <param name="block">Generator that produces subsequent layers given a specific input size for the next layer</param>
    /// <param name="repetitions">Number of times to repeat the block</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence ThenBlock(this IFeedforwardNetworkLayer layer, NetworkBlockGenerator block, int repetitions = 1) {
        yield return layer;
        var shape = layer.OutputShape;

        for (var i = 0; i < repetitions; i++) {
            foreach (var ilayer in block(shape)) {
                yield return ilayer;
                shape = ilayer.OutputShape;
            }
        }
    }

    /// <summary>
    /// Create a sequence of layers followed by a block of layers repeated the given number of times. The block is bookend by an input capture and a skip connection.
    /// </summary>
    /// <param name="layer">Starting layer</param>
    /// <param name="block">Generator that produces the block body</param>
    /// <param name="connection">Skip connection generator</param>
    /// <param name="repetitions">Number of times to repeat the block</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence ThenSkipBlock(this IFeedforwardNetworkLayer layer, NetworkBlockGenerator block, SkipConnectionGenerator connection, int repetitions = 1) {
        yield return layer;
        var shape = layer.OutputShape;

        for (var i = 0; i < repetitions; i++) {
            // Begin with an input capture
            var capture = new InputCapture(shape);
            yield return capture;

            // Do the body of the block
            foreach (var ilayer in block(shape)) {
                yield return ilayer;
                shape = ilayer.OutputShape;
            }

            // Create the residual connection
            var residual = connection(shape);
            yield return residual;
            shape = residual.OutputShape;
        }
    }

    /// <summary>
    /// Execute an activation function immediately after this layer
    /// </summary>
    /// <param name="layer">Starting layer</param>
    /// <param name="fn">Activation function to invoke</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence WithActivation(this IFeedforwardNetworkLayer layer, ActivationFunction fn) {
        yield return layer;
        yield return new ActivationLayer(layer.OutputShape, fn);
    }

    /// <summary>
    /// Perform dropout immediately after this layer
    /// </summary>
    /// <param name="layer">Starting layer</param>
    /// <param name="amount">Dropout normalized percent</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence WithDropout(this IFeedforwardNetworkLayer layer, float amount) {
        yield return layer;
        yield return new DropoutLayer(layer.OutputShape, amount);
    }
    #endregion

    #region Subsequent Layers
    /// <summary>
    /// Add a layer to the sequence
    /// </summary>
    /// <param name="seq">initial sequence</param>
    /// <param name="next">next layer</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence Then(this LayerSequence seq, IFeedforwardNetworkLayer next) {
        foreach (var layer in seq) {
            yield return layer;
        }
        yield return next;
    }

    /// <summary>
    /// Add a layer to the sequence if the condition is true
    /// </summary>
    /// <param name="seq">initial sequence</param>
    /// <param name="next">next layer</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence ThenIf(this LayerSequence seq, bool condition, IFeedforwardNetworkLayer next) {
        foreach (var layer in seq) {
            yield return layer;
        }
        if (condition)
            yield return next;
    }

    /// <summary>
    /// Add a layer to the sequence
    /// </summary>
    /// <param name="seq">initial sequence</param>
    /// <param name="next">generator that produces the next layer with the given input size</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence Then(this LayerSequence seq, NetworkLayerGenerator next) {
        Shape3D shape = new Shape3D();
        foreach (var layer in seq) {
            yield return layer;
            shape = layer.OutputShape;
        }

        yield return next(shape);
    }

    /// <summary>
    /// Add a layer to the sequence if the condition is true
    /// </summary>
    /// <param name="seq">initial sequence</param>
    /// <param name="next">generator that produces the next layer with the given input size</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence ThenIf(this LayerSequence seq, bool condition, NetworkLayerGenerator next) {
        Shape3D shape = new Shape3D();
        foreach (var layer in seq) {
            yield return layer;
            shape = layer.OutputShape;
        }

        if (condition)
            yield return next(shape);
    }

    /// <summary>
    /// Add several layers to the sequence
    /// </summary>
    /// <param name="seq">initial sequence</param>
    /// <param name="next">next layers</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence Then(this LayerSequence seq, LayerSequence next) {
        foreach (var layer in seq) {
            yield return layer;
        }
        foreach (var layer in next) {
            yield return layer;
        }
    }

    /// <summary>
    /// Add several layers to the sequence if the condition is true
    /// </summary>
    /// <param name="seq">initial sequence</param>
    /// <param name="next">next layers</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence ThenIf(this LayerSequence seq, bool condition, LayerSequence next) {
        foreach (var layer in seq) {
            yield return layer;
        }
        if (condition) {
            foreach (var layer in next) {
                yield return layer;
            }
        }
    }

    /// <summary>
    /// Add several layers to the sequence
    /// </summary>
    /// <param name="seq">initial sequence</param>
    /// <param name="next">next layer generator starting from the given input size</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence Then(this LayerSequence seq, NetworkBlockGenerator next) {
        Shape3D shape = new Shape3D();
        foreach (var layer in seq) {
            yield return layer;
            shape = layer.OutputShape;
        }

        foreach (var layer in next(shape)) {
            yield return layer;
        }
    }

    /// <summary>
    /// Add several layers to the sequence of the condition is true
    /// </summary>
    /// <param name="seq">initial sequence</param>
    /// <param name="next">next layer generator starting from the given input size</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence ThenIf(this LayerSequence seq, bool condition, NetworkBlockGenerator next) {
        Shape3D shape = new Shape3D();
        foreach (var layer in seq) {
            yield return layer;
            shape = layer.OutputShape;
        }

        if (condition) {
            foreach (var layer in next(shape)) {
                yield return layer;
            }
        }
    }

    /// <summary>
    /// Create a sequence of layers followed by a block of layers repeated the given number of times
    /// </summary>
    /// <param name="seq">initial sequence</param>
    /// <param name="block">generator that produces subsequent layers given a specific input size for the next layer</param>
    /// <param name="repetitions">Number of times to repeat the block</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence ThenBlock(this LayerSequence seq, NetworkBlockGenerator block, int repetitions = 1) {
        Shape3D shape = new Shape3D();
        foreach (var layer in seq) {
            yield return layer;
            shape = layer.OutputShape;
        }

        for (var i = 0; i < repetitions; i++) {
            foreach (var layer in block(shape)) {
                yield return layer;
                shape = layer.OutputShape;
            }
        }
    }

    /// <summary>
    /// Create a sequence of layers followed by a block of layers repeated the given number of times. The block is bookend by an input capture and a skip connection.
    /// </summary>
    /// <param name="layer">Starting layer</param>
    /// <param name="block">Generator that produces the block body</param>
    /// <param name="connection">Skip connection generator</param>
    /// <param name="repetitions">Number of times to repeat the block</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence ThenSkipBlock(this LayerSequence seq, NetworkBlockGenerator block, SkipConnectionGenerator connection, int repetitions = 1) {
        Shape3D shape = new Shape3D();
        foreach (var layer in seq) {
            yield return layer;
            shape = layer.OutputShape;
        }

        for (var i = 0; i < repetitions; i++) {
            // Begin with an input capture
            var capture = new InputCapture(shape);
            yield return capture;

            // Do the body of the block
            foreach (var ilayer in block(shape)) {
                yield return ilayer;
                shape = ilayer.OutputShape;
            }

            // Create the residual connection
            var residual = connection(shape);
            yield return residual;
            shape = residual.OutputShape;
        }
    }

    /// <summary>
    /// Add an activation function to the end of the sequence
    /// </summary>
    /// <param name="seq">layer sequence</param>
    /// <param name="fn">activation function</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence WithActivation(this LayerSequence seq, ActivationFunction fn) {
        Shape3D shape = new Shape3D();
        foreach (var layer in seq) {
            yield return layer;
            shape = layer.OutputShape;
        }
        yield return new ActivationLayer(shape, fn);
    }

    /// <summary>
    /// Add a dropout action to the end of the sequence
    /// </summary>
    /// <param name="layer">Starting layer</param>
    /// <param name="amount">Dropout normalized percent</param>
    /// <returns>layer sequence</returns>
    public static LayerSequence WithDropout(this LayerSequence seq, float amount) {
        Shape3D shape = new Shape3D();
        foreach (var layer in seq) {
            yield return layer;
            shape = layer.OutputShape;
        }
        yield return new DropoutLayer(shape, amount);
    }
    #endregion
}

// Example
/*
// Sequence already made objects
layer1.Then(layer2).Then(layer3);

// Conditionally include layers
layer1.ThenIf(this.include_dropout, new DropoutLayer(0.25));

// Sequence a new object, ensure the size is compatible
layer1.Then(i => new Layer(i));

// Sequence already made objects, but add an activation function
layer1.Then(layer2.WithActivation(ReLU.Instance).WithDropout(0.25));

// Sequence a new object, but add an activation function
layer2.Then(i => new Layer(i).WithActivation(ReLU.Instance).WithDropout(0.25));
*/