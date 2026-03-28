using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// A simple network composed of sequential layers with no skip connections, residuals, or branches
/// <para>
/// <code>
/// --> 1st --> 2nd --> ... --> Nth -->
/// </code>
/// </para>
/// </summary>
public class SequentialBlock : INetworkModule, IBlockVisitable
{
    public string? Alias { get; set; }

    private List<INetworkModule> layers;

    public SequentialBlock()
    {
        layers = new List<INetworkModule>();
    }

    public SequentialBlock(int capacity)
    {
        layers = new List<INetworkModule>(capacity);
    }

    public SequentialBlock(IEnumerable<INetworkModule> modules)
    {
        layers = new List<INetworkModule>(modules);
    }

    public static SequentialBlockBuilder Begin(Shape ishape) => new SequentialBlockBuilder(ishape);

    public void Add(INetworkModule module) => layers.Add(module);
    public void Remove(INetworkModule module) => layers.Remove(module);
    public void RemoveAt(int index) => layers.RemoveAt(index);
    public void Replace(INetworkModule module, INetworkModule replacement)
    {
        for (var i = 0; i < layers.Count; i++)
        {
            if (layers[i] == module)
                layers[i] = replacement;
        }
    }
    public void InsertBefore(INetworkModule module, INetworkModule inserted)
    {
        var ind = this.layers.IndexOf(module);
        if (ind >= 0)
            this.layers.Insert(ind, inserted);
    }
    public void InsertAfter(INetworkModule module, INetworkModule inserted)
    {
        var ind = this.layers.IndexOf(module);
        if (ind >= 0)
            this.layers.Insert(ind + 1, inserted);
    }

    public INetworkModule this[int index] => layers[index];

    /// <summary>
    /// Number of submodules contained within this module
    /// </summary>
    public int SubmoduleCount => layers.Sum(l => l.SubmoduleCount) + LayerCount;

    /// <summary>
    /// Number of layers
    /// </summary>
    public int LayerCount => layers.Count;

    /// <summary>
    /// Reference to the first layer in the network
    /// </summary>
    /// <returns>layer</returns>
    public INetworkModule GetFirstLayer() => layers[0];

    /// <summary>
    /// Get a specific layer by index
    /// </summary>
    /// <param name="index">index of layer</param>
    /// <returns>layer</returns>
    public INetworkModule GetLayer(int index) => layers[index];

    /// <summary>
    /// Reference to the output layer of the network
    /// </summary>
    /// <returns>layer</returns>
    public INetworkModule GetOutputLayer() => layers[^1];

    /// <summary>
    /// Initialize network weights and biases
    /// </summary>
    /// <param name="initializer">initialization strategy</param>
    public void Initialize(IInitializer initializer)
    {
        foreach (var layer in this.layers)
        {
            layer.Initialize(initializer);
        }
    }

    public Shape ForwardShapeUntil(Shape input, int layer)
    {
        var i = input;
        for (var index = 0; index < Math.Min(layer + 1, layers.Count); index++)
        {
            var o = layers[index].ForwardShape(i);
            i = o;
        }
        return i;
    }

    public Shape ForwardShape(Shape input)
    {
        var i = input;
        foreach (var layer in layers)
        {
            var o = layer.ForwardShape(i);
            i = o;
        }
        return i;
    }

    /// <summary>
    /// Evaluate the sequential network using the provided input
    /// </summary>
    /// <param name="input">input tensor of shape [N,C,H,W]</param>
    /// <param name="ctx">optional evaluation context to store intermediary tensors</param>
    /// <returns>output tensor of shape [N,C,H,W]</returns>
    public Tensor<float> Forward(Tensor<float> input, EvaluationContext? ctx = null, ISteppedProgress? progress = null)
    {
        Tensor<float> i = input;
        foreach (var layer in layers)
        {
            var o = layer.Forward(i, ctx, progress);
            i = o;
        }

        progress?.Advance(steps: 1);
        return i;
    }

    /// <summary>
    /// Perform a backwards evaluation of the network 
    /// </summary>
    /// <param name="dy">gradient of the output</param>
    /// <param name="ctx">the evaluation context to retrieve intermediary tensors from</param>
    /// <returns>gradient of the input as well as gradients for each layer in a list</returns>
    public Gradients Backward(Tensor<float> dy, EvaluationContext ctx, ILocalClippingStrategy<float>? clipping = null)
    {
        var subGrad = new Gradients[layers.Count];
        for (var i = 0; i < subGrad.Length; i++)
        {
            var j = layers.Count - 1 - i;
            var layer = layers[j];
            var grad = layer.Backward(dy, ctx, clipping);
            dy = grad.dX;
            subGrad[j] = grad;
        }
        return new GradientList(dy, subGrad);
    }

    public void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)
    {
        if (gradients is not GradientList lst)
            throw new ArgumentException("Expected a GradientList object", nameof(gradients));

        for (var i = 0; i < layers.Count; i++)
        {
            var layer = layers[i];
            layer.Update(learningRate, lst.dN(i), optimizer, regularization);
        }
    }

    public IEnumerable<INetworkModule> AsEnumerable()
    {
        for (var i = 0; i < layers.Count; i++)
        {
            yield return layers[i];
        } 
    }

    public TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg) => visitor.Visit(this, arg);
    
}

public class SequentialBlockBuilder
{
    public Shape InputShape { get; init; }
    private Shape outputshape;
    private SequentialBlock block = new SequentialBlock();

    public SequentialBlockBuilder(Shape ishape)
    {
        this.InputShape = ishape;
        this.outputshape = ishape;
    }
    public SequentialBlock Finalize() => block;

    public SequentialBlockBuilder Then(INetworkModule layer)
    {
        block.Add(layer);
        outputshape = layer.ForwardShape(outputshape);
        return this;
    }

    public SequentialBlockBuilder Then(Func<Shape, INetworkModule> layerFactory)
    {
        var layer = layerFactory(outputshape);
        block.Add(layer);
        outputshape = layer.ForwardShape(outputshape);
        return this;
    }
    
    public SequentialBlockBuilder ThenIf(bool condition, INetworkModule layer)
    {
        if (condition)
        {
            block.Add(layer);
            outputshape = layer.ForwardShape(outputshape);
        }
        return this;
    }

    public SequentialBlockBuilder ThenIf(bool condition, Func<Shape, INetworkModule> layerFactory)
    {
        if (condition)
        {
            var layer = layerFactory(outputshape);
            block.Add(layer);
            outputshape = layer.ForwardShape(outputshape);
        }
        return this;
    }

    public SequentialBlockBuilder WithActivation(ActivationFunction activation)
    {
        var act = new Activation(activation);
        block.Add(act);
        // No need to recompute outputshape, activation does not change shape
        return this;
    }

    public SequentialBlockBuilder WithDropout(float dropoutRate)
    {
        var act = new Dropout(dropoutRate);
        block.Add(act); 
        // No need to recompute outputshape, Dropout does not change shape
        return this;
    }
}