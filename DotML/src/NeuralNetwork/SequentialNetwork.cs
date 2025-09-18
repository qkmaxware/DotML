using DotML.Network.Initialization;

namespace DotML.Network;

public interface IModuleContext
{
    Tensor<float> Input { get; }
    Tensor<float> Output { get; }
}

public class IOContext : IModuleContext
{
    public Tensor<float> Input { get; init; }
    public Tensor<float> Output { get; init; }

    public IOContext(Tensor<float> input, Tensor<float> output)
    {
        Input = input;
        Output = output;
    }
}

public class EvaluationContext
{
    private Dictionary<object, IModuleContext> _storage = new Dictionary<object, IModuleContext>();

    public void Save(object module, IModuleContext context)
    {
        _storage[module] = context;
    }

    public TCtx Get<TCtx>(object module)
    where TCtx : IModuleContext
    {
        if (_storage.TryGetValue(module, out var ctx) &&
            ctx is TCtx typedCtx)
        {
            return typedCtx;
        }

        throw new KeyNotFoundException($"No context saved for module '{module}' of type '{typeof(TCtx)}'");
    }

}

/// <summary>
/// A simple network composed of sequential layers with no skip connections, residuals, or branches
/// </summary>
public class SequentialNetwork
{
    private List<NetworkLayer> layers = new List<NetworkLayer>();

    /// <summary>
    /// Number of layers
    /// </summary>
    public int LayerCount => layers.Count;

    /// <summary>
    /// Reference to the first layer in the network
    /// </summary>
    /// <returns>layer</returns>
    public NetworkLayer GetFirstLayer() => layers[0];

    /// <summary>
    /// Get a specific layer by index
    /// </summary>
    /// <param name="index">index of layer</param>
    /// <returns>layer</returns>
    public NetworkLayer GetLayer(int index) => layers[index];

    /// <summary>
    /// Reference to the output layer of the network
    /// </summary>
    /// <returns>layer</returns>
    public NetworkLayer GetOutputLayer() => layers[^1];

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

    /// <summary>
    /// Number of trainable parameters in this network
    /// </summary>
    /// <returns>Number of trainable parameters</returns>
    public int TrainableParameterCount() => this.layers.Sum(layer => layer.TrainableParameterCount());

    /// <summary>
    /// Number of un-trainable parameters in this network
    /// </summary>
    /// <returns>Number of trainable parameters</returns>
    public int UnTrainableParameterCount() => this.layers.Sum(layer => layer.UnTrainableParameterCount());

    /// <summary>
    /// Total storage size for all parameters in this network
    /// </summary>
    /// <returns>data storage size</returns>
    public DataSize StorageSize() => DataSize.FromValues64(TrainableParameterCount());

    /// <summary>
    /// Evaluate the sequential network using the provided input
    /// </summary>
    /// <param name="input">input tensor of shape [N,C,H,W]</param>
    /// <param name="ctx">optional evaluation context to store intermediary tensors</param>
    /// <returns>output tensor of shape [N,C,H,W]</returns>
    public Tensor<float> Forward(Tensor<float> input, EvaluationContext? ctx = null)
    {
        Tensor<float> i = input.ReshapeShared(input.Shape.NormalizeRank(4));
        foreach (var layer in layers)
        {
            var o = layer.Forward(i);
            if (ctx is not null)
            {
                var lctx = new IOContext(i, o);
                ctx.Save(layer, lctx);
            }
            i = o;
        }
        return i;
    }

    /// <summary>
    /// Perform a backwards evaluation of the network 
    /// </summary>
    /// <param name="dy">gradient of the output</param>
    /// <param name="ctx">the evaluation context to retrieve intermediary tensors from</param>
    /// <returns>gradient of the input as well as gradients for each layer in a list</returns>
    public Gradients Backward(Tensor<float> dy, EvaluationContext ctx)
    {
        var subGrad = new Gradients[layers.Count];
        for (var i = 0; i < subGrad.Length; i++)
        {
            var j = layers.Count - 1 - i;
            var layer = layers[j];
            var lctx = ctx.Get<IOContext>(layer);
            var grad = layer.Backward(lctx.Input, lctx.Output, dy);
            dy = grad.dX;
            subGrad[j] = grad;
        }
        return new GradientList(dy, subGrad);
    }

}