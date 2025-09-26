using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// A simple network composed of sequential layers with no skip connections, residuals, or branches
/// </summary>
public class SequentialNetwork: INetworkModule
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
            var o = layer.Forward(i, ctx);
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
    public Gradients Backward(Tensor<float> dy, EvaluationContext ctx, IClippingStrategy? clipping = null)
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

    public void Update(float learningRate,Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)
    {
        if (gradients is not GradientList lst)
            throw new ArgumentException("Expected a GradientList object", nameof(gradients));

        for (var i = 0; i < layers.Count; i++)
        {
            var layer = layers[i];
            layer.Update(learningRate, lst.dN(i), optimizer, regularization);
        }
    }
}