using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// A named architecture block that encapsulates a named sub-network as a single module
/// </summary>
public class ArchitectureBlock : INetworkModule, IBlockVisitable
{
    /// <summary>
    /// Architecture name
    /// </summary>
    public string Name { get; init; }
    /// <summary>
    /// Architecture description
    /// </summary>
    public string? Description { get; init; }
    public Uri? ReferenceUrl { get; init; }
    /// <summary>
    /// Architecture input shape if network only works with inputs of the given shape
    /// </summary>
    public Shape? RequiredInputShape { get; init; }
    /// <summary>
    /// Architecture output shape if networks only work with inputs of a given shape
    /// </summary>
    public Shape? OutputShape { get; init; }
    /// <summary>
    /// The underlying network module that constitutes the network architecture
    /// </summary>
    public INetworkModule RootModule { get; init; }

    public ArchitectureBlock(string name, Shape? inputShape, INetworkModule rootModule, string? description = null, Uri? referenceUri = null)
    {
        Name = name;
        RequiredInputShape = inputShape;
        if (RequiredInputShape.HasValue)
            OutputShape = rootModule.ForwardShape(RequiredInputShape.Value);
        RootModule = rootModule;
        this.Description = description;
        this.ReferenceUrl = referenceUri;
    }

    public Gradients Backward(Tensor<float> dy, EvaluationContext ctx, ILocalClippingStrategy<float>? clipping = null)
    {
        return RootModule.Backward(dy, ctx, clipping);
    }

    public Tensor<float> Forward(Tensor<float> channels, EvaluationContext? ctx = null)
    {
        return RootModule.Forward(channels, ctx);
    }

    public Shape ForwardShape(Shape input)
    {
        return RootModule.ForwardShape(input);
    }

    public void Initialize(IInitializer initializer)
    {
        RootModule.Initialize(initializer);
    }

    public void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)
    {
        RootModule.Update(learningRate, gradients, optimizer, regularization);
    }

    public TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg)
    {
        if (RootModule is IBlockVisitable visitable)
        {
            return visitable.Accept(visitor, arg);
        }
        throw new NotSupportedException();
    }

}