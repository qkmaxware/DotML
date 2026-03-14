using System.ComponentModel;
using System.Reflection;
using DotML.Network;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

public class ResidualBlockContext : IModuleContext
{
    public Shape InputShape => Input.Shape;
    public Shape OutputShape => Output.Shape;

    public Shape MainPathShape { get; init; }
    public Shape ResidualPathShape { get; init; }
    public Tensor<float> Input { get; init; }
    public Tensor<float> Output { get; init; }

    public ResidualBlockContext(Tensor<float> input, Tensor<float> output, Shape mainPath, Shape residualPath)
    {
        this.Input = input;
        this.Output = output;
        this.MainPathShape = mainPath;
        this.ResidualPathShape = residualPath;
    }
}

/// <summary>
/// A basic residual block which transforms the input along the main path and then accumulates the residual with the output of that transformation
/// <para>
/// <code>
/// --- ( ) --- Main Path --- (R) -->
///      |                     ^
///      |                     |
///       ----- Residual ------
/// </code>
/// </para>
/// </summary>
public abstract class ResidualBlock : INetworkModule, IBlockVisitable
{
    public INetworkModule MainPath;
    public INetworkModule? ResidualPath;

    public ResidualBlock(INetworkModule gx, INetworkModule? residual = null)
    {
        this.MainPath = gx;
        this.ResidualPath = residual;
    }

    public abstract Shape ForwardShape(Shape input);

    public Tensor<float> Forward(Tensor<float> X, EvaluationContext? ctx = null)
    {
        // Compute the outputs of both paths, if ResidualPath is null, treat it as the identity function (aka apply no transformation)
        var residual = ResidualPath?.Forward(X, ctx) ?? X;
        var Y = MainPath.Forward(X, ctx);

        // Combine the results using whatever strategy 
        var result = Combine(Y, residual);

        // Store backpropagation context if required
        if (ctx is not null)
        {
            ctx.Save(this, new ResidualBlockContext(X, result, Y.Shape, residual.Shape));
        }

        return result;
    }

    /// <summary>
    /// Strategy to combine the output of the main path and residual path together
    /// </summary>
    /// <param name="output">output of the main path</param>
    /// <param name="residual">output of the residual path</param>
    /// <returns>combined tensor</returns>
    protected abstract Tensor<float> Combine(Tensor<float> output, Tensor<float> residual);

    public Gradients Backward(Tensor<float> dY, EvaluationContext ctx, ILocalClippingStrategy<float>? clipping = null)
    {
        // Fetch the cached shapes for proper splitting
        var context = ctx.Get<ResidualBlockContext>(this);
        var (dY_main, dY_res) = SplitGradient(dY, context.MainPathShape, context.ResidualPathShape);

        // Gradients through the main path
        var mainGrads = MainPath.Backward(dY_main, ctx, clipping);

        // Gradients through the residual path (or identity)
        Gradients residualGrads = (ResidualPath is not null) ? ResidualPath.Backward(dY_res, ctx, clipping) : new Gradient(dY_res);

        return CombineGradients(mainGrads, residualGrads);
    }

    /// <summary>
    /// Split the output gradient into multiple gradients which can be passed backwards to the main path and the residual path separately
    /// </summary>
    /// <param name="dY">gradient of the combined output</param>
    /// <param name="mainOutputShape">shape of the output of the main path</param>
    /// <param name="residualOutputShape">shape of the output of the residual path</param>
    /// <returns>gradient pair for the main and residual paths</returns>
    protected abstract (Tensor<float> dM, Tensor<float> dR) SplitGradient(Tensor<float> dY, Shape mainOutputShape, Shape residualOutputShape);

    /// <summary>
    /// Combine gradients of the main path and residual path into a single input gradient which can be passed backwards to prior layers
    /// </summary>
    /// <param name="main">gradients along the main path</param>
    /// <param name="residual">gradients along the residual path</param>
    /// <returns>input gradient</returns>
    protected abstract ResidualBlockGradients CombineGradients(Gradients main, Gradients residual);

    public void Initialize(IInitializer initializer)
    {
        // Initialize both paths using the same initializer
        MainPath.Initialize(initializer);
        ResidualPath?.Initialize(initializer);
    }

    public void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)
    {
        if (gradients is not ResidualBlockGradients res)
            throw new ArgumentException("Expected ResidualBlockGradients", nameof(gradients));

        this.MainPath.Update(learningRate, res.Main, optimizer, regularization);
        this.ResidualPath?.Update(learningRate, res.Residual, optimizer, regularization);
    }

    public TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg) => visitor.Visit(this, arg);
}

public class ResidualBlockGradients : Gradients
{
    public Gradients Main { get; }
    public Gradients Residual { get; }

    public ResidualBlockGradients(Tensor<float> dX, Gradients main, Gradients residual)
        : base(dX)
    {
        Main = main;
        Residual = residual;
    }

    public override void Clip(ILocalClippingStrategy<float> clipping)
    {
        clipping.ClipInput(dX);

        this.Main.Clip(clipping);
        this.Residual.Clip(clipping);
    }

    public override IEnumerable<Tensor<float>> EnumerateParameterGradients()
    {
        foreach (var p in Main.EnumerateParameterGradients())
            yield return p;
        foreach (var p in Residual.EnumerateParameterGradients())
            yield return p;
    }

}