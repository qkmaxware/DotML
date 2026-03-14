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

public class MaskContext : IOContext {
    public Tensor<float> Mask { get; init; }
    public MaskContext(Tensor<float> input, Tensor<float> output, Tensor<float> mask): base(input, output) {
        this.Mask = mask;
    }
}

public enum EvaluationMode
{
    Inference, Training
}

public class EvaluationContext
{
    public EvaluationMode Mode {get; init;}
    private Dictionary<object, IModuleContext> _storage = new Dictionary<object, IModuleContext>();

    public EvaluationContext() : this(EvaluationMode.Inference) {}

    public EvaluationContext(EvaluationMode mode)
    {
        this.Mode = mode;
    }

    public virtual void Save(object module, IModuleContext context)
    {
        _storage[module] = context;
    }

    public virtual TCtx Get<TCtx>(object module)
    where TCtx : IModuleContext
    {
        if (_storage.TryGetValue(module, out var ctx)
            && ctx is TCtx typedCtx)
        {
            return typedCtx;
        }

        throw new KeyNotFoundException($"No context saved for module '{module}' of type '{typeof(TCtx)}'");
    }

}