using System.Numerics;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;

namespace DotML.Network.IO;

public class SafetensorSerializer : IBlockVisitor
{
    private Safetensors safetensors = new Safetensors();

    public Safetensors ToSafetensors() => safetensors;

    public Safetensors Serialize(IBlockVisitable module)
    {
        module.Accept(this);
        return safetensors;
    }

    private Stack<string> scopes = new Stack<string>();

    private string Scope
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            scopes.TryPeek(out var scope);
            return scope ?? string.Empty;
        }
    }
    private void EnterScope(object subspace)
    {
        String subspaceStr = subspace.ToString() ?? "?";
        if (string.IsNullOrEmpty(Scope))
            scopes.Push(subspaceStr);
        else
            scopes.Push(Scope + "." + subspaceStr);
    }

    private void ExitScope()
    {
        scopes.TryPop(out var _);
    }

    private void Push<TNum>(string name, Tensor<TNum> tensor)
    where TNum : INumber<TNum>
    {
        var scope = Scope;
        var id = string.IsNullOrEmpty(scope) ? name : scope + "." + name;
        safetensors.Add(id, tensor);
    }

    public None Visit(Activation activation, None arg) { /* No tensors */ return None.Value; }

    public None Visit(Conv2D conv, None arg)
    {
        Push(nameof(Conv2D.Weights), conv.Weights);
        Push(nameof(Conv2D.Biases), conv.Biases);
        return None.Value;
    }

    public None Visit(TransposeConv2D tconv, None arg)
    {
        Push(nameof(TransposeConv2D.Weights), tconv.Weights);
        Push(nameof(TransposeConv2D.Biases), tconv.Biases);
        return None.Value;
    }

    public None Visit(DenseLinear dense, None arg)
    {
        Push(nameof(DenseLinear.Weights), dense.Weights);
        Push(nameof(DenseLinear.Biases), dense.Biases);
        return None.Value;
    }

    public None Visit(Dropout dropout, None arg) { /* No tensors */ return None.Value; }

    public None Visit(Center2D center, None arg) { /* No tensors */ return None.Value; }

    public None Visit(BatchNorm2D norm, None arg)
    {
        Push(nameof(BatchNorm2D.RunningMean), norm.RunningMean);
        Push(nameof(BatchNorm2D.RunningVariance), norm.RunningVariance);
        Push(nameof(BatchNorm2D.Weights), norm.Weights);
        Push(nameof(BatchNorm2D.Biases), norm.Biases);
        return None.Value;
    }

    public None Visit(GroupNorm norm, None arg)
    {
        Push(nameof(GroupNorm.Weights), norm.Weights);
        Push(nameof(GroupNorm.Biases), norm.Biases);
        return None.Value;
    }

    public None Visit(LayerNorm norm, None arg)
    {
        Push(nameof(LayerNorm.Weights), norm.Weights);
        Push(nameof(LayerNorm.Biases), norm.Biases);
        return None.Value;
    }

    public None Visit(PixelShuffler shuffle, None arg) { /* No tensors */ return None.Value; }

    public None Visit(AvgPool2D pool, None arg) { /* No tensors */ return None.Value; }

    public None Visit(MaxPool2D pool, None arg) { /* No tensors */ return None.Value; }

    public None Visit(MinPool2D pool, None arg) { /* No tensors */ return None.Value; }

    public None Visit(GlobalAvgPool2D pool, None arg) { /* No tensors */ return None.Value; }

    public None Visit(GlobalMaxPool2D pool, None arg) { /* No tensors */ return None.Value; }

    public None Visit(GlobalMinPool2D pool, None arg) { /* No tensors */ return None.Value; }

    public None Visit(Reshape reshape, None arg) { /* No tensors */ return None.Value; }

    public None Visit(Flatten flatten, None arg) { /* No tensors */ return None.Value; }

    public None Visit(SoftmaxOutput softmax, None arg) { /* No tensors */ return None.Value; }



    public None Visit(ResidualBlock block, None arg)
    {
        EnterScope(nameof(ResidualBlock));
        if (block.MainPath is IBlockVisitable visitable)
        {
            EnterScope(nameof(ResidualBlock.MainPath));
            visitable.Accept(this);
            ExitScope();
        }
        if (block.ResidualPath is IBlockVisitable visitable2)
        {
            EnterScope(nameof(ResidualBlock.ResidualPath));
            visitable2.Accept(this);
            ExitScope();
        }
        ExitScope();
        return None.Value;
    }

    public None Visit(SequentialBlock block, None arg)
    {
        EnterScope(nameof(SequentialBlock));

        for (var layerIndex = 0; layerIndex < block.LayerCount; layerIndex++)
        {
            if (block[layerIndex] is not IBlockVisitable visitable)
                continue;

            EnterScope(layerIndex);
            visitable.Accept(this);
            ExitScope();
        }

        ExitScope();
        return None.Value;
    }
}

public class SafetensorDeserializer : IBlockVisitor<Safetensors, None>
{
    public void Deserialize(IBlockVisitable module, Safetensors safetensors)
    {
        module.Accept(this, safetensors);
    }

    private Stack<string> scopes = new Stack<string>();

    private string Scope
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            scopes.TryPeek(out var scope);
            return scope ?? string.Empty;
        }
    }

    private void EnterScope(object subspace)
    {
        String subspaceStr = subspace.ToString() ?? "?";
        if (string.IsNullOrEmpty(Scope))
            scopes.Push(subspaceStr);
        else
            scopes.Push(Scope + "." + subspaceStr);
    }

    private void ExitScope()
    {
        scopes.TryPop(out var _);
    }

    private Tensor<TNum> Load<TNum>(Safetensors st, string name)
    where TNum : INumber<TNum>
    {
        var scope = Scope;
        var id = string.IsNullOrEmpty(scope) ? name : scope + "." + name;
        return st.GetTensor<TNum>(id);
    }

    public None Visit(Activation activation, Safetensors arg) { /* No tensors */ return None.Value; }

    public None Visit(Conv2D conv, Safetensors arg)
    {
        conv.Weights = Load<float>(arg, nameof(BatchNorm2D.Weights));
        conv.Biases = Load<float>(arg, nameof(BatchNorm2D.Biases));
        return None.Value;
    }

    public None Visit(TransposeConv2D tconv, Safetensors arg)
    {
        tconv.Weights = Load<float>(arg, nameof(BatchNorm2D.Weights));
        tconv.Biases = Load<float>(arg, nameof(BatchNorm2D.Biases));
        return None.Value;
    }

    public None Visit(DenseLinear dense, Safetensors arg)
    {
        dense.Weights = Load<float>(arg, nameof(BatchNorm2D.Weights));
        dense.Biases = Load<float>(arg, nameof(BatchNorm2D.Biases));
        return None.Value;
    }

    public None Visit(Dropout dropout, Safetensors arg) { /* No tensors */ return None.Value; }

    public None Visit(Center2D center, Safetensors arg) { /* No tensors */ return None.Value; }

    public None Visit(BatchNorm2D norm, Safetensors arg)
    {
        norm.Weights = Load<float>(arg, nameof(BatchNorm2D.Weights));
        norm.Biases = Load<float>(arg, nameof(BatchNorm2D.Biases));
        norm.RunningMean = Load<float>(arg, nameof(BatchNorm2D.RunningMean));
        norm.RunningVariance = Load<float>(arg, nameof(BatchNorm2D.RunningVariance));
        return None.Value;
    }

    public None Visit(GroupNorm norm, Safetensors arg)
    {
        norm.Weights = Load<float>(arg, nameof(BatchNorm2D.Weights));
        norm.Biases = Load<float>(arg, nameof(BatchNorm2D.Biases));
        return None.Value;
    }

    public None Visit(LayerNorm norm, Safetensors arg)
    {
        norm.Weights = Load<float>(arg, nameof(BatchNorm2D.Weights));
        norm.Biases = Load<float>(arg, nameof(BatchNorm2D.Biases));
        return None.Value;
    }

    public None Visit(PixelShuffler shuffle, Safetensors arg) { /* No tensors */ return None.Value; }

    public None Visit(AvgPool2D pool, Safetensors arg) { /* No tensors */ return None.Value; }

    public None Visit(MaxPool2D pool, Safetensors arg) { /* No tensors */ return None.Value; }

    public None Visit(MinPool2D pool, Safetensors arg) { /* No tensors */ return None.Value; }

    public None Visit(GlobalAvgPool2D pool, Safetensors arg) { /* No tensors */ return None.Value; }

    public None Visit(GlobalMaxPool2D pool, Safetensors arg) { /* No tensors */ return None.Value; }

    public None Visit(GlobalMinPool2D pool, Safetensors arg) { /* No tensors */ return None.Value; }

    public None Visit(Reshape reshape, Safetensors arg) { /* No tensors */ return None.Value; }

    public None Visit(Flatten flatten, Safetensors arg) { /* No tensors */ return None.Value; }
    public None Visit(SoftmaxOutput softmax, Safetensors arg) { /* No tensors */ return None.Value; }

    public None Visit(ResidualBlock block, Safetensors arg)
    {
        EnterScope(nameof(ResidualBlock));
        if (block.MainPath is IBlockVisitable visitable)
        {
            EnterScope(nameof(ResidualBlock.MainPath));
            visitable.Accept(this, arg);
            ExitScope();
        }
        if (block.ResidualPath is IBlockVisitable visitable2)
        {
            EnterScope(nameof(ResidualBlock.ResidualPath));
            visitable2.Accept(this, arg);
            ExitScope();
        }
        ExitScope();
        return None.Value;
    }

    public None Visit(SequentialBlock block, Safetensors arg)
    {
        EnterScope(nameof(SequentialBlock));

        for (var layerIndex = 0; layerIndex < block.LayerCount; layerIndex++)
        {
            if (block[layerIndex] is not IBlockVisitable visitable)
                continue;

            EnterScope(layerIndex);
            visitable.Accept(this, arg);
            ExitScope();
        }

        ExitScope();
        return None.Value;
    }
}