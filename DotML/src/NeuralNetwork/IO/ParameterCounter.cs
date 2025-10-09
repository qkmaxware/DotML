
namespace DotML.Network.IO;

/// <summary>
/// A visitor which can compute the number of trainable and un-trainable parameters in a network.
/// </summary>
public class ParameterCounter : IBlockVisitor
{
    /// <summary>
    /// Number of trainable parameters
    /// </summary>
    public int Trainable { get; private set; } = 0;
    /// <summary>
    /// Number of un-trainable parameters
    /// </summary>
    public int UnTrainable { get; private set; } = 0;

    public None Visit(object? obj, None arg) { /* Do nothing fallback */ return None.Value; }

    public None Visit(Activation activation, None arg)
    {
        Trainable += activation.TrainableParameterCount();
        UnTrainable += activation.UnTrainableParameterCount();
        return None.Value;
    }

    public None Visit(Conv2D conv, None arg)
    {
        Trainable += conv.TrainableParameterCount();
        UnTrainable += conv.UnTrainableParameterCount();
        return None.Value;
    }

    public None Visit(TransposeConv2D tconv, None arg)
    {
        Trainable += tconv.TrainableParameterCount();
        UnTrainable += tconv.UnTrainableParameterCount();
        return None.Value;
    }

    public None Visit(DenseLinear dense, None arg)
    {
        Trainable += dense.TrainableParameterCount();
        UnTrainable += dense.UnTrainableParameterCount();
        return None.Value;
    }

    public None Visit(Dropout dropout, None arg)
    {
        Trainable += dropout.TrainableParameterCount();
        UnTrainable += dropout.UnTrainableParameterCount();
        return None.Value;
    }

    public None Visit(Center2D center, None arg)
    {
        Trainable += center.TrainableParameterCount();
        UnTrainable += center.UnTrainableParameterCount();
        return None.Value;
    }

    public None Visit(BatchNorm2 norm, None arg)
    {
        Trainable += norm.TrainableParameterCount();
        UnTrainable += norm.UnTrainableParameterCount();
        return None.Value;
    }

    public None Visit(GroupNorm2 norm, None arg)
    {
        Trainable += norm.TrainableParameterCount();
        UnTrainable += norm.UnTrainableParameterCount();
        return None.Value;
    }

    public None Visit(LayerNorm2 norm, None arg)
    {
        Trainable += norm.TrainableParameterCount();
        UnTrainable += norm.UnTrainableParameterCount();
        return None.Value;
    }

    public None Visit(PixelShuffler shuffle, None arg)
    {
        Trainable += shuffle.TrainableParameterCount();
        UnTrainable += shuffle.UnTrainableParameterCount();
        return None.Value;
    }

    public None Visit(AvgPool2D pool, None arg)
    {
        Trainable += pool.TrainableParameterCount();
        UnTrainable += pool.UnTrainableParameterCount();
        return None.Value;
    }

    public None Visit(MaxPool2D pool, None arg)
    {
        Trainable += pool.TrainableParameterCount();
        UnTrainable += pool.UnTrainableParameterCount();
        return None.Value;
    }

    public None Visit(MinPool2D pool, None arg)
    {
        Trainable += pool.TrainableParameterCount();
        UnTrainable += pool.UnTrainableParameterCount();
        return None.Value;
    }

    public None Visit(Reshape reshape, None arg)
    {
        Trainable += reshape.TrainableParameterCount();
        UnTrainable += reshape.UnTrainableParameterCount();
        return None.Value;
    }

    public None Visit(Flatten flatten, None arg)
    {
        Trainable += flatten.TrainableParameterCount();
        UnTrainable += flatten.UnTrainableParameterCount();
        return None.Value;
    }

    public None Visit(SoftmaxOutput softmax, None arg)
    {
        Trainable += softmax.TrainableParameterCount();
        UnTrainable += softmax.UnTrainableParameterCount();
        return None.Value;
    }

    public None Visit(ResidualBlock block, None arg)
    {
        if (block.MainPath is IBlockVisitable visitable)
        {
            visitable.Accept(this, arg);
        }
        if (block.ResidualPath is IBlockVisitable visitable2)
        {
            visitable2.Accept(this, arg);
        }
        return None.Value;
    }

    public None Visit(SequentialBlock block, None arg)
    {
        for (var layerIndex = 0; layerIndex < block.LayerCount; layerIndex++)
        {
            if (block[layerIndex] is not IBlockVisitable visitable)
                continue;

            visitable.Accept(this, arg);
        }
        return None.Value;
    }
}