namespace DotML.Network;

public abstract class BlockWalker : IBlockVisitor
{

    public abstract void VisitModule(INetworkModule module);

    public None Visit(object? obj, None arg)
    {
        if (obj is INetworkModule mod)
        {
            VisitModule(mod);
            return None.Value;
        }

        throw new NotImplementedException();
    } 

    public None Visit(Activation activation, None arg)
    {
        VisitModule(activation);
        return None.Value;
    }

    public None Visit(SoftmaxOutput softmax, None arg)
    {
        VisitModule(softmax);
        return None.Value;
    }

    public None Visit(Conv2D conv, None arg)
    {
        VisitModule(conv);
        return None.Value;
    }

    public None Visit(TransposeConv2D tconv, None arg)
    {
        VisitModule(tconv);
        return None.Value;
    }

    public None Visit(DenseLinear dense, None arg)
    {
        VisitModule(dense);
        return None.Value;
    }

    public None Visit(Dropout dropout, None arg)
    {
        VisitModule(dropout);
        return None.Value;
    }

    public None Visit(BatchNorm2D norm, None arg)
    {
        VisitModule(norm);
        return None.Value;
    }

    public None Visit(GroupNorm norm, None arg)
    {
        VisitModule(norm);
        return None.Value;
    }

    public None Visit(LayerNorm norm, None arg)
    {
        VisitModule(norm);
        return None.Value;
    }

    public None Visit(PixelShuffler shuffle, None arg)
    {
        VisitModule(shuffle);
        return None.Value;
    }

    public None Visit(AvgPool2D pool, None arg)
    {
        VisitModule(pool);
        return None.Value;
    }

    public None Visit(MaxPool2D pool, None arg)
    {
        VisitModule(pool);
        return None.Value;
    }

    public None Visit(MinPool2D pool, None arg)
    {
        VisitModule(pool);
        return None.Value;
    }

    public None Visit(GlobalAvgPool2D pool, None arg)
    {
        VisitModule(pool);
        return None.Value;
    }

    public None Visit(GlobalMaxPool2D pool, None arg)
    {
        VisitModule(pool);
        return None.Value;
    }

    public None Visit(GlobalMinPool2D pool, None arg)
    {
        VisitModule(pool);
        return None.Value;
    }

    public None Visit(Reshape reshape, None arg)
    {
        VisitModule(reshape);
        return None.Value;
    }

    public None Visit(Flatten flatten, None arg)
    {
        VisitModule(flatten);
        return None.Value;
    }

    public None Visit(Center2D center, None arg)
    {
        VisitModule(center);
        return None.Value;
    }

    public None Visit(ResidualBlock block, None arg)
    {
        VisitModule(block);

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
        VisitModule(block);

        for (var layerIndex = 0; layerIndex < block.LayerCount; layerIndex++)
        {
            if (block[layerIndex] is not IBlockVisitable visitable)
                continue;

            visitable.Accept(this, arg);
        }

        return None.Value;
    }
}