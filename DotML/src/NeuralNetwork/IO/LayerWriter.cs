namespace DotML.Network;

/// <summary>
/// Writer to encode layer information to a text format
/// </summary>
public class LayerWriter : ILayerInputVisitor<int>, IDisposable {

    protected TextWriter sb;
    private static LayerDescriber describer = new LayerDescriber();

    public LayerWriter(TextWriter writer) {
        this.sb = writer;
        WriteHeader();
    }

    protected virtual void WriteHeader() {
    
    }

    protected virtual void WriteFooter() {
        
    }

    protected virtual void WriteInputLayer(IFeedforwardNetworkLayer first) {
        
    }

    protected virtual void WriteLayerRow(IFeedforwardNetworkLayer layer, string description) {
        
    }

    public void Visit(ConvolutionLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return;
    }

    public void Visit(DepthwiseConvolutionLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }

        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return;
    }

    public void Visit(TransposeConvolutionLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }

        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return;
    }

    public void Visit(PoolingLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }

        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return;
    }

    public void Visit(FlatteningLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return;
    }

    public void Visit(DropoutLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return;
    }

    public void Visit(LayerNorm layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return;
    }

    public void Visit(BatchNorm layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return;
    }

    public void Visit(DenseLinearLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return;
    }

    public void Visit(ActivationLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return;
    }

    public void Visit(SoftmaxLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return;
    }

    public void Dispose() {
        WriteFooter();
    }

    public void Visit(InputCapture capture, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(capture);
        }
        
        WriteLayerRow(
            capture, 
            capture.Visit(describer)
        );
        return;
    }
}