namespace DotML.Network;

/// <summary>
/// Writer to encode layer information to a text format
/// </summary>
public class LayerWriter : ILayerVisitor<int, bool>, IDisposable {

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

    public bool Visit(ConvolutionLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return true;
    }

    public bool Visit(DepthwiseConvolutionLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }

        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return true;
    }

    public bool Visit(PoolingLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }

        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return true;
    }

    public bool Visit(FlatteningLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return true;
    }

    public bool Visit(DropoutLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return true;
    }

    public bool Visit(LayerNorm layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return true;
    }

    public bool Visit(BatchNorm layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return true;
    }

    public bool Visit(DenseLinearLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return true;
    }

    public bool Visit(ActivationLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return true;
    }

    public bool Visit(SoftmaxLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return true;
    }

    public void Dispose() {
        WriteFooter();
    }

    public bool Visit(InputCapture capture, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(capture);
        }
        
        WriteLayerRow(
            capture, 
            capture.Visit(describer)
        );
        return true;
    }
}