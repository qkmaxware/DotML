namespace DotML.Network;

/// <summary>
/// Writer to encode layer information to a text format
/// </summary>
public class LayerWriter : IConvolutionalLayerVisitor<int, bool>, IDisposable {

    protected TextWriter sb;

    public LayerWriter(TextWriter writer) {
        this.sb = writer;
        WriteHeader();
    }

    protected virtual void WriteHeader() {
    
    }

    protected virtual void WriteFooter() {
        
    }

    protected virtual void WriteInputLayer(IConvolutionalFeedforwardNetworkLayer first) {
        
    }

    protected virtual void WriteLayerRow(IConvolutionalFeedforwardNetworkLayer layer, string description) {
        
    }

    public bool Visit(ConvolutionLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            $"{layer.FilterCount} filters of size {layer.Filters.FirstOrDefault()?.FirstOrDefault().Shape}"
        );
        return true;
    }

    public bool Visit(DepthwiseConvolutionLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }

        WriteLayerRow(
            layer, 
            $"{layer.Filter.Count} kernels of size {layer.Filter?.FirstOrDefault().Shape}"
        );
        return true;
    }

    public bool Visit(PoolingLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }

        WriteLayerRow(
            layer, 
            $"Pooling with a size of {layer.FilterHeight}x{layer.FilterWidth}"
        );
        return true;
    }

    public bool Visit(FlatteningLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            $"Flatten multi-dimensional input to 1D"
        );
        return true;
    }

    public bool Visit(DropoutLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            $"Dropout with probability {layer.DropoutRate} to reduce overfitting"
        );
        return true;
    }

    public bool Visit(LayerNorm layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            $"Normalize the inputs across the layer"
        );
        return true;
    }

    public bool Visit(BatchNorm layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            $"Normalize the inputs across the entire batch"
        );
        return true;
    }

    public bool Visit(FullyConnectedLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            $"Fully connected layer of {layer.NeuronCount} neurons"
        );
        return true;
    }

    public bool Visit(ActivationLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            $"Apply {layer.ActivationFunction?.GetType().Name} activation function to inputs"
        );
        return true;
    }

    public bool Visit(SoftmaxLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            $"Convert output to probability distribution over {layer.OutputShape.Count} classes"
        );
        return true;
    }

    public void Dispose() {
        WriteFooter();
    }
}