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

    private void WriteLayer(IFeedforwardNetworkLayer layer, int layerIndex) {
        if (layerIndex == 0) {
            WriteInputLayer(layer);
        }
        
        WriteLayerRow(
            layer, 
            layer.Visit(describer)
        );
        return;
    }

    public void Visit(ConvolutionLayer layer, int layerIndex)  => WriteLayer(layer, layerIndex);

    public void Visit(DepthwiseConvolutionLayer layer, int layerIndex) => WriteLayer(layer, layerIndex);

    public void Visit(PixelShuffle layer, int layerIndex)  => WriteLayer(layer, layerIndex);

    public void Visit(TransposeConvolutionLayer layer, int layerIndex) => WriteLayer(layer, layerIndex);

    public void Visit(PoolingLayer layer, int layerIndex)  => WriteLayer(layer, layerIndex);

    public void Visit(FlatteningLayer layer, int layerIndex)  => WriteLayer(layer, layerIndex);

    public void Visit(DropoutLayer layer, int layerIndex)  => WriteLayer(layer, layerIndex);

    public void Visit(LayerNorm layer, int layerIndex)  => WriteLayer(layer, layerIndex);

    public void Visit(BatchNorm layer, int layerIndex)  => WriteLayer(layer, layerIndex);

    public void Visit(DenseLinearLayer layer, int layerIndex)  => WriteLayer(layer, layerIndex);

    public void Visit(ActivationLayer layer, int layerIndex)  => WriteLayer(layer, layerIndex);

    public void Visit(SoftmaxLayer layer, int layerIndex)  => WriteLayer(layer, layerIndex);

    public void Visit(InputCapture layer, int layerIndex) => WriteLayer(layer, layerIndex);

    public void Visit(AdditionSkipConnection layer, int layerIndex) => WriteLayer(layer, layerIndex);

    public void Visit(ConcatenationSkipConnection layer, int layerIndex) => WriteLayer(layer, layerIndex);

    public void Dispose() {
        WriteFooter();
    }
}