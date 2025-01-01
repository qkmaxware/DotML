namespace DotML.Network;

/// <summary>
/// Writer to encode layer information and biases to a markdown file
/// </summary>
public class LayerMarkdownWriter : LayerWriter{

    public LayerMarkdownWriter(TextWriter writer) : base(writer) { }

    protected override void WriteHeader() {
        sb.Write('|'); sb.Write("Layer Type"); sb.Write('|'); sb.Write("Output Shape"); sb.Write('|'); sb.Write("Trainable Parameters"); sb.Write('|'); sb.Write("Un-trainable Parameters"); sb.Write('|'); sb.Write("Description"); sb.Write('|'); sb.WriteLine();
        sb.Write('|'); sb.Write("---"); sb.Write('|'); sb.Write("---"); sb.Write('|'); sb.Write("---"); sb.Write('|'); sb.Write("---"); sb.Write('|'); sb.WriteLine();
    }

    protected override void WriteFooter() {
        // Do nothing
    }

    protected override void WriteInputLayer(IConvolutionalFeedforwardNetworkLayer first) {
        sb.Write('|');
            sb.Write("Input");
        sb.Write('|');
            sb.Write(first.InputShape);
        sb.Write('|');
            sb.Write(string.Empty);
        sb.Write('|');
            sb.Write(string.Empty);
        sb.Write('|');
            sb.Write("Input image/tensor");
        sb.Write('|');
            sb.WriteLine();
    }

    protected override void WriteLayerRow(IConvolutionalFeedforwardNetworkLayer layer, string description) {
        sb.Write('|');
            sb.Write(layer.GetType().Name);
        sb.Write('|');
            sb.Write(layer.OutputShape);
        sb.Write('|');
            sb.Write(layer.TrainableParameterCount());
        sb.Write('|');
            sb.Write(layer.UnTrainableParameterCount());
        sb.Write('|');
            sb.Write(description);
        sb.Write('|');
        sb.WriteLine();
    }
}