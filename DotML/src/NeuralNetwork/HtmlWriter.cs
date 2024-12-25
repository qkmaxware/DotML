namespace DotML.Network;

/// <summary>
/// Writer to encode layer information and biases to an HTML file
/// </summary>
public class LayerHtmlWriter : LayerWriter{

    public LayerHtmlWriter(TextWriter writer) : base(writer) { }

    protected override void WriteHeader() {
        sb.WriteLine("<table>");
        sb.WriteLine("<thead>");
        sb.WriteLine("<tr>");
        sb.Write("<th>");sb.Write("Layer Type"); sb.Write("</th><th>"); sb.Write("Output Shape"); sb.Write("</th><th>"); sb.Write("Parameters"); sb.Write("</th><th>"); sb.Write("Description"); sb.WriteLine("</th>");
        sb.WriteLine("</tr>");
        sb.WriteLine("</thead>");
        sb.WriteLine("<tbody>");
    }

    protected override void WriteFooter() {
        sb.WriteLine("</tbody>");
        sb.WriteLine("</table>");
    }

    protected override void WriteInputLayer(IConvolutionalFeedforwardNetworkLayer first) {
        sb.WriteLine("<tr>");
        sb.Write("<td>");
            sb.Write("Input");
        sb.Write("</td><td>");
            sb.Write(first.InputShape);
        sb.Write("</td><td>");
            sb.Write(string.Empty);
        sb.Write("</td><td>");
            sb.Write("Input image/tensor");
        sb.Write("</td>");
            sb.WriteLine("</tr>");
    }

    protected override void WriteLayerRow(IConvolutionalFeedforwardNetworkLayer layer, string description) {
        sb.WriteLine("<tr>");
        sb.Write("<td>");
            sb.Write(layer.GetType().Name);
        sb.Write("</td><td>");
            sb.Write(layer.OutputShape);
        sb.Write("</td><td>");
            sb.Write(layer.TrainableParameterCount());
        sb.Write("</td><td>");
            sb.Write(description);
        sb.Write("</td>");
        sb.WriteLine("</tr>");
    }
}