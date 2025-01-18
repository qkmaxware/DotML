using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace DotML.Network;

/// <summary>
/// Writer to decode layer weights and biases from a safetensor file
/// </summary>
public class NetworkJsonSerializer : ILayerVisitor<int, bool>, IDisposable {

    protected TextWriter sb;
    static string tab = "    ";

    public NetworkJsonSerializer(TextWriter writer) {
        this.sb = writer;
        sb.WriteLine('[');
    }

    public void Dispose() {
        sb.WriteLine();
        sb.Write(']');
    }

    private bool Encode(ILayer layer, int layerIndex) {
        if (layerIndex != 0)
            sb.WriteLine(',');

        sb.Write(tab);
        sb.Write('{');

        // Type
        sb.Write(JsonSerializer.Serialize("$Type")); sb.Write(": "); sb.Write(JsonSerializer.Serialize(layer.GetType()?.FullName ?? string.Empty));

        // Args
        foreach (PropertyInfo property in layer.GetType().GetProperties()) {
            if (!property.CanRead)
                continue;
            if (property.GetCustomAttribute<JsonIgnoreAttribute>() is not null)
                continue;

            object? value = property.GetValue(layer, null);
            if (value is LossFunction loss)
                value = loss.Method.Name;
            if (value is ActivationFunction activation)
                value = activation.GetType().FullName;
            
            sb.Write(", ");
            sb.Write(JsonSerializer.Serialize(property.Name)); sb.Write(": "); sb.Write(JsonSerializer.Serialize(value));
        }


        sb.Write('}');

        return true;
    }

    public bool Visit(ConvolutionLayer layer, int layerIndex) {
        return Encode(layer, layerIndex);
    }

    public bool Visit(DepthwiseConvolutionLayer layer, int layerIndex) {
        return Encode(layer, layerIndex);
    }

    public bool Visit(PoolingLayer layer, int layerIndex) {
        return Encode(layer, layerIndex);
    }

    public bool Visit(FlatteningLayer layer, int layerIndex) {
        return Encode(layer, layerIndex);
    }

    public bool Visit(DropoutLayer layer, int layerIndex) {
        return Encode(layer, layerIndex);
    }

    public bool Visit(LayerNorm layer, int layerIndex) {
        return Encode(layer, layerIndex);
    }

    public bool Visit(BatchNorm layer, int layerIndex) {
        return Encode(layer, layerIndex);
    }

    public bool Visit(FullyConnectedLayer layer, int layerIndex) {
        return Encode(layer, layerIndex);
    }

    public bool Visit(ActivationLayer layer, int layerIndex) {
        return Encode(layer, layerIndex);
    }

    public bool Visit(SoftmaxLayer layer, int layerIndex) {
        return Encode(layer, layerIndex);
    }
}