namespace DotML.Network.IO.Netbuild;

public class InsertBeforeStatement : Statement {
    private LayerReference reference;
    string layer_name;
    Func<Shape3D, ArgumentMap, IFeedforwardNetworkLayer> factory;
    public Dictionary<string, Literal> arguments = new Dictionary<string, Literal>();

    public InsertBeforeStatement(LayerReference reference, string layer_name, Func<Shape3D, ArgumentMap, IFeedforwardNetworkLayer> factory, List<(Token<string>, Literal)> args) {
        this.reference = reference;
        this.layer_name = layer_name;
        this.factory = factory;
        foreach (var pair in args) {
            arguments[pair.Item1.Value] = pair.Item2;
        } 
    }

    public override void Action(BuildEnvironment env) {
        var network = env.Network;
        if (network is null)
            return;
        
        // Find the layer to insert after
        var layer_index = this.reference.IndexOf(env.LayerAliases);
        var layer = network.GetLayer(layer_index);
        if (layer is null)
            throw new ArgumentException($"Unknown layer '{layer_name}'");

        // Create the layer 
        var output_shape = layer.InputShape;
        IFeedforwardNetworkLayer created = factory(output_shape, new ArgumentMap(env, arguments));
        var index = layer_index;

        network.InsertLayerBefore(created, (index, l) => layer_index == index);

        // Add the alias TODO
    }

    public override string ToString() {
        return $"INSERT BEFORE {reference} {layer_name} {string.Join(' ', arguments.Select(kv => $"{kv.Key}={kv.Value}"))}";
    }
}

internal class InsertAfterStatement : Statement {
    private LayerReference reference;
    string layer_name;
    Func<Shape3D, ArgumentMap, IFeedforwardNetworkLayer> factory;
    public Dictionary<string, Literal> arguments = new Dictionary<string, Literal>();

    public InsertAfterStatement(LayerReference reference, string layer_name, Func<Shape3D, ArgumentMap, IFeedforwardNetworkLayer> factory, List<(Token<string>, Literal)> args) {
        this.reference = reference;
        this.layer_name = layer_name;
        this.factory = factory;
        foreach (var pair in args) {
            arguments[pair.Item1.Value] = pair.Item2;
        } 
    }

    public override void Action(BuildEnvironment env) {
        var network = env.Network;
        if (network is null)
            return;
        
        // Find the layer to insert after
        var layer_index = this.reference.IndexOf(env.LayerAliases);
        var layer = network.GetLayer(layer_index);
        if (layer is null)
            throw new ArgumentException($"Unknown layer '{layer_name}'");

        // Create the layer 
        var output_shape = layer.OutputShape;
        IFeedforwardNetworkLayer created = factory(output_shape, new ArgumentMap(env, arguments));
        var index = layer_index + 1;

        network.InsertLayerAfter(created, (index, l) => layer_index == index);

        // Add the alias TODO
    }

    public override string ToString() {
        return $"INSERT AFTER {reference} {layer_name} {string.Join(' ', arguments.Select(kv => $"{kv.Key}={kv.Value}"))}";
    }
}