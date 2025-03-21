namespace DotML.Network.IO.Netbuild;

public abstract class LayerReference {
    public abstract int IndexOf(Dictionary<string, int> aliases);
}

public class NamedLayerReference : LayerReference {
    private string index;
    public NamedLayerReference(string index) => this.index = index;
    public override int IndexOf(Dictionary<string, int> aliases) => aliases[this.index];
    
    public override string ToString() {
        return index.ToString();
    }
}

public class IndexedLayerReference : LayerReference {
    private int index;
    public IndexedLayerReference(int index) => this.index = index;
    public override int IndexOf(Dictionary<string, int> aliases) => index;

    public override string ToString() {
        return index.ToString();
    }
}