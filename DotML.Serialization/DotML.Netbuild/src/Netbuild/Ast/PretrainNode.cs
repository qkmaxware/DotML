using DotML.Network.IO.Netbuild;

namespace DotML.Network.IO.Netbuild;

public class PretrainStatement : Statement {
    private string path;
    public PretrainStatement(string path) {
        this.path = path;
    }

    public override void ModuleAction(BuildEnvironment env)
    {
        var net = env.NetworkBlock;
        if (net is null)
            return;

        var safe = Safetensors.ReadFromFile(this.path);
        var deserializer = new SafetensorDeserializer();
        deserializer.Deserialize(net, safe);
    }

    public override string ToString()
    {
        return $"PRETRAIN {path}";
    }
}