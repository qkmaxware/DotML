using DotML.Network.IO.Netbuild;

namespace DotML.Network.IO.Netbuild;

public class PretrainStatement : Statement {
    private string path;
    public PretrainStatement(string path) {
        this.path = path;
    }

    public override void Action(BuildEnvironment env) {
        var net = env.Network;
        if (net is null)
            return;

        var safe = Safetensors.ReadFromFile(this.path);
        net.FromSafetensor(safe);
    }

    public override string ToString() {
        return $"PRETRAIN {path}";
    }
}