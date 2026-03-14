namespace DotML.Network.IO.Netbuild;

public class ArgStatement : Statement {

    private Dictionary<string, Literal> values = new Dictionary<string, Literal>();

    public ArgStatement() { }

    public void Set(Token<string> key, Literal value) {
        this.values[key.Value] = value;
    }
    
    public override void ModuleAction(BuildEnvironment env) {
        foreach (var arg in values) {
            env.Arguments[arg.Key] = arg.Value.ValueOf();
        }
    }

    public override string ToString()
    {
        return $"ARG {string.Join(' ', values.Select(kv => $"{kv.Key}={kv.Value}"))}";
    }
    }