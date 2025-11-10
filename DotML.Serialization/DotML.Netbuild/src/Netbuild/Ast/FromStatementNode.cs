using DotML.Network.Templates;

namespace DotML.Network.IO.Netbuild;

public class Preamble : AstNode{
    public FromStatement From;

    public Preamble(FromStatement from) {
        this.From = from;
    }
}

public class FromStatement : Statement {
    private Shape3D? manualInputShape;
    public string? Identifier;
    public string? Version;

    public string? key;

    public FromStatement(Shape3D? input_shape, string? identifier, string? version) {
        this.manualInputShape = input_shape;
        this.Identifier = identifier;
        this.Version = version;

        this.key = string.IsNullOrEmpty(Version) ? Identifier: $"{Identifier}:{Version}";
    }

    private static T make_enum<T>(string? str, T @default) where T:struct {
        if (string.IsNullOrEmpty(str))
            return @default;

        if (Enum.TryParse<T>(value: str, ignoreCase: true, out T result))
            return result;
        return @default;
    }

    public override void ModuleAction(BuildEnvironment env)
    {
        env.NetworkBlock = new SequentialBlock();
        env.InputShape = manualInputShape ?? new Shape3D(1,1,1); // Default input shape if nothing else is specified

        INetworkModule? root = (Identifier?.ToLower()) switch
        {
            "alexnet" => new AlexNetFactory().MakeDefault(),
            "lenet" => new LeNetFactory().MakeDefault(),
            "espcn" => ((INetworkModuleFactory)new ESPCNFactory()).MakeDefault(),
            "fsrcnn" => ((INetworkModuleFactory)new FSRCNNFactory()).MakeDefault(),
            // ...
            "xor" => new MultilayerPerceptronFactory().MakeDefault(),
            _ => null
        };
        if (root is not null)
        {
            env.NetworkBlock.Add(root);

            if (root is ArchitectureBlock)
            {
                var arch = (ArchitectureBlock)root;
                if (arch.RequiredInputShape.HasValue)
                {
                    var tshape = arch.RequiredInputShape.Value;
                    env.InputShape = new Shape3D(tshape.LengthOrDefault(0), tshape.LengthOrDefault(1), tshape.LengthOrDefault(2));
                }
            }
        }
    }

    public override string ToString()
    {
        if (Identifier is null)
        {
            var ishape = manualInputShape ?? new Shape3D(1, 1, 1);
            return $"FROM SCRATCH INPUT {ishape.Channels} {ishape.Rows} {ishape.Columns}";
        }
        return $"FROM {Identifier}:{Version}";
    }
}