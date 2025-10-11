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

    public override void Action(BuildEnvironment env) {
        env.Network = (Identifier?.ToLower()) switch {
            // Named networks!
            "alexnet"   => AlexNet.Make(make_enum<AlexNet.Version>(Version, AlexNet.Version.Latest), output_classes: AlexNet.OUT_CLASSES),
            "espcn"     => ESPCN.Make(make_enum<ESPCN.Version>(Version, ESPCN.Version.Latest)),
            "fsrcnn"    => FSRCNN.Make(make_enum<FSRCNN.Version>(Version, FSRCNN.Version.Latest)),
            "lenet"     => LeNet.Make(make_enum<LeNet.Version>(Version, LeNet.Version.Latest), output_classes: LeNet.OUT_CLASSES),
            "mobilenet" => MobileNet.Make(make_enum<MobileNet.Version>(Version, MobileNet.Version.Latest), output_classes: MobileNet.OUT_CLASSES),
            "resnet"    => ResNet.Make(make_enum<ResNet.Version>(Version, ResNet.Version.Latest)),
            "vgg"       => VGGNet.Make(make_enum<VGGNet.Version>(Version, VGGNet.Version.Latest), output_classes: VGGNet.OUT_CLASSES),

            // Scratch
            null           => new FeedforwardNetwork(),

            // Other networks in the environment
            _              => key is not null && env.Serializer is not null && env.ScopedNetworks is not null && env.ScopedNetworks.ContainsKey(key) ? env.Serializer.Deserialize(env.ScopedNetworks[key]()) : new FeedforwardNetwork(),
        };
        if (env.Network.LayerCount < 1) {
            env.InputShape = manualInputShape ?? new Shape3D(1,1,1);
        } else {
            env.InputShape = env.Network.InputShape;
        }
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