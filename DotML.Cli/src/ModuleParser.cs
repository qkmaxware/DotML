using DotML.Network;
using DotML.Network.IO;
using DotML.Network.IO.Netbuild;

namespace DotML.Cli;

public static class ModuleParser
{
    public static readonly string PreferredExtension = ".netdot";
    public static readonly string[] AllowedExtensions = [".netdot", ".netbuild"];

    public static INetworkModule Parse(string contents, string format)
    {
        return format switch
        {
            ".netbuild" => ParseNetBuild(contents),
            ".netdot" => ParseNetDot(contents),
            _ => throw new FormatException("format not supported")
        };
    }

    private static INetworkModule ParseNetDot(string contents)
    {
        var dotParser = new NetDot.Dot.Parser();
        var nodeGraph = dotParser.Parse(contents);

        var compiler = new NetDot.ModuleCompiler();
        return compiler.Compile(nodeGraph);
    }

    private static INetworkModule ParseNetBuild(string contents)
    {
        NetbuildSerializer builder = new NetbuildSerializer();

        var ast = builder.Parse(contents);
        var network = ast.MakeModule(
            new BuildEnvironment
            {
                Serializer = builder,
                ScopedNetworks = null // TODO maybe add scoped networks in again?
            },
            (index, count, statement) => { }
        );
        return network;
    }
}