using System.Xml;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.IO.Netbuild;
using DotML.Serialization.Xml;

namespace DotML.Cli;

public static class ModuleParser
{
    public static readonly string PreferredExtension = ".netdot";
    public static readonly string[] AllowedExtensions = [".netdot", ".netxml", ".netbuild"];

    public static INetworkModule Parse(string contents, string format)
    {
        return format switch
        {
            ".netbuild" => ParseNetBuild(contents),
            ".netdot" => ParseNetDot(contents),
            ".netxml" => ParseNetXml(contents),
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

    private static INetworkModule ParseNetXml(string contents)
    {
        var parser = new NetworkXmlSerializer();
        var doc = new XmlDocument();
        doc.InnerXml = contents;
        return parser.Deserialize(doc);
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