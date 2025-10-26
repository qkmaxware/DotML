using DotML.Network.IO.Netbuild;

namespace DotML.Network.IO;

/// <summary>
/// Network construction language (Netbuild) serialization and deserialization
/// </summary>
public class NetbuildSerializer {
    private Tokenizer tokenizer;
    private Parser parser;

    private Dictionary<string, Func<string>> imports = new Dictionary<string, Func<string>>();

    public NetbuildSerializer() {
        this.tokenizer = new Tokenizer();
        this.parser = new Parser();
    }

    /// <summary>
    /// Allow the serializer to reference a buildscript which can be used in from statements
    /// </summary>
    /// <param name="alias">alias of the build script</param>
    /// <param name="version">version of the build scriot</param>
    /// <param name="script_reader">function that returns the input text of the build script</param>
    public void ReferencingScript(string alias, string? version, Func<string> script_reader) {
        if (string.IsNullOrEmpty(version)) {
            this.imports[alias] = script_reader;
        } else {
            this.imports[$"{alias}:{version}"] = script_reader;
        }
    }

    /// <summary>
    /// Just parse a netbuild file and don't attempt to deserialize the network yet
    /// </summary>
    /// <param name="text">script contents</param>
    /// <returns>File AST node</returns>
    public DotML.Network.IO.Netbuild.File Parse(string text) {
        var tokens = tokenizer.GetTokens(text);
        var ast = parser.ParseFile(tokens);

        return ast;
    }

    /// <summary>
    /// Deserialize the given reader's contents as a netbuild script
    /// </summary>
    /// <param name="reader">reader</param>
    /// <returns>neural network</returns>
    public INetworkModule DeserializeModule(TextReader reader) => DeserializeModule(reader.ReadToEnd());

    /// <summary>
    /// Deserialize the contents of a string as a netbuild script
    /// </summary>
    /// <param name="text">script contents</param>
    /// <returns>neural network</returns>
    public INetworkModule DeserializeModule(string text) {
        var tokens = tokenizer.GetTokens(text);
        var ast = parser.ParseFile(tokens);

        return ast.MakeModule(
            env: new BuildEnvironment {
                Serializer = this,
                ScopedNetworks = imports // TODO other files that can be imported in a FROM statement
            },
            stmt_action: null
        );
    }
}