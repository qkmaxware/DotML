using System.Data;
using System.Text.RegularExpressions;

namespace DotML.Network.IO;

/// <summary>
/// Network construction language (NetBuild) utilities
/// </summary>
public class NetBuild {

    private static Regex _token_keyword_scratch = new Regex(@"\s*\b(SCRATCH)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _token_keyword_input = new Regex(@"\s*\b(INPUT)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _token_keyword_from = new Regex(@"\s*\b(FROM)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _token_keyword_add = new Regex(@"\s*\b(ADD)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _token_keyword_arg = new Regex(@"\s*\b(ARG)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _token_keyword_name = new Regex(@"\s*\b(NAME)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _token_keyword_remove = new Regex(@"\s*\b(REMOVE)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _token_keyword_replace = new Regex(@"\s*\b(REPLACE)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _token_keyword_as = new Regex(@"\s*\b(AS)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _token_keyword_with = new Regex(@"\s*\b(WITH)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);

    private static Regex _token_literal_identifier = new Regex(@"\s*([a-zA-Z_][a-zA-Z_\-0-9]*)\s*|\s*""((?:[^""\\]|\\.)*)""\s*", RegexOptions.Compiled);
    private static Regex _token_literal_number = new Regex(@"\s*((?:\+|\-)\d*(?:\.\d*)?)\s*", RegexOptions.Compiled);

    private static Regex _token_operator_spec = new Regex(@"\s*(:)\s*", RegexOptions.Compiled);
    private static Regex _token_operator_assign = new Regex(@"\s*(=)\s*", RegexOptions.Compiled);

    private Regex[] _all_tokens = [
        _token_keyword_scratch,
        _token_keyword_input,
        _token_keyword_from,
        _token_keyword_add,
        _token_keyword_arg,
        _token_keyword_name,
        _token_keyword_remove,
        _token_keyword_replace,
        _token_keyword_as,
        _token_keyword_with,
        _token_literal_identifier,
        _token_literal_number,
        _token_operator_spec,
        _token_operator_assign
    ];

    private class Token {
        public int Position;
        public Regex Type;
        public string Value;

        public Token(int position, Regex type, string value) {
            this.Position = position;
            this.Type = type;
            this.Value = value;
        }
    }

    /// <summary>
    /// Parse a given string of text into a network
    /// </summary>
    /// <param name="text">NetBuild source</param>
    /// <returns>created network</returns>
    /// <exception cref="SyntaxErrorException">thrown if there is a syntax exception</exception>
    public FeedforwardNetwork Parse(string text) {
        // Being lazy here, doesn't need to be a full-featured parser or very optimized lol

        // Tokenize
        #region  Tokenizing
        var tokens = new List<Token>();
        var start_index = 0;
        while (start_index < text.Length) {
            bool was_matched = false;
            foreach (Regex type in _all_tokens) {
                var match = type.Match(text, start_index);
                if (!match.Success)
                    continue;
                var token = new Token(start_index, type, match.Groups[1].Value);
                tokens.Add(token);
                start_index += match.Length;
                was_matched = true;
                Console.WriteLine("Matched " + token.Value);
                break;
            }

            if (!was_matched) {
                throw new SyntaxErrorException($"Invalid symbol/token '{text[start_index]}' at position {start_index}");
            }
        }
        #endregion

        // Parse
        int lookahead = 0;
        var file = new NetbuilderFile {
            From = ParseSrc(ref lookahead, tokens),
        };
        while (lookahead < tokens.Count) {
            var stmt = ParseCommand(ref lookahead, tokens);
            file.Commands.Add(stmt);
        }

        // Process
        var env = new BuildEnv();
        return file.Make(env); // Make the network!
    }

    #region Parsing
    private static int getPosition(int lookahead, List<Token> tokens) {
        if (lookahead < tokens.Count)
            return tokens[lookahead].Position;
        else 
            if (tokens.Count > 0)
                return tokens[^1].Position;
            else 
                return 0;
    }
    private static bool isLookahead(int lookahead, List<Token> tokens, Regex type) {
        if (lookahead >= tokens.Count)
            return false;
        
        if (!object.ReferenceEquals(tokens[lookahead].Type, type)) {
            return false;
        }

        return true;
    }

    class NetbuilderFile {
        public FromStmt? From;
        public List<Stmt> Commands = new List<Stmt>();

        public FeedforwardNetwork Make(BuildEnv env) {
            From?.Action(env);
            if (env.Network is null)
                return new FeedforwardNetwork();
            
            foreach (var command in Commands) {
                command.Action(env);
            }

            return env.Network;
        }
    }

    class BuildEnv {
        public Shape3D InputShape;
        public Dictionary<string, object> Arguments {get; private set;} = new Dictionary<string, object>();
        public Dictionary<string, int> LayerAliases {get; private set;} = new Dictionary<string, int>();
        public FeedforwardNetwork? Network {get; set;}
    }

    abstract class Stmt {
        public abstract void Action(BuildEnv env);
    }

    class FromStmt : Stmt {
        private Shape3D? manualInputShape;
        public string? Identifier;
        public string? Version;

        public FromStmt(Shape3D? input_shape, string? identifier, string? version) {
            this.manualInputShape = input_shape;
            this.Identifier = identifier;
            this.Version = version;
        }

        private static T make_enum<T>(string? str, T @default) where T:struct {
            if (string.IsNullOrEmpty(str))
                return @default;

            if (Enum.TryParse<T>(str, out T result))
                return result;
            return @default;
        }

        public override void Action(BuildEnv env) {
            env.Network = (Identifier) switch {
                // Named networks!
                "lenet"     => LeNet.Make(make_enum<LeNet.Version>(Version, LeNet.Version.Latest), output_classes: LeNet.OUT_CLASSES),
                "alexnet"   => AlexNet.Make(make_enum<AlexNet.Version>(Version, AlexNet.Version.Latest), output_classes: AlexNet.OUT_CLASSES),
                "mobilenet" => MobileNet.Make(make_enum<MobileNet.Version>(Version, MobileNet.Version.Latest), output_classes: MobileNet.OUT_CLASSES),
                "vgg"       => VGGNet.Make(make_enum<VGGNet.Version>(Version, VGGNet.Version.Latest), output_classes: VGGNet.OUT_CLASSES),

                // Scratch
                _           => new FeedforwardNetwork()
            };
            if (env.Network.LayerCount < 1) {
                env.InputShape = manualInputShape ?? new Shape3D(1,1,1);
            } else {
                env.InputShape = env.Network.InputShape;
            }
        }
    }

    private FromStmt ParseSrc(ref int lookahead, List<Token> tokens) {
        if (!isLookahead(lookahead, tokens, _token_keyword_from)) {
            throw new SyntaxErrorException($"Missing FROM keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        lookahead++;

        Token? identifier = null;
        Shape3D? ishape = null;
        if (isLookahead(lookahead, tokens, _token_keyword_scratch)) {
            identifier = tokens[lookahead];
            lookahead++;

            // Requires the input size to be specified
            if (!isLookahead(lookahead, tokens, _token_keyword_input)) {
                throw new SyntaxErrorException($"Missing INPUT keyword at position {getPosition(lookahead, tokens)}"); // Missing "INPUT" keyword
            }
            lookahead++;

            // Get the size channels, rows, columns in that order
            var channels = ParseLiteral(ref lookahead, tokens);
            var rows = ParseLiteral(ref lookahead, tokens);
            var columns = ParseLiteral(ref lookahead, tokens);

            ishape = new Shape3D(channels.AsInt(), rows.AsInt(), columns.AsInt());
        }
        else if (isLookahead(lookahead, tokens, _token_literal_identifier)) {
            identifier = tokens[lookahead];
            lookahead++;
        }
        else {
            throw new SyntaxErrorException($"Missing SCRATCH keyword or base network IDENTIFIER at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword // Missing SCRATCH or id
        }

        Token? version_spec = null;
        if (isLookahead(lookahead, tokens, _token_operator_spec)) {
            lookahead++;
            if (!isLookahead(lookahead, tokens, _token_literal_identifier)) {
                throw new SyntaxErrorException($"Missing VERSION identifier at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword // Missing version number
            }
            version_spec = tokens[lookahead];
            lookahead++;
        }

        return new FromStmt(ishape, identifier.Value.ToLower(), version_spec?.Value);
    }

    private Stmt ParseCommand(ref int lookahead, List<Token> tokens) {
        if (isLookahead(lookahead, tokens, _token_keyword_arg)) {
            return ParseArg(ref lookahead, tokens);
        } 
        else if (isLookahead(lookahead, tokens, _token_keyword_name)) {
            return ParseName(ref lookahead, tokens);
        }
        else if (isLookahead(lookahead, tokens, _token_keyword_remove)) {
            return ParseRemove(ref lookahead, tokens);
        }
        else if (isLookahead(lookahead, tokens, _token_keyword_add)) {
            return ParseAdd(ref lookahead, tokens);
        }
        else if (isLookahead(lookahead, tokens, _token_keyword_replace)) {
            return ParseReplace(ref lookahead, tokens);
        }
         
        else {
            throw new SyntaxErrorException("Expecting one of ARG, ADD, NAME, REMOVE, or REPLACE");
        }
    }

    class NameStmt : Stmt {
        private string name;
        public NameStmt(string name) {
            this.name = name;
        }

        public override void Action(BuildEnv env) {
            var net = env.Network;
            if (net is not null)
                net.Name = name;
        }
    }

    private Stmt ParseName(ref int lookahead, List<Token> tokens) {
        if (!isLookahead(lookahead, tokens, _token_keyword_name)) {
            throw new SyntaxErrorException($"Missing NAME keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        lookahead++;

        if (!isLookahead(lookahead, tokens, _token_literal_identifier)) {
            throw new SyntaxErrorException($"Missing identifier at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        var token = tokens[lookahead];
        lookahead++;

        NameStmt stmt = new NameStmt(token.Value);

        return stmt;
    }

    abstract class Literal {
        public abstract object ValueOf();

        public string AsString() => Convert.ToString(ValueOf()) ?? string.Empty;
        public int AsInt() => Convert.ToInt32(ValueOf());
        public long AsLong() => Convert.ToInt64(ValueOf());
        public float AsFloat() => Convert.ToSingle(ValueOf());
        public double AsDouble() => Convert.ToDouble(ValueOf());
    }
    class ObjectLiteral : Literal {
        private object obj;
        public ObjectLiteral(object o) {
            this.obj = o;
        }
        public override object ValueOf() => obj;
    }

    class ArgStmt : Stmt {

        private Dictionary<string, Literal> values = new Dictionary<string, Literal>();

        public ArgStmt() { }

        public void Set(Token key, Literal value) {
            this.values[key.Value] = value;
        }

        public override void Action(BuildEnv env) {
            foreach (var arg in values) {
                env.Arguments[arg.Key] = arg.Value.ValueOf();
            }
        }
    }

    private Stmt ParseArg(ref int lookahead, List<Token> tokens) {
        if (!isLookahead(lookahead, tokens, _token_keyword_arg)) {
            throw new SyntaxErrorException($"Missing ARG keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        lookahead++;

        ArgStmt stmt = new ArgStmt();

        while (isLookahead(lookahead, tokens, _token_literal_identifier) && isLookahead(lookahead + 1, tokens, _token_operator_assign)) {
            var (ident, value) = ParseAssignment(ref lookahead, tokens);
            stmt.Set(ident, value);
        }

        return stmt;
    }

    private (Token, Literal) ParseAssignment(ref int lookahead, List<Token> tokens) {
        if (!isLookahead(lookahead, tokens, _token_literal_identifier) || !isLookahead(lookahead + 1, tokens, _token_operator_assign)) {
            throw new SyntaxErrorException("Expecting an assignment of the form 'name=value'");
        }
        // Name
        var ident = tokens[lookahead];
        lookahead+=2;

        // Value
        var value = ParseLiteral(ref lookahead, tokens);

        return (ident, value);
    }

    private Literal ParseLiteral(ref int lookahead, List<Token> tokens) {
        if (isLookahead(lookahead, tokens, _token_literal_identifier)) {
            var token = tokens[lookahead++];
            return new ObjectLiteral(token.Value);
        }
        else if (isLookahead(lookahead, tokens, _token_literal_number)) {
            var token = tokens[lookahead++];
            return new ObjectLiteral(double.Parse(token.Value));
        } 
        else {
            throw new SyntaxErrorException("Expecting one of STRING, IDENTIFIER, or NUMBER");
        }
    }

    abstract class LayerReference {
        public abstract int IndexOf(Dictionary<string, int> aliases);
    }
    class NamedLayerReference : LayerReference {
        private string index;
        public NamedLayerReference(string index) => this.index = index;
        public override int IndexOf(Dictionary<string, int> aliases) => aliases[this.index];
    }
    class IndexedLayerReference : LayerReference {
        private int index;
        public IndexedLayerReference(int index) => this.index = index;
        public override int IndexOf(Dictionary<string, int> aliases) => index;
    }

    class RemoveStmt : Stmt {
        private LayerReference reference;
        public RemoveStmt(LayerReference reference) {
            this.reference = reference;
        }

        public override void Action(BuildEnv env) {
            var network = env.Network;
            if (network is null)
                return;
            
            network.RemoveLayer(reference.IndexOf(env.LayerAliases));
        }
    }

    private Stmt ParseRemove(ref int lookahead, List<Token> tokens) {
        if (!isLookahead(lookahead, tokens, _token_keyword_remove)) {
            throw new SyntaxErrorException($"Missing REMOVE keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        lookahead++;

        var reference = ParseLayerReference(ref lookahead, tokens);

        RemoveStmt stmt = new RemoveStmt(reference);

        return stmt;
    }

    private LayerReference ParseLayerReference(ref int lookahead, List<Token> tokens) {
        if (isLookahead(lookahead, tokens, _token_literal_identifier)) {
            var token = tokens[lookahead++];
            return new NamedLayerReference(token.Value);
        }
        else if (isLookahead(lookahead, tokens, _token_literal_number)) {
            var token = tokens[lookahead++];
            return new IndexedLayerReference(int.Parse(token.Value));
        } 
        else {
            throw new SyntaxErrorException("Expecting one of STRING, IDENTIFIER, or NUMBER");
        }
    }

    class AddStmt : Stmt {
        Func<Shape3D, Dictionary<string, Literal>, IFeedforwardNetworkLayer> factory;
        string? ident;
        public Dictionary<string, Literal> arguments = new Dictionary<string, Literal>();

        public AddStmt(Func<Shape3D, Dictionary<string, Literal>, IFeedforwardNetworkLayer> factory, List<(Token, Literal)> args, string? alias) {
            this.factory = factory;
            foreach (var pair in args) {
                arguments[pair.Item1.Value] = pair.Item2;
            } 
            this.ident = alias;
        }


        public override void Action(BuildEnv env) {
            var network = env.Network;
            if (network is null)
                return;
            
            var output_shape = network.LayerCount > 0 ? network.OutputShape : env.InputShape;
            IFeedforwardNetworkLayer layer = factory(output_shape, arguments);
            var index = network.LayerCount;
            network.AddLayer(layer);

            if (!string.IsNullOrEmpty(ident)) {
                env.LayerAliases[ident] = index;
            }
        }
    }

    class ReplaceStmt : Stmt {
        LayerReference reference;
        Func<Shape3D, Dictionary<string, Literal>, IFeedforwardNetworkLayer> factory;
        public Dictionary<string, Literal> arguments = new Dictionary<string, Literal>();

        public ReplaceStmt(LayerReference reference, Func<Shape3D, Dictionary<string, Literal>, IFeedforwardNetworkLayer> factory, List<(Token, Literal)> args) {
            this.factory = factory;
            foreach (var pair in args) {
                arguments[pair.Item1.Value] = pair.Item2;
            } 
            this.reference = reference;
        }

        public override void Action(BuildEnv env) {
            var network = env.Network;
            if (network is null)
                return;
            
            var replacement_index = reference.IndexOf(env.LayerAliases);
            var input_shape = network.GetLayer(replacement_index).InputShape;
            IFeedforwardNetworkLayer layer = factory(input_shape, arguments);
            network.ReplaceLayer(replacement_index, layer);
        }
    }

    private Stmt ParseAdd(ref int lookahead, List<Token> tokens) {
        if (!isLookahead(lookahead, tokens, _token_keyword_add)) {
            throw new SyntaxErrorException($"Missing ADD keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        lookahead++;
        
        // Get the layer type
        if (!isLookahead(lookahead, tokens, _token_literal_identifier)) {
            throw new SyntaxErrorException($"Missing IDENTIFIER keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        var layer_type = tokens[lookahead++];
        var factory = GetLayerFactory(layer_type.Value.ToLower(), layer_type.Position);

        // Get the layer arguments
        var args = new List<(Token, Literal)>();
        while (isLookahead(lookahead, tokens, _token_literal_identifier) && isLookahead(lookahead + 1, tokens, _token_operator_assign)) {
            args.Add(ParseAssignment(ref lookahead, tokens));
        }

        // Custom alias
        string? alias = null;
        if (isLookahead(lookahead, tokens, _token_keyword_as)) {
            lookahead++;
            if (!isLookahead(lookahead, tokens, _token_literal_identifier)) {
                throw new SyntaxErrorException($"Missing IDENTIFIER keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
            }
            alias = tokens[lookahead++].Value;
        }
    
        return new AddStmt(factory, args, alias);
    }

    private Stmt ParseReplace(ref int lookahead, List<Token> tokens) {
        if (!isLookahead(lookahead, tokens, _token_keyword_replace)) {
            throw new SyntaxErrorException($"Missing REPLACE keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        lookahead++;

        // Get the layer index
        var reference = ParseLayerReference(ref lookahead, tokens);

        if (!isLookahead(lookahead, tokens, _token_keyword_with)) {
            throw new SyntaxErrorException($"Missing WITH keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        lookahead++;
        
        // Get the layer type
        if (!isLookahead(lookahead, tokens, _token_literal_identifier)) {
            throw new SyntaxErrorException($"Missing IDENTIFIER keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        var layer_type = tokens[lookahead++];
        var factory = GetLayerFactory(layer_type.Value.ToLower(), layer_type.Position);

        // Get the layer arguments
        var args = new List<(Token, Literal)>();
        while (isLookahead(lookahead, tokens, _token_literal_identifier) && isLookahead(lookahead + 1, tokens, _token_operator_assign)) {
            args.Add(ParseAssignment(ref lookahead, tokens));
        }
    
        return new ReplaceStmt(reference, factory, args);
    }

    #endregion

    #region Object Mapping

    private ActivationFunction GetActivation(string name, double alpha) {
        return name switch {
            "step" => BinaryStep.Instance,
            "elu" => new ExponentialLU(alpha),
            "tanh" => HyperbolicTangent.Instance,
            "id" => Identity.Instance,
            "leaky-relu" => LeakyReLU.Instance,
            "prelu" => new PReLU(alpha),
            "relu" => ReLU.Instance,
            "sigmoid" => Sigmoid.Instance,
            "telu" => TeLU.Instance,
            _ => throw new ArgumentException($"Unknown activation function {name}")
        };
    }

    private Func<Shape3D, Dictionary<string, Literal>, IFeedforwardNetworkLayer> GetLayerFactory(string layer_name, int position) {
        Func<Shape3D, Dictionary<string, Literal>, IFeedforwardNetworkLayer> factory = layer_name switch {
            "convolution" => (Shape3D ishape, Dictionary<string, Literal> args) => {
                var padding = args.ContainsKey("padding") ? Enum.Parse<Padding>(args["padding"].AsString(), true) : Padding.Same;
                var x_stride = args.ContainsKey("stride") ? args["stride"].AsInt() : 1;
                var y_stride = args.ContainsKey("stride") ? args["stride"].AsInt() : 1;
                var filters = args["filter"].AsInt();
                var kernel_size = args["kernel"].AsInt();

                return new ConvolutionLayer(
                    input_size: ishape,
                    padding: padding,
                    strideX: x_stride,
                    strideY: y_stride,
                    filters: ConvolutionFilter.Make(
                        filters: filters,
                        kernels_per_filter: ishape.Channels,
                        kernel_size: kernel_size
                    ) 
                );
            },
            "activation" => (Shape3D ishape, Dictionary<string, Literal> args) => {
                var activation = GetActivation(
                    args["fn"].AsString().ToLower(),
                    args.ContainsKey("alpha") ? args["alpha"].AsDouble() : 0.0
                );

                return new ActivationLayer(ishape, activation);
            },
            "maxpool" => (Shape3D ishape, Dictionary<string, Literal> args) => {
                var x_stride = args.ContainsKey("stride") ? args["stride"].AsInt() : 1;
                var y_stride = args.ContainsKey("stride") ? args["stride"].AsInt() : 1;
                var kernel = args["kernel"].AsInt();

                return new LocalMaxPoolingLayer(
                    input_size: ishape,
                    width: kernel,
                    height: kernel,
                    strideX: x_stride,
                    strideY: y_stride
                );
            }, 
            "avgpool" => (Shape3D ishape, Dictionary<string, Literal> args) => {
                var x_stride = args.ContainsKey("stride") ? args["stride"].AsInt() : 1;
                var y_stride = args.ContainsKey("stride") ? args["stride"].AsInt() : 1;
                var kernel = args["kernel"].AsInt();

                return new LocalAvgPoolingLayer(
                    input_size: ishape,
                    width: kernel,
                    height: kernel,
                    strideX: x_stride,
                    strideY: y_stride
                );
            }, 
            "dense" => (Shape3D ishape, Dictionary<string, Literal> args) => {
                var neurons = args["neurons"].AsInt();

                return new FullyConnectedLayer(
                    input_size: ishape.Count,
                    neurons: neurons
                );
            }, 

            _ => throw new SyntaxErrorException($"Unknown layer type {layer_name} at position {position}")
        };

        return factory;
    }

    #endregion

    #region Emitting

    public void Emit() {}

    #endregion
}