using System.Data;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace DotML.Network.IO;

/// <summary>
/// Network construction language (NetBuild) utilities
/// </summary>
public class NetBuild {

    private static Regex _token_keyword_scratch = new Regex(@"\G\s*\b(?<value>SCRATCH)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _token_keyword_input = new Regex(@"\G\s*\b(?<value>INPUT)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _token_keyword_from = new Regex(@"\G\s*\b(?<value>FROM)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _token_keyword_add = new Regex(@"\G\s*\b(?<value>ADD)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _token_keyword_arg = new Regex(@"\G\s*\b(?<value>ARG)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _token_keyword_name = new Regex(@"\G\s*\b(?<value>NAME)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _token_keyword_remove = new Regex(@"\G\s*\b(?<value>REMOVE)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _token_keyword_replace = new Regex(@"\G\s*\b(?<value>REPLACE)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _token_keyword_as = new Regex(@"\G\s*\b(?<value>AS)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static Regex _token_keyword_with = new Regex(@"\G\s*\b(?<value>WITH)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled);

    private static Regex _token_literal_identifier = new Regex(@"\G\s*(?:(?<value>([a-zA-Z_][a-zA-Z_\-0-9]*))|""(?<value>(?:[^""\\]|\\.)*)""|'(?<value>(?:[^'\\]|\\.)*)')\s*", RegexOptions.Compiled);
    private static Regex _token_literal_number = new Regex(@"\G\s*(?<value>(?:\+|\-)?\d*(?:\.\d*)?)\s*", RegexOptions.Compiled);

    private static Regex _token_operator_spec = new Regex(@"\G\s*(?<value>:)\s*", RegexOptions.Compiled);
    private static Regex _token_operator_assign = new Regex(@"\G\s*(?<value>=)\s*", RegexOptions.Compiled);

    private static Regex _token_comment = new Regex(@"\G\s*#(?<value>[^\n]+)\s*", RegexOptions.Compiled);

    private Regex[] _all_tokens = [
        _token_keyword_from,
        _token_keyword_scratch,
        _token_keyword_input,
        _token_keyword_add,
        _token_keyword_arg,
        _token_keyword_name,
        _token_keyword_remove,
        _token_keyword_replace,
        _token_keyword_as,
        _token_keyword_with,

        _token_operator_spec,
        _token_operator_assign,

        _token_literal_identifier,
        _token_literal_number,

        _token_comment
    ];

    public class Token {
        public int Position;
        public Regex Type;
        public string Value;

        public Token(int position, Regex type, string value) {
            this.Position = position;
            this.Type = type;
            this.Value = value;
        }

        public override string ToString() {
            return $"{{Position: {Position}, Type: '{Type}', Value: '{Value}'}}";
        }
    }

    public List<Token> Tokenize(string text) {
        #region  Tokenizing
        var tokens = new List<Token>();
        var start_index = 0;
        while (start_index < text.Length) {
            bool was_matched = false;
            foreach (Regex type in _all_tokens) {
                var match = type.Match(text, start_index);
                if (!match.Success)
                    continue;
                var token = new Token(start_index, type, match.Groups["value"].Value);
                start_index += match.Length;
                was_matched = true;
                if (!ReferenceEquals(type, _token_comment))
                    tokens.Add(token); // Only add the token if it isn't a comment
                break;
            }

            if (!was_matched) {
                throw new SyntaxErrorException($"Invalid symbol/token '{text[start_index]}' at position {start_index}");
            }
        }
        #endregion
        return tokens;
    }

    /// <summary>
    /// Parse a given string of text into a network
    /// </summary>
    /// <param name="text">NetBuild source</param>
    /// <returns>created network</returns>
    /// <exception cref="SyntaxErrorException">thrown if there is a syntax exception</exception>
    public NetbuilderFile Parse(string text) {
        // Being lazy here, doesn't need to be a full-featured parser or very optimized lol

        // Tokenize
        var tokens = Tokenize(text);

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
        return file; // Make the network!
    }

    public FeedforwardNetwork ParseAndBuild(string text) {
        return Parse(text).Make(new BuildEnv());
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

    public delegate void StatementProcessingHandler(int current_statement_index, int total_statement_count, Stmt? current_statement);

    public class NetbuilderFile {
        public FromStmt? From;
        public List<Stmt> Commands = new List<Stmt>();

        public FeedforwardNetwork Make(StatementProcessingHandler? stmt_action = null) => Make(new BuildEnv(), stmt_action);

        public FeedforwardNetwork Make(BuildEnv env, StatementProcessingHandler? stmt_action = null) {
            var statment_count = 1 + Commands.Count;
            stmt_action?.Invoke(0, statment_count, From);
            From?.Action(env);
            if (env.Network is null)
                return new FeedforwardNetwork();
            
            var stmt_index = 1;
            foreach (var command in Commands) {
                stmt_action?.Invoke(stmt_index++, statment_count, command);
                command.Action(env);
            }

            return env.Network;
        }
    }

    public class BuildEnv {
        public Shape3D InputShape;
        public Dictionary<string, object> Arguments {get; private set;} = new Dictionary<string, object>();
        public Dictionary<string, int> LayerAliases {get; private set;} = new Dictionary<string, int>();
        public FeedforwardNetwork? Network {get; set;}
    }

    public abstract class Stmt {
        public abstract void Action(BuildEnv env);
    }

    public class FromStmt : Stmt {
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

            if (Enum.TryParse<T>(value: str, ignoreCase: true, out T result))
                return result;
            return @default;
        }

        public override void Action(BuildEnv env) {
            env.Network = (Identifier?.ToLower()) switch {
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

        public override string ToString() {
            return $"FROM {Identifier}:{Version}";
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

    public class NameStmt : Stmt {
        private string name;
        public NameStmt(string name) {
            this.name = name;
        }

        public override void Action(BuildEnv env) {
            var net = env.Network;
            if (net is not null)
                net.Name = name;
        }

        public override string ToString() {
            return $"NAME {name}";
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

    public abstract class Literal {
        public abstract object ValueOf();

        public string AsString() => Convert.ToString(ValueOf()) ?? string.Empty;
        public int AsInt() => Convert.ToInt32(ValueOf());
        public long AsLong() => Convert.ToInt64(ValueOf());
        public float AsFloat() => Convert.ToSingle(ValueOf());
        public double AsDouble() => Convert.ToDouble(ValueOf());
    }
    public class ObjectLiteral : Literal {
        private object obj;
        public ObjectLiteral(object o) {
            this.obj = o;
        }
        public override object ValueOf() => obj;

        public override string ToString() {
            return obj?.ToString() ?? string.Empty;
        }
    }

    public class ArgStmt : Stmt {

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

        public override string ToString() {
            return $"ARG {string.Join(' ', values.Select(kv => $"{kv.Key}={kv.Value}"))}";
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

    public class RemoveStmt : Stmt {
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

        public override string ToString() {
            return $"REMOVE {reference}";
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

    public class AddStmt : Stmt {
        string layer_name;
        Func<Shape3D, Dictionary<string, Literal>, IFeedforwardNetworkLayer> factory;
        string? ident;
        public Dictionary<string, Literal> arguments = new Dictionary<string, Literal>();

        public AddStmt(string layer_name, Func<Shape3D, Dictionary<string, Literal>, IFeedforwardNetworkLayer> factory, List<(Token, Literal)> args, string? alias) {
            this.layer_name = layer_name;
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

        public override string ToString() {
            if (ident is not null) {
                return $"ADD {layer_name} {string.Join(' ', arguments.Select(kv => $"{kv.Key}={kv.Value}"))} AS {ident}";
            } else {
                return $"ADD {layer_name} {string.Join(' ', arguments.Select(kv => $"{kv.Key}={kv.Value}"))}";
            }
        }
    }

    public class ReplaceStmt : Stmt {
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

        public override string ToString() {
            return $"REPLACE {reference} WITH ...";
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
    
        return new AddStmt(layer_type.Value, factory, args, alias);
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

    private static Literal One = new ObjectLiteral(1);
    private static Literal Zero = new ObjectLiteral(0);
    private static Literal Same = new ObjectLiteral("same");
    private static Literal FirstOf(Dictionary<string, Literal> args, Literal @default, params string[] tokens) {
        foreach (var token in tokens) {
            if (args.TryGetValue(token, out var literal)) {
                return literal;
            }
        }
        return @default;
    }
    private static Literal FirstOf(Dictionary<string, Literal> args, params string[] tokens) {
        foreach (var token in tokens) {
            if (args.TryGetValue(token, out var literal)) {
                return literal;
            }
        }
        throw new KeyNotFoundException(string.Join(',', tokens));
    }

    private ActivationFunction GetActivation(string name, double alpha) {
        return name.ToLower() switch {
            "step" => BinaryStep.Instance,
            "binarystep" => BinaryStep.Instance,
            "elu" => new ExponentialLU(alpha),
            "exponentiallu" => new ExponentialLU(alpha),
            "tanh" => HyperbolicTangent.Instance,
            "hyperbolictangent" => HyperbolicTangent.Instance,
            "id" => Identity.Instance,
            "identity" => Identity.Instance,
            "leaky-relu" => LeakyReLU.Instance,
            "leakyrelu" => LeakyReLU.Instance,
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
                var padding = Enum.Parse<Padding>(FirstOf(args, Same, "padding").AsString(), true);
                var x_stride = FirstOf(args, One, "stride-x", "stride").AsInt();
                var y_stride = FirstOf(args, One, "stride-y", "stride").AsInt();
                var filters = args["filters"].AsInt();
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
                var x_stride = FirstOf(args, One, "stride-x", "stride").AsInt();
                var y_stride = FirstOf(args, One, "stride-y", "stride").AsInt();
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
                var x_stride = FirstOf(args, One, "stride-x", "stride").AsInt();
                var y_stride = FirstOf(args, One, "stride-y", "stride").AsInt();
                var kernel = args["kernel"].AsInt();

                return new LocalAvgPoolingLayer(
                    input_size: ishape,
                    width: kernel,
                    height: kernel,
                    strideX: x_stride,
                    strideY: y_stride
                );
            }, 
            "flattening" => (Shape3D ishape, Dictionary<string, Literal> args) => {
                return new FlatteningLayer(ishape);
            },
            "dropout" => (Shape3D ishape, Dictionary<string, Literal> args) => {
                var dropout = args["percent"].AsDouble();

                return new DropoutLayer(ishape, dropout);
            },
            "layernorm" => (Shape3D ishape, Dictionary<string, Literal> args) => {
                return new LayerNorm(ishape);
            },
            "batchnorm" => (Shape3D ishape, Dictionary<string, Literal> args) => {
                return new BatchNorm(ishape);
            },
            "softmax" => (Shape3D ishape, Dictionary<string, Literal> args) => {
                return new SoftmaxLayer(ishape.Count);
            },
            "dense" => (Shape3D ishape, Dictionary<string, Literal> args) => {
                var neurons = args["neurons"].AsInt();

                return new DenseLinearLayer(
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