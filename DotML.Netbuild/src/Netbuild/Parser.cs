using System.Data;

namespace DotML.Network.IO.Netbuild;

internal class Parser {

    private ActivationFunctionMapper activations;
    private LayerMapper layers;

    public Parser() {
        this.activations = new ActivationFunctionMapper();
        this.layers = new LayerMapper(activations);
    }

    private static int getPosition(int lookahead, List<Token> tokens) {
        if (lookahead < tokens.Count)
            return tokens[lookahead].Position;
        else 
            if (tokens.Count > 0)
                return tokens[^1].Position;
            else 
                return 0;
    }
    private static bool isLookahead<T>(int lookahead, List<Token> tokens) {
        if (lookahead >= tokens.Count)
            return false;
        
        if (tokens[lookahead].Type is not T) {
            return false;
        }

        return true;
    }

    public File ParseFile(List<Token> tokens) {
        // Parse
        int lookahead = 0;
        var file = new File(ParsePreamble(ref lookahead, tokens));
        while (lookahead < tokens.Count) {
            var stmt = ParseStatements(ref lookahead, tokens);
            file.Statements.Add(stmt);
        }

        // Process
        return file; // Make the network!
    }
    
    public Preamble ParsePreamble(ref int lookahead, List<Token> tokens) {
        var src = ParseSource(ref lookahead, tokens);

        return new Preamble(src);
    }
    public FromStatement ParseSource(ref int lookahead, List<Token> tokens) {
        if (!isLookahead<KeywordFrom>(lookahead, tokens)) {
            throw new SyntaxErrorException($"Missing FROM keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        lookahead++;

        Token<string>? identifier = null;
        Shape3D? ishape = null;
        bool is_scratch = false;
        if (isLookahead<KeywordScratch>(lookahead, tokens)) {
            identifier = (Token<string>)tokens[lookahead];
            lookahead++;
            is_scratch = true;

            // Requires the input size to be specified
            if (!isLookahead<KeywordInput>(lookahead, tokens)) {
                throw new SyntaxErrorException($"Missing INPUT keyword at position {getPosition(lookahead, tokens)}"); // Missing "INPUT" keyword
            }
            lookahead++;

            // Get the size channels, rows, columns in that order
            var channels = ParseLiteral(ref lookahead, tokens);
            var rows = ParseLiteral(ref lookahead, tokens);
            var columns = ParseLiteral(ref lookahead, tokens);

            ishape = new Shape3D(channels.AsInt(), rows.AsInt(), columns.AsInt());
        }
        else if (isLookahead<Identifier>(lookahead, tokens)) {
            identifier = (Token<string>)tokens[lookahead];
            lookahead++;
        }
        else {
            throw new SyntaxErrorException($"Missing SCRATCH keyword or base network IDENTIFIER at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword // Missing SCRATCH or id
        }

        Token<string>? version_spec = null;
        if (isLookahead<OperatorColon>(lookahead, tokens)) {
            lookahead++;
            if (!isLookahead<Identifier>(lookahead, tokens)) {
                throw new SyntaxErrorException($"Missing VERSION identifier at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword // Missing version number
            }
            version_spec = (Token<string>)tokens[lookahead];
            lookahead++;
        }

        return new FromStatement(ishape, is_scratch ? null : identifier.Value.ToLower(), version_spec?.Value);
    }

    public Statement ParseStatements(ref int lookahead, List<Token> tokens) {
        if (isLookahead<KeywordArg>(lookahead, tokens)) {
            return ParseArg(ref lookahead, tokens);
        } 
        else if (isLookahead<KeywordName>(lookahead, tokens)) {
            return ParseName(ref lookahead, tokens);
        }
        else if (isLookahead<KeywordRemove>(lookahead, tokens)) {
            return ParseRemove(ref lookahead, tokens);
        }
        else if (isLookahead<KeywordAdd>(lookahead, tokens)) {
            return ParseAdd(ref lookahead, tokens);
        }
        else if (isLookahead<KeywordReplace>(lookahead, tokens)) {
            return ParseReplace(ref lookahead, tokens);
        }
        else if (isLookahead<KeywordInsert>(lookahead, tokens)) {
            return ParseInsert(ref lookahead, tokens);
        }
        else if (isLookahead<KeywordPretrain>(lookahead, tokens)) {
            return ParsePretrain(ref lookahead, tokens);
        }
         
        else {
            throw new SyntaxErrorException("Expecting one of NAME, ARG, ADD, REMOVE, REPLACE, INSERT BEFORE, INSERT AFTER, or PRETRAIN");
        }
    }
    public NameStatement ParseName(ref int lookahead, List<Token> tokens) {
        if (!isLookahead<KeywordName>(lookahead, tokens)) {
            throw new SyntaxErrorException($"Missing NAME keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        lookahead++;

        if (!isLookahead<Identifier>(lookahead, tokens)) {
            throw new SyntaxErrorException($"Missing identifier at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        var token = (Token<string>)tokens[lookahead];
        lookahead++;

        NameStatement stmt = new NameStatement(token.Value);

        return stmt;
    }
    public ArgStatement ParseArg(ref int lookahead, List<Token> tokens) {
        if (!isLookahead<KeywordArg>(lookahead, tokens)) {
            throw new SyntaxErrorException($"Missing ARG keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        lookahead++;

        ArgStatement stmt = new ArgStatement();

        while (isLookahead<Identifier>(lookahead, tokens) && isLookahead<OperatorAssign>(lookahead + 1, tokens)) {
            var (ident, value) = ParseArgumentAssignment(ref lookahead, tokens);
            stmt.Set(ident, value);
        }

        return stmt;
    }
    public (Token<string>, Literal) ParseArgumentAssignment(ref int lookahead, List<Token> tokens) {
        if (!isLookahead<Identifier>(lookahead, tokens) || !isLookahead<OperatorAssign>(lookahead + 1, tokens)) {
            throw new SyntaxErrorException("Expecting an assignment of the form 'name=value'");
        }
        // Name
        var ident = (Token<string>)tokens[lookahead];
        lookahead+=2;

        // Value
        var value = ParseLiteral(ref lookahead, tokens);

        return (ident, value);
    }
    public AddStatement ParseAdd(ref int lookahead, List<Token> tokens) {
        if (!isLookahead<KeywordAdd>(lookahead, tokens)) {
            throw new SyntaxErrorException($"Missing ADD keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        lookahead++;
        
        // Get the layer type
        if (!isLookahead<Identifier>(lookahead, tokens)) {
            throw new SyntaxErrorException($"Missing IDENTIFIER keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        var layer_type = (Token<string>)tokens[lookahead++];
        Func<Shape3D, Dictionary<string, Literal>, IFeedforwardNetworkLayer> factory = (ishape, args) => layers.Decode(layer_type.Position, layer_type.Value, ishape, args);

        // Get the layer arguments
        var args = new List<(Token<string>, Literal)>();
        while (isLookahead<Identifier>(lookahead, tokens) && isLookahead<OperatorAssign>(lookahead + 1, tokens)) {
            args.Add(ParseArgumentAssignment(ref lookahead, tokens));
        }

        // Custom alias
        string? alias = null;
        if (isLookahead<KeywordAs>(lookahead, tokens)) {
            lookahead++;
            if (!isLookahead<Identifier>(lookahead, tokens)) {
                throw new SyntaxErrorException($"Missing IDENTIFIER keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
            }
            alias = ((Token<string>)tokens[lookahead++]).Value;
        }
    
        return new AddStatement(layer_type.Value, factory, args, alias);
    }
    public RemoveStatement ParseRemove(ref int lookahead, List<Token> tokens) {
        if (!isLookahead<KeywordRemove>(lookahead, tokens)) {
            throw new SyntaxErrorException($"Missing REMOVE keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        lookahead++;

        var reference = ParseLayerReference(ref lookahead, tokens);

        RemoveStatement stmt = new RemoveStatement(reference);

        return stmt;
    }
    public ReplaceStatement ParseReplace(ref int lookahead, List<Token> tokens) {
        if (!isLookahead<KeywordReplace>(lookahead, tokens)) {
            throw new SyntaxErrorException($"Missing REPLACE keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        lookahead++;

        // Get the layer index
        var reference = ParseLayerReference(ref lookahead, tokens);

        if (!isLookahead<KeywordWith>(lookahead, tokens)) {
            throw new SyntaxErrorException($"Missing WITH keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        lookahead++;
        
        // Get the layer type
        if (!isLookahead<Identifier>(lookahead, tokens)) {
            throw new SyntaxErrorException($"Missing IDENTIFIER keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        var layer_type = (Token<string>)tokens[lookahead++];
        Func<Shape3D, Dictionary<string, Literal>, IFeedforwardNetworkLayer> factory = (ishape, args) => layers.Decode(layer_type.Position, layer_type.Value, ishape, args);

        // Get the layer arguments
        var args = new List<(Token<string>, Literal)>();
        while (isLookahead<Identifier>(lookahead, tokens) && isLookahead<OperatorAssign>(lookahead + 1, tokens)) {
            args.Add(ParseArgumentAssignment(ref lookahead, tokens));
        }
    
        return new ReplaceStatement(reference, layer_type.Value, factory, args);
    }
    public Statement ParseInsert(ref int lookahead, List<Token> tokens) {
        if (!isLookahead<KeywordInsert>(lookahead, tokens)) {
            throw new SyntaxErrorException($"Missing INSERT keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        lookahead++;

        // Get the layer type
        if (!isLookahead<Identifier>(lookahead, tokens)) {
            throw new SyntaxErrorException($"Missing IDENTIFIER keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        var layer_type = (Token<string>)tokens[lookahead++];
        Func<Shape3D, Dictionary<string, Literal>, IFeedforwardNetworkLayer> factory = (ishape, args) => layers.Decode(layer_type.Position, layer_type.Value, ishape, args);

        // Get the layer arguments
        var args = new List<(Token<string>, Literal)>();
        while (isLookahead<Identifier>(lookahead, tokens) && isLookahead<OperatorAssign>(lookahead + 1, tokens)) {
            args.Add(ParseArgumentAssignment(ref lookahead, tokens));
        }

        if (isLookahead<KeywordBefore>(lookahead, tokens)) {
            lookahead++;
            var reference = ParseLayerReference(ref lookahead, tokens);
            InsertBeforeStatement stmt = new InsertBeforeStatement(reference, layer_type.Value, factory, args);
            return stmt;
        }
        else if (isLookahead<KeywordAfter>(lookahead, tokens)) {
            lookahead++;
            var reference = ParseLayerReference(ref lookahead, tokens);
            InsertAfterStatement stmt = new InsertAfterStatement(reference, layer_type.Value, factory, args);
            return stmt;
        } else {
            throw new SyntaxErrorException($"Missing BEFORE or AFTER keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
    }
    public PretrainStatement ParsePretrain(ref int lookahead, List<Token> tokens) {
        if (!isLookahead<KeywordPretrain>(lookahead, tokens)) {
            throw new SyntaxErrorException($"Missing PRETRAIN keyword at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        lookahead++;

        if (!isLookahead<Identifier>(lookahead, tokens)) {
            throw new SyntaxErrorException($"Missing path at position {getPosition(lookahead, tokens)}"); // Missing "FROM" keyword
        }
        var token = (Token<string>)tokens[lookahead];
        lookahead++;

        PretrainStatement stmt = new PretrainStatement(token.Value);

        return stmt;
    }

    public LayerReference ParseLayerReference(ref int lookahead, List<Token> tokens) {
        if (isLookahead<Identifier>(lookahead, tokens)) {
            var token = (Token<string>)tokens[lookahead++];
            return new NamedLayerReference(token.Value);
        }
        else if (isLookahead<Number>(lookahead, tokens)) {
            var token = (Token<string>)tokens[lookahead++];
            return new IndexedLayerReference(int.Parse(token.Value));
        } 
        else {
            throw new SyntaxErrorException("Expecting one of STRING, IDENTIFIER, or NUMBER");
        }
    }
    public Literal ParseLiteral(ref int lookahead, List<Token> tokens) {
        if (isLookahead<Identifier>(lookahead, tokens)) {
            var token = (Token<string>)tokens[lookahead++];
            return new ObjectLiteral(token.Value);
        }
        else if (isLookahead<Number>(lookahead, tokens)) {
            var token = (Token<string>)tokens[lookahead++];
            return new ObjectLiteral(double.Parse(token.Value));
        } 
        else {
            throw new SyntaxErrorException("Expecting one of STRING, IDENTIFIER, or NUMBER");
        }
    }
}