using System.Diagnostics.CodeAnalysis;
using Qkmaxware.Parsing;

namespace DotML.NetDot.Dot;

public class Parser
{
    // Grammar
    // See: https://graphviz.org/doc/info/lang.html
    /*
    graph	:	[ strict ] (graph | digraph) [ ID ] '{' stmt_list '}'
    stmt_list	:	[ stmt [ ';' ] stmt_list ]
    stmt	:	node_stmt
    |	edge_stmt
    |	attr_stmt
    |	ID '=' ID
    |	subgraph
    attr_stmt	:	(graph | node | edge) attr_list
    attr_list	:	'[' [ a_list ] ']' [ attr_list ]
    a_list	:	ID '=' ID [ (';' | ',') ] [ a_list ]
    edge_stmt	:	(node_id | subgraph) edgeRHS [ attr_list ]
    edgeRHS	:	edgeop (node_id | subgraph) [ edgeRHS ]
    node_stmt	:	node_id [ attr_list ]
    node_id	:	ID [ port ]
    port	:	':' ID [ ':' compass_pt ]
    |	':' compass_pt
    subgraph	:	[ subgraph [ ID ] ] '{' stmt_list '}'
    compass_pt	:	n | ne | e | se | s | sw | w | nw | c | _
    */

    private Parser<DotGraph> graph_file;

    public Parser()
    {
        var compass_pt = Text.Is("ne").Or(Text.Is("nw")).Or(Text.Is("se")).Or(Text.Is("sw")).Or(Text.Is("n")).Or(Text.Is("e")).Or(Text.Is("s")).Or(Text.Is("w")).Or(Text.Is("c")).Or(Text.Is("_")).Map(
            (dir) => dir switch
            {
                "ne" => Compass.NE,
                "nw" => Compass.NW,
                "se" => Compass.SE,
                "sw" => Compass.SW,
                "n" => Compass.N,
                "s" => Compass.S,
                "e" => Compass.E,
                "w" => Compass.W,
                "c" => Compass.C,
                _ => Compass.Unknown
            }
        );

        var numeral =
            Character.Is('-').Optional()
            .ThenMap(
                Character.Is('.').ThenMap(Text.FromCharacters(Character.Range('0', '9').OneOrMore()), (dot, rest) => dot + rest)
                .Or(
                    Text.FromCharacters(Character.Range('0', '9').OneOrMore()).ThenMap(Text.FromCharacters(Character.Is('.').ThenMap(Character.Range('0', '9').ZeroOrMore(), (dot, decimals) => { decimals.Insert(0, dot); return decimals; })).Optional(), (whole, fractional) =>
                    {
                        if (fractional.TryGetValue(out string? str))
                        {
                            return whole + str;
                        }
                        else
                        {
                            return whole;
                        }
                    })
                ),
                (negage, content) =>
                {
                    if (negage.TryGetValue(out char minus))
                    {
                        return minus + content;
                    }
                    else
                    {
                        return content;
                    }
                }
            );

        var id = Literal.Identifier().Or(numeral).Or(Text.DoubleQuoted()); // Or HTML text... todo?

        // Keywords
        var k_strict = Text.Is("strict");
        var k_graph = Text.Is("graph");
        var k_digraph = Text.Is("digraph");
        var k_subgraph = Text.Is("subgraph");
        var k_node = Text.Is("node");
        var k_edge = Text.Is("edge");

        // Symbols
        var open_brace = Character.Is('{');
        var close_brace = Character.Is('}');
        var open_attr = Character.Is('[');
        var close_attr = Character.Is(']');
        var semi = Character.Is(';');
        var comma = Character.Is(',');
        var commaOrSemi = semi.Or(comma);
        var colon = Character.Is(':');
        var equals = Character.Is('=');

        var attr = id.ThenMap(equals, (id, eq) => id).ThenMap(id, (key, value) => new KeyValuePair<string, string>(key, value));
        var a_list = attr.ZeroOrMoreSeparatedBy(commaOrSemi).Between(open_attr, close_attr);
        var attr_list = a_list.OneOrMore();

        var port = colon.Then(id).ThenMap(colon.Then(compass_pt).Optional(), (id, compass_pt) => id + compass_pt.Match((some) => ":" + some.Value, (none) => string.Empty))
            .Or(colon.Then(compass_pt).Map(compass => ":" + compass));
        var node_id = id.ThenMap(port.Optional(), (id, port) => id + port.Match((some) => some.Value, (none) => string.Empty));

        var node_stmt = node_id.ThenMap<string, Maybe<List<List<KeyValuePair<string, string>>>>, Action<DotGraph>>(attr_list.Optional(), (id, attr) =>
        {
            return (DotGraph g) =>
            {
                // Fetch or create vertex
                var vertex = g.GetOrAdd(id, (i) => new DotVertex(i));

                // Fill attributes
                if (attr.TryGetValue(out var values))
                {
                    if (values is not null)
                    {
                        foreach (var valueList in values)
                        {
                            foreach (var a in valueList)
                            {
                                vertex.Attributes[a.Key] = a.Value;
                            }
                        }
                    }
                }
            };
        });

        var edge_op = Text.Is("--").Or(Text.Is("->")).Map((str) => str == "--" ? DotGraphMode.Undirected : DotGraphMode.Directed);
        var id_list = node_id.Map(id => new List<string> { id }); // TODO .OrSubgraph
        var edge_stmt = id_list
            .ThenMap(edge_op, (from, opts) => (from, opts))
            .ThenMap(id_list, (a, to) => (a.from, a.opts, to))
            .ThenMap<(List<string> from, DotGraphMode opts, List<string> to), Maybe<List<List<KeyValuePair<string, string>>>>, Action<DotGraph>>(attr_list.Optional(), (x, attr) =>
            {
                return (DotGraph g) =>
                {
                    // Loop over all combinations of from and to
                    foreach (var pair in x.from.SelectMany(f => x.to, (f, t) => new { First = f, Second = t }))
                    {
                        // Fetch or create vertices
                        var lhs = g.GetOrAdd(pair.First, (id) => new DotVertex(id));
                        var rhs = g.GetOrAdd(pair.Second, (id) => new DotVertex(id));

                        // Fetch or create edge
                        var edge = g.GetOrConnect(lhs, rhs, static (edge) => { });
                        if (edge.Data is null)
                            edge.Data = new DotEdge();

                        // Fill attributes
                        if (attr.TryGetValue(out var values))
                        {
                            if (values is not null)
                            {
                                foreach (var valueList in values)
                                {
                                    foreach (var at in valueList)
                                    {
                                        edge.Data.Attributes[at.Key] = at.Value;
                                    }
                                }
                            }
                        }

                        // If undirected, do the inverse connection as well by repeating the above with lhs and rhs swapped
                        if (x.opts == DotGraphMode.Undirected)
                        {
                            (rhs, lhs) = (lhs, rhs);

                            // Fetch or create edge
                            edge = g.GetOrConnect(lhs, rhs, static (edge) => { });
                            if (edge.Data is null)
                                edge.Data = new DotEdge();

                            // Fill attributes
                            if (attr.TryGetValue(out values))
                            {
                                if (values is not null)
                                {
                                    foreach (var valueList in values)
                                    {
                                        foreach (var at in valueList)
                                        {
                                            edge.Data.Attributes[at.Key] = at.Value;
                                        }
                                    }
                                }
                            }
                        }
                    }
                };
            });

        var attr_stmt = k_graph.Or(k_node).Or(k_edge).ThenMap<string, List<List<KeyValuePair<string, string>>>, Action<DotGraph>>(attr_list, (type, attrs) =>
        {
            return (DotGraph g) =>
            {
                switch (type)
                {
                    case "graph":
                        foreach (var valueList in attrs)
                        {
                            foreach (var a in valueList)
                            {
                                g.Attributes[a.Key] = a.Value;
                            }
                        }
                        break;
                    case "node":
                        foreach (var node in g.EnumerateVertices())
                        {
                            foreach (var valueList in attrs)
                            {
                                foreach (var a in valueList)
                                {
                                    node.Attributes[a.Key] = a.Value;
                                }
                            }
                        }
                        break;
                    case "edge":
                        foreach (var edge in g.EnumerateEdges())
                        {
                            if (edge.Data is null)
                                edge.Data = new DotEdge();
                            foreach (var valueList in attrs)
                            {
                                foreach (var a in valueList)
                                {
                                    edge.Data.Attributes[a.Key] = a.Value;
                                }
                            }
                        }
                        break;
                }
            };
        });

        var graph_attr = attr.Map<KeyValuePair<string, string>, Action<DotGraph>>((kv) =>
        {
            return (DotGraph g) =>
            {
                g.Attributes[kv.Key] = kv.Value;
            };
        });

        var stmt = node_stmt.Or(edge_stmt).Or(attr_stmt).Or(graph_attr); // .Or(subgraph_stmt);

        var stmt_block = stmt.ZeroOrMore().Between(open_brace, close_brace);

        Parser<DotGraph> graph_file = k_strict.Optional()
            .Then(k_graph.Or(k_digraph).Map(mode => mode == "graph" ? DotGraphMode.Undirected : DotGraphMode.Directed))
            .ThenMap(id.Optional(), (mode, id) => new DotGraph { Id = id.Match((some) => some.Value, (none) => string.Empty), Mode = mode })
            .ThenMap(stmt_block, (graph, stmts) =>
            {
                // Each statement modifies the graph
                foreach (var stmt in stmts)
                {
                    stmt(graph);
                }

                // Return the graph after modifications
                return graph;
            });
        this.graph_file = graph_file;
    }

    public DotGraph Parse(string text)
    {
        var result = this.graph_file(new InputString(text));
        if (!result.TryGetValue(out DotGraph? graph))
            throw result.Error ?? new Exception("Unknown parsing error");
        return graph;
    }

    public bool TryParse(string text, [NotNullWhen(true)]out DotGraph? graph)
    {
        var result = this.graph_file(new InputString(text));
        if (!result.TryGetValue(out graph))
            return false;
        return true;
    }

}