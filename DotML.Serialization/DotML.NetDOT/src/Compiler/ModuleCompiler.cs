using System.Diagnostics.CodeAnalysis;
using DotML.NetDot.Dot;
using DotML.Network;
using DotML.Network.Embedding.Text;

namespace DotML.NetDot;

/// <summary>
/// DOT to Module compiler
/// </summary>
public class ModuleCompiler
{

    private List<IModulePattern> matchers = new List<IModulePattern> {
        // Default pattern matchers
        // Simple layer matchers
		new LayerPattern<DenseLinear>(),
        new LayerPattern<Conv2D>(),
        new LayerPattern<TransposeConv2D>(),
        new LayerPattern<Flatten>(),
        new LayerPattern<Activation>(),
        new LayerPattern<MaxPool2D>(),
        new LayerPattern<MinPool2D>(),
        new LayerPattern<AvgPool2D>(),
        new LayerPattern<Dropout>(),
        new LayerPattern<SoftmaxOutput>(),
        new LayerPattern<SelfAttention>(),
        new LayerPattern<LearnedEmbedding>(),

        // Block matchers
        new SequentialBlockPattern(),
        new ResidualPattern()
    };

    public void Register(IModulePattern pattern) => matchers.Add(pattern);

    private Dot.Parser parser = new Parser();

    public INetworkModule Compile(string str)
    {
        return Compile(parser.Parse(str));
    }

    public INetworkModule Compile(DotGraph graph)
    {
        bool matched;
        do
        {
            matched = false;
            foreach (var matcher in matchers)
            {
                if (matcher.TryReplacePattern(graph))
                {
                    matched = true;
                    break; // Try all from beginning again
                }
            }
        } while (matched);

        if (!isCompiled(graph, out var compiled, out var error))
        {
            throw new InvalidOperationException("Failed to fully compile the graph", error);
        }

        return compiled;
    }

    private bool isCompiled(DotGraph graph, [NotNullWhen(true)] out INetworkModule? compiled, [NotNullWhen(false)] out Exception? error)
    {
        compiled = null;
        error = null;

        // If we contain anything except inputs, outputs, and compiled modules we failed
        if (graph.EnumerateVertices().Any(x => x is not CompiledModuleNode && !x.IsInput() && !x.IsOutput()))
        {
            error = new Exception("Graph contains nodes that were unable to be compiled");
            return false;
        }

        // Get the compiled node
        var compNode = graph.EnumerateVertices().Where(x => x is CompiledModuleNode).Cast<CompiledModuleNode>().FirstOrDefault();
        if (compNode is null)
        {
            error = new Exception("Something has happened during compilation");
            return false;
        }
        compiled = compNode.Module;

        // Ensure all inputs and outputs stem from this node
        if (graph.EnumerateVertices().Where(x => x.IsInput()).Any(x => !graph.HasEdge(x, compNode)))
        {
            error = new Exception("Graphs inputs are mismatched with compiled module");
            return false;
        }
        if (graph.EnumerateVertices().Where(x => x.IsOutput()).Any(x => !graph.HasEdge(compNode, x)))
        {
            error = new Exception("Graphs outputs are mismatched with compiled module");
            return false;
        }

        return true;
    }

}