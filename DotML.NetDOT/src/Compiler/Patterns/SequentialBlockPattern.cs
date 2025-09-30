using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Runtime.InteropServices;
using DotML.NetDot.Dot;
using DotML.Network;

namespace DotML.NetDot;

// Pattern for converting a list of compiled modules into a single sequential module
public class SequentialBlockPattern : IModulePattern
{
    public bool TryReplacePattern(DotGraph graph)
    {
        for (var nodeIndex = 0; nodeIndex < graph.VertexCount; nodeIndex++)
        {
            var node = graph.GetVertex(nodeIndex);
            if (isChain(graph, node, out var chain))
            {
                graph.ReplaceAll(
                    replacement: new CompiledModuleNode(new SequentialBlock(chain.Where(x => x is CompiledModuleNode).Cast<CompiledModuleNode>().Select(x => x.Module))), 
                    originals: CollectionsMarshal.AsSpan(chain)
                );
                return true;
            }
        }

        return false;
    }

    private bool isChain(DotGraph graph, DotVertex startAt, [NotNullWhen(true)] out List<DotVertex>? chain)
    {
        var firstNode = startAt;

        var current = firstNode;
        var incoming = graph.IncomingEdges(current).ToList();
        var outgoing = graph.OutgoingEdges(current).ToList();
        // Checks:
        // 1. Node can't be an input or output node (these don't have a concept in the module system, inputs are just whatever you pass to the forward method)
        // 2. Sequence start node has exactly 1 output edge
        // 3. Sequence start node has to already be a compiled module
        if (current.IsInput() || current.IsOutput() || outgoing.Count != 1 || current is not CompiledModuleNode first)
        {
            chain = null;
            return false;
        }

        chain = new List<DotVertex>();
        chain.Add(first);
        current = outgoing[0].To;

        while (current is not null)
        {
            incoming = incoming = graph.IncomingEdges(current).ToList();
            outgoing = outgoing = graph.OutgoingEdges(current).ToList();
            // Checks:
            // 1. Node can't be an input or output node (these don't have a concept in the module system, inputs are just whatever you pass to the forward method)
            // 2. Sequence middle node has exactly 1 input edge.
            // 3. Sequence middle node has to already be a compiled module
            if (current.IsInput() || current.IsOutput() || incoming.Count != 1 || current is not CompiledModuleNode)
            {
                break;
            }

            chain.Add((CompiledModuleNode)current);
            current = outgoing[0].To;
        }

        return chain.Count >= 2;
    }
}
