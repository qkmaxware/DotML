using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Runtime.InteropServices;
using DotML.NetDot.Dot;
using DotML.Network;

namespace DotML.NetDot;

// Pattern for handling residual connections 
public class ResidualPattern : IModulePattern {

    public bool TryReplacePattern(DotGraph graph)
    {
        for (var nodeIndex = 0; nodeIndex < graph.VertexCount; nodeIndex++)
        {
            var node = graph.GetVertex(nodeIndex);
            if (!IsResidualEndpoint(node))
                continue;

            var incoming = graph.IncomingEdges(node).ToList();
            if (incoming.Count != 2)
                continue;

            var input1 = incoming[0].From;
            var input2 = incoming[1].From;

            if (TryMatchResidual(graph, input1, input2, out var residualBlock, out var involvedNodes))
            {
                involvedNodes.Append(node);
                graph.ReplaceAll(new CompiledModuleNode(residualBlock), CollectionsMarshal.AsSpan(involvedNodes));
                return true;
            }

            // Try flipped order
            if (TryMatchResidual(graph, input2, input1, out residualBlock, out involvedNodes))
            {
                involvedNodes.Append(node);
                graph.ReplaceAll(new CompiledModuleNode(residualBlock), CollectionsMarshal.AsSpan(involvedNodes));
                return true;
            }
        }
        
        return false;
	}
	
	private static bool IsResidualEndpoint(DotVertex node)
    {
        return node.Attributes.TryGetValue("type", out var val) &&
               val.Equals("residual", StringComparison.OrdinalIgnoreCase);
    }
	
	private static bool TryMatchResidual(
        DotGraph graph,
        DotVertex shortcutNode,
        DotVertex mainNode,
        out INetworkModule block,
        out List<DotVertex> involvedNodes)
    {
        block = null!;
        involvedNodes = new();

        if (mainNode is not CompiledModuleNode compiledMain)
            return false;
		
		var mainIncoming = graph.IncomingEdges(mainNode).ToList();
		
		// Case:    Shortcut is the direct input of main (typical ResNet)
        // Pattern: A -> Module -> Residual, A -> Residual
        if (mainIncoming.Count == 1 && mainIncoming[0].From == shortcutNode)
        {
            // Treat shortcut as identity connection (null means identity ie no transformation of the input)
            block = BuildResidualBlock(null, compiledMain.Module, mainNode);
            involvedNodes = new() { mainNode }; // Don't include shortcutNode, it's not compiled
            return true;
        }
		
		// Case:    Shortcut and main share a single common parent (ResNet with transformation on X like 1x1 conv)
		// Pattern: A -> Module -> Residual, A -> Shortcut -> Residual
		var shortCutIncoming = graph.IncomingEdges(shortcutNode).ToList();
        if (shortCutIncoming.Count == 1 && shortcutNode is CompiledModuleNode compiledShortcut && mainIncoming[0].From == shortCutIncoming[0].From)
        {
            block = BuildResidualBlock(compiledShortcut.Module, compiledMain.Module, mainNode);
            involvedNodes = new() { shortcutNode, mainNode };
            return true;
        }
		
        return false;
    }
	
	private static INetworkModule BuildResidualBlock(INetworkModule? shortcut, INetworkModule main, DotVertex node)
    {
        var type = "add";
        if (node.Attributes.TryGetValue("mode", out var val))
            type = val;

        return (type) switch
        {
            "add" => new ResidualAdd(main, shortcut),
            "concat" => new ResidualConcat(main, shortcut),
            _ => throw new NotSupportedException($"Residual mode '{type}' is not supported for deserialization"),
        };
    }
}