using DotML.NetDot.Dot;
using DotML.Network;

namespace DotML.NetDot;

public class CompiledModuleNode : DotVertex
{
	public INetworkModule Module { get; }

	public CompiledModuleNode(INetworkModule compiled)
	{
		this.Module = compiled;
	}
}