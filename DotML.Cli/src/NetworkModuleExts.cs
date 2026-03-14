using DotML.Network;

public static class NetworkModuleExtensions
{
    public static string Name(this INetworkModule module)
    {
        if (module is ArchitectureBlock arch)
            return arch.Name;
        
        // TODO other cases??

        return "Network";
    }
}