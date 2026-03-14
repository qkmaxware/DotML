namespace DotML.Network;

/// <summary>
/// A factory object capable of building network modules based on well-known architectures
/// </summary>
public interface INetworkModuleFactory
{
    /// <summary>
    /// Makes a module using the default settings. For more specific factory methods see the documentation individual factories.
    /// </summary>
    /// <returns>module</returns>
    public INetworkModule MakeDefault();
}

/// <summary>
/// A factory object capable of building network modules based on well-known architectures
/// </summary>
/// <typeparam name="TSettings">object type which can be used to configure the network</typeparam>
public interface INetworkModuleFactory<TSettings> : INetworkModuleFactory
    where TSettings : new()
{
    /// <summary>
    /// Makes a module using the provided settings
    /// </summary>
    /// <param name="settings">settings</param>
    /// <returns>module</returns>
    public INetworkModule Make(TSettings settings);

    INetworkModule INetworkModuleFactory.MakeDefault()
    {
        return Make(new TSettings());
    }
}