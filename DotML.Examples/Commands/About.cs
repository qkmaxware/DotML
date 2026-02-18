using System.Reflection;
using CommandLine;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;

namespace DotML.Examples;

[Verb("about", HelpText = "List information about this library and pc")]
public class About : Command
{

    public override int TryExec()
    {
        try
        {
            Exec(null);
            return 0;
        } catch
        {
            return 1;
        }
    }
    public override void Exec(IExample? _)
    {
        const string unknown = "unknown";
        Console.WriteLine("OS:");
        Console.WriteLine("    Name: " + (Environment.OSVersion.Platform));
        Console.WriteLine("    Version: " + (Environment.OSVersion.VersionString));
        Console.WriteLine();

        Console.WriteLine("Processor:");
        Console.WriteLine("    Name: " + (Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? unknown));
        Console.WriteLine("    Arch: " + (Environment.GetEnvironmentVariable("PROCESSOR_ARCHITECTURE") ?? unknown));
        Console.WriteLine("    Level: " + (Environment.GetEnvironmentVariable("PROCESSOR_LEVEL") ?? unknown));
        Console.WriteLine("    Revision: " + (Environment.GetEnvironmentVariable("PROCESSOR_REVISION") ?? unknown));
        Console.WriteLine("    Cores: " + (Environment.ProcessorCount));
        Console.WriteLine();

        Console.WriteLine("DotML:");
        Console.WriteLine("    Version: " + (typeof(INetworkModule).Assembly.GetName().Version));
        Console.WriteLine("    Config: " + (typeof(INetworkModule).Assembly.GetCustomAttribute<AssemblyConfigurationAttribute>()?.Configuration ?? unknown));
        Console.WriteLine();
    }

    protected IEnumerable<string> GetExamples()
    {
        var types = Assembly
            .GetExecutingAssembly()
            .GetExportedTypes()
            .Where(type => type.IsClass && !type.IsAbstract && type.IsAssignableTo(typeof(IExample)))
            .Select(type => type.Name);
        return types;
    }
}