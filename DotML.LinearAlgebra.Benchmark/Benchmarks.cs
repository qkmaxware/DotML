using BenchmarkDotNet.Running;

namespace DotML.Benchmark;

public class BenchmarkableAttribute : System.Attribute { }

public class Benchmarks {

    public static void Main(string[] args) {
        var assembly = typeof(Benchmarks).Assembly;
        var runables = assembly.GetExportedTypes().Where(x => x.IsDefined(typeof(BenchmarkableAttribute), false)).ToArray();

        BenchmarkSwitcher
            .FromTypes(runables)
            .Run(args);
    }
}