using BenchmarkDotNet.Attributes;

namespace DotML.Benchmark;

[Benchmarkable]
public class BenchmarkVec {

    [ParamsSource(nameof(VectorSizeValues))]
    public int VectorSize {get; set;}
    public static IEnumerable<int> VectorSizeValues() {
        for (var i = 0; i < 1000; i+=100)
            yield return i;
    }

    [Benchmark]
    public void CreateZero() {
        var vec = new Vec<double>(VectorSize);
    }

    [Benchmark]
    public void CreateFill() {
        var vec = new Vec<double>(VectorSize, 5.0);
    }

    private static Random rng = new Random();

    [Benchmark]
    public void CreateRandom() {
        var vec = new Vec<double>(VectorSize, rng.NextDouble);
    }

    [Benchmark]
    public void Hadamard() {
        var v0 = new Vec<double>(VectorSize);
        var v1 = new Vec<double>(VectorSize);

        var v2 = v0.Hadamard(v1);
    } 

    [Benchmark]
    public void Add() {
        var v0 = new Vec<double>(VectorSize);
        var v1 = new Vec<double>(VectorSize);

        var v2 = v0.AddedWith(v1);
    } 

    [Benchmark]
    public void Sub() {
        var v0 = new Vec<double>(VectorSize);
        var v1 = new Vec<double>(VectorSize);

        var v2 = v0.SubtractWith(v1);
    } 
}