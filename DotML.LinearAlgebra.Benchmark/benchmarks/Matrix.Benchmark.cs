using BenchmarkDotNet.Attributes;

namespace DotML.Benchmark;

[Benchmarkable]
public class BenchmarkMatrix {

    [ParamsSource(nameof(VectorSizeValues))]
    public int DimSize {get; set;}
    public static IEnumerable<int> VectorSizeValues() {
        for (var i = 0; i < 1000; i+=100)
            yield return Math.Max(10, i);
    }

    private Matrix<double> v0;
    private Matrix<double> v1;

    private Matrix<float> v0f;
    private Matrix<float> v1f;

    [GlobalSetup]
    public void Setup() {
        v0 = new Matrix<double>(DimSize, DimSize);
        v1 = new Matrix<double>(DimSize, DimSize);
        v0f = new Matrix<float>(DimSize, DimSize);
        v1f = new Matrix<float>(DimSize, DimSize);
    }

    //[Benchmark]
    public void CreateDefault() {
        var vec = new Matrix<double>(DimSize, DimSize);
    }

    //[Benchmark]
    public void CreateWithValue() {
        var vec = new Matrix<double>(DimSize, DimSize, 12.0);
    }

    [Benchmark]
    public void Add() {
        var v2 = v0.AddWith(v1);
    } 
    [Benchmark]
    public void AddFloat() {
        var v2f = v0f.AddWith(v1f);
    } 

    //[Benchmark]
    public void Hadamard() {
        var v2 = v0.HadamardWith(v1);
    } 

    //[Benchmark]
    public void FlattenRows() {
        var matrix = new Matrix<double>(DimSize, DimSize);

        matrix.FlattenRows().ToArray();
    }

}