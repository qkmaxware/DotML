using BenchmarkDotNet.Attributes;

namespace DotML.Benchmark;

[Benchmarkable]
public class BenchmarkMatrix {

    [ParamsSource(nameof(VectorSizeValues))]
    public int DimSize {get; set;}
    public static IEnumerable<int> VectorSizeValues() {
        for (var i = 0; i < 1000; i+=100)
            yield return i;
    }

    //[Benchmark]
    public void CreateDefault() {
        var vec = new Matrix<double>(DimSize, DimSize);
    }

    //[Benchmark]
    public void CreateWithValue() {
        var vec = new Matrix<double>(DimSize, DimSize, 12.0);
    }

    //[Benchmark]
    public void Hadamard() {
        var v0 = new Matrix<double>(DimSize, DimSize);
        var v1 = new Matrix<double>(DimSize, DimSize);

        var v2 = v0.HadamardWith(v1);
    } 

    [Benchmark]
    public void FlattenRows() {
        var matrix = new Matrix<double>(DimSize, DimSize);

        matrix.FlattenRows().ToArray();
    }

}