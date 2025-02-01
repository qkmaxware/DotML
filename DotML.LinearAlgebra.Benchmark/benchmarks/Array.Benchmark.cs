using BenchmarkDotNet.Attributes;

namespace DotML.Benchmark;

[Benchmarkable]
public class BenchmarkArray {

    [Params(10, 100, 1000, 10_000)]
    public int DimensionLength {get; set;}

    [Benchmark]
    public void OneDimensionalAsOneD() {
        var array = new double[DimensionLength*DimensionLength];
        // Use as if 2D
        for (var i = 0; i < array.Length; i++)
            array[i] = 10;
    }

    [Benchmark]
    public void OneDimensionalAs2D() {
        var array = new double[DimensionLength*DimensionLength];
        // Use as if 2D
        for (var row = 0; row < DimensionLength; row++) {
            for (var col = 0; col < DimensionLength; col++) {
                array[row * DimensionLength + col] = 10;
            }
        }
    }

    [Benchmark]
    public void TwoDimensional() {
        var array = new double[DimensionLength, DimensionLength];
        for (var row = 0; row < DimensionLength; row++) {
            for (var col = 0; col < DimensionLength; col++) {
                array[row, col] = 10;
            }
        }
    }

}