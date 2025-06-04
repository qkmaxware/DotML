using System.Numerics;
using BenchmarkDotNet.Attributes;

namespace DotML.Benchmark;

[Benchmarkable]
public class BenchmarkMatrixConvolve {

    [Params(3)]
    public int KERNEL_SIZE {get; set;}

    [Params(32, 128, 512)]
    public int DIM_SIZE {get; set;}

    private Matrix<double> kernel;
    private Matrix<double> feature;
    private double[] kernel_flat;
    private double[] feature_flat;

    [GlobalSetup]
    public void Setup() {
        feature = new Matrix<double>(DIM_SIZE, DIM_SIZE);
        feature_flat = feature.ToArray();
        kernel = new Matrix<double>(KERNEL_SIZE, KERNEL_SIZE);
        kernel_flat = kernel.ToArray();
    }

    [Benchmark]
    public void Naive() {
        feature.Convolve(kernel);
    }

    [Benchmark]
    public void FFT() {
        FftConvolve.Convolve2D(feature, kernel);
    }

}