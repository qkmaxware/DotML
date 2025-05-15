using System.Numerics;
using BenchmarkDotNet.Attributes;

namespace DotML.Benchmark;

[Benchmarkable]
public class BenchmarkMatrixConvolve {

    [Params(3)]
    public int KERNEL_SIZE {get; set;}

    [Params(8, 16, 32, 64, 128, 256, 512, 1024)]
    public int DIM_SIZE {get; set;}

    private Matrix<double> kernel;
    private Matrix<double> feature;
    private Complex[] flat;
    private double[] kernel_flat;
    private double[] feature_flat;

    [GlobalSetup]
    public void Setup() {
        feature = new Matrix<double>(DIM_SIZE, DIM_SIZE);
        feature_flat = feature.ToArray();
        kernel = new Matrix<double>(KERNEL_SIZE, KERNEL_SIZE);
        kernel_flat = kernel.ToArray();
        flat = new Complex[DIM_SIZE * DIM_SIZE];
    }

    [Benchmark]
    public void NaiveConvolution() {
        feature.Convolve(kernel);
    }

    [Benchmark]
    public void CooleyTukeyFFT() {
        CooleyTukey.FFT(flat);
    }

    [Benchmark]
    public void CooleyTukeyConvolution() {
        CooleyTukey.ConvolveFFT(feature, kernel);
    }

}