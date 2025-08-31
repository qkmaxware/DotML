using BenchmarkDotNet.Attributes;

namespace DotML.Benchmark;

[Benchmarkable]
public class BenchmarkMatrixVsTensor {
    #region Arguments
    [Params(1)]
    public int Channels;

    [ParamsSource(nameof(VectorSizeValues))]
    public int DimLength {get; set;}
    private const int MinDimLength = 1000;
    private const int MaxDimLength = 1000;
    public static IEnumerable<int> VectorSizeValues() {
        for (var i = MinDimLength; i <= MaxDimLength; i+=100)
            yield return i;
    }
    #endregion

    #region Initialization
    private Matrix<float>[] matrix;
    private Matrix<float> matrix_kernel = new Matrix<float>(5, 5); // 5x5 kernel
    private Tensor<float> tensor;
    private Tensor<float> tensor_kernel = Tensor<float>.Defaults(new TensorShape(5, 5));

    [GlobalSetup]
    public void Setup() {
        matrix = new Matrix<float>[Channels];
        for(var i = 0; i < Channels; i++)
            matrix[i] = new Matrix<float>(DimLength, DimLength);
        tensor = Tensor<float>.Defaults(new TensorShape(Channels, DimLength, DimLength));
    }
    [GlobalCleanup]
    public void Cleanup() {

    }
    #endregion

    [Benchmark]
    public void MatrixAdd() {
        var result = new Matrix<float>[Channels];
        for (var i = 0; i < Channels; i++) {
            result[i] = matrix[i] + matrix[i];
        }
    }
    [Benchmark]
    public void TensorAdd() {
        var result = tensor + tensor;
    }

    [Benchmark]
    public void MatrixHadamard() {
        var result = new Matrix<float>[Channels];
        for (var i = 0; i < Channels; i++) {
            result[i] = matrix[i].HadamardWith(matrix[i]);
        }
    }
    [Benchmark]
    public void TensorHadamard() {
        var result = tensor.HadamardWith(tensor);
    } 

    [Benchmark]
    public void MatrixBatchedMultiply() {
        var result = new Matrix<float>[Channels];
        for (var i = 0; i < Channels; i++) {
            result[i] = matrix[i] * matrix[i];
        }
    }
    [Benchmark]
    public void TensorBatchedMultiply() {
        var result = tensor.BatchedMatMul(tensor);
    }

    [Benchmark]
    public void MatrixTranspose() {
        var result = new Matrix<float>[Channels];
        for (var i = 0; i < Channels; i++) {
            result[i] = matrix[i].Transpose();
        }
    }

    [Benchmark]
    public void TensorTranspose() {
        // Swap only the last 2 for this case to match MatrixTranspose
        var perm = Enumerable.Range(0, tensor.Rank).ToArray();
        var temp = perm[^1];
        perm[^1] = perm[^2];
        perm[^2] = temp;

        var result = tensor.Permute(perm);
    }

    [Benchmark]
    public void MatrixConvolve() {
        var result = new Matrix<float>[Channels];
        for (var i = 0; i < Channels; i++) {
            result[i] = matrix[i].Convolve(matrix_kernel);
        }
    }
    [Benchmark]
    public void TensorConvolve() {
        var result = tensor.Convolve2D(tensor_kernel);
    }

    [Benchmark]
    public void MatrixTransposeConvolve() {
        var result = new Matrix<float>[Channels];
        for (var i = 0; i < Channels; i++) {
            result[i] = matrix[i].TransposeConvolve(matrix_kernel);
        }
    }
    [Benchmark]
    public void TensorTransposeConvolve() {
        var result = tensor.TransposeConvolve2D(tensor_kernel);
    }

    
    /// A Convolution stress test. Current best time: 4.8s
    /*[Benchmark]
    public void TensorConvolveStress() {
        var tensor = Tensor<float>.ConstantValued(new TensorShape(1, 512, 512, 512), 1);
        var kernel = Tensor<float>.ConstantValued(new TensorShape(1, 512, 3, 3), 2);

        tensor.Convolve2D(kernel);
    }*/
}