namespace DotML.Test;

[TestClass]
public class TestTensor
{
    [TestMethod]
    public void TestCreation()
    {
        Tensor<double> matrix = Tensor<double>.Zeros(new TensorShape(5, 2));
        Assert.AreEqual(5, matrix.Shape.Length(0));
        Assert.AreEqual(2, matrix.Shape.Length(1));

        Tensor<double> m2 = Tensor<double>.FromRectangularArray(new double[,]{
            {1, 2, 3},
            {4, 5, 6}
        });
        Assert.AreEqual(2, m2.Shape.Length(0));
        Assert.AreEqual(3, m2.Shape.Length(1));
        Assert.AreEqual(1, m2[0, 0]);
        Assert.AreEqual(3, m2[0, 2]);
        Assert.AreEqual(4, m2[1, 0]);
        Assert.AreEqual(6, m2[1, 2]);

        var zero = Tensor<double>.Zeros(new TensorShape(3, 3));
        Assert.AreEqual(3, zero.Shape.Length(0));
        Assert.AreEqual(3, zero.Shape.Length(1));
        foreach (var element in zero.AsSpan())
            Assert.AreEqual(0, element);

        var ones = Tensor<double>.Ones(new TensorShape(4, 4));
        Assert.AreEqual(4, ones.Shape.Length(0));
        Assert.AreEqual(4, ones.Shape.Length(1));
        foreach (var element in ones.AsSpan())
            Assert.AreEqual(1, element);
    }

    [TestMethod]
    public void TestTranspose()
    {
        Tensor<double> m2 = Tensor<double>.FromRectangularArray(new double[,]{
            {1, 2, 3},
            {4, 5, 6}
        });
        Assert.AreEqual(2, m2.Shape.Length(0));
        Assert.AreEqual(3, m2.Shape.Length(1));
        Assert.AreEqual(1, m2[0, 0]);
        Assert.AreEqual(3, m2[0, 2]);
        Assert.AreEqual(4, m2[1, 0]);
        Assert.AreEqual(6, m2[1, 2]);
        var m2shape = m2.Shape;
        Assert.AreEqual(2, m2shape.Length(0));
        Assert.AreEqual(3, m2shape.Length(1));

        {
            var transposed = m2.Transpose();
            Assert.AreEqual(3, transposed.Shape.Length(0));
            Assert.AreEqual(2, transposed.Shape.Length(1));
            Assert.AreEqual(1, transposed[0, 0]);
            Assert.AreEqual(4, transposed[0, 1]);
            Assert.AreEqual(3, transposed[2, 0]);
            Assert.AreEqual(6, transposed[2, 1]);
            var tshape = transposed.Shape;
            Assert.AreEqual(3, tshape.Length(0));
            Assert.AreEqual(2, tshape.Length(1));
        }

        {
            var transposed = m2.MatrixTranspose();
            Assert.AreEqual(3, transposed.Shape.Length(0));
            Assert.AreEqual(2, transposed.Shape.Length(1));
            Assert.AreEqual(1, transposed[0, 0]);
            Assert.AreEqual(4, transposed[0, 1]);
            Assert.AreEqual(3, transposed[2, 0]);
            Assert.AreEqual(6, transposed[2, 1]);
            var tshape = transposed.Shape;
            Assert.AreEqual(3, tshape.Length(0));
            Assert.AreEqual(2, tshape.Length(1));
        }

        {
            var transposed = m2.Transpose(^2, ^1);
            Assert.AreEqual(3, transposed.Shape.Length(0));
            Assert.AreEqual(2, transposed.Shape.Length(1));
            Assert.AreEqual(1, transposed[0, 0]);
            Assert.AreEqual(4, transposed[0, 1]);
            Assert.AreEqual(3, transposed[2, 0]);
            Assert.AreEqual(6, transposed[2, 1]);
            var tshape = transposed.Shape;
            Assert.AreEqual(3, tshape.Length(0));
            Assert.AreEqual(2, tshape.Length(1));
        }
    }

    [TestMethod]
    public void TestElementWise()
    {
        {
            Tensor<double> A = Tensor<double>.FromRectangularArray(new double[,] {
                { 1, 2 },
                { 3, 4 }
            });

            var B = A.ElementWise((x) => x * x);
            Assert.AreEqual(1, B[0, 0]);
            Assert.AreEqual(2 * 2, B[0, 1]);
            Assert.AreEqual(3 * 3, B[1, 0]);
            Assert.AreEqual(4 * 4, B[1, 1]);

            var C = A.ElementWise<float>((x) => (float)(x * x));
        }

        {
            Tensor<double> A = Tensor<double>.FromRectangularArray(new double[,] {
                { 1, 2 },
                { 3, 4 }
            });
            Tensor<double> B = Tensor<double>.FromRectangularArray(new double[,] {
                { 5, 6 },
                { 7, 8 }
            });

            var C = A.ElementWiseBinary(B, (a, b) => a + b);
            Assert.AreEqual(1 + 5, C[0, 0]);
            Assert.AreEqual(2 + 6, C[0, 1]);
            Assert.AreEqual(3 + 7, C[1, 0]);
            Assert.AreEqual(4 + 8, C[1, 1]);
        }
    }

    [TestMethod]
    public void TestReshape()
    {
        Tensor<double> A = Tensor<double>.FromRectangularArray(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });
        var row = A.Reshape(new TensorShape(1, 4));
        var col = A.Reshape(new TensorShape(4, 1));

        //Assert.AreEqual(1, row.Length);
        Assert.AreEqual(1, row.Shape.Length(0));
        Assert.AreEqual(4, row.Shape.Length(1));

        //Assert.AreEqual(1, col.Length);
        Assert.AreEqual(4, col.Shape.Length(0));
        Assert.AreEqual(1, col.Shape.Length(1));
    }

    [TestMethod]
    public void TestHadamard()
    {
        Tensor<double> A = Tensor<double>.FromRectangularArray(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });

        var c = A.HadamardWith(A);
        Assert.AreEqual(2, c.Shape.Length(0));
        Assert.AreEqual(2, c.Shape.Length(1));
        Assert.AreEqual(1 * 1, c[0, 0]);
        Assert.AreEqual(2 * 2, c[0, 1]);
        Assert.AreEqual(3 * 3, c[1, 0]);
        Assert.AreEqual(4 * 4, c[1, 1]);
    }

    [TestMethod]
    public void TestAdd()
    {
        Tensor<double> A = Tensor<double>.FromRectangularArray(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });

        var c = A.AddWith(A);
        Assert.AreEqual(2, c.Shape.Length(0));
        Assert.AreEqual(2, c.Shape.Length(1));
        Assert.AreEqual(1 + 1, c[0, 0]);
        Assert.AreEqual(2 + 2, c[0, 1]);
        Assert.AreEqual(3 + 3, c[1, 0]);
        Assert.AreEqual(4 + 4, c[1, 1]);
    }

    [TestMethod]
    public void TestAbs()
    {
        Tensor<double> A = Tensor<double>.FromRectangularArray(new double[,] {
            { -1, 2 },
            { 3, -4 }
        });

        var c = A.Abs();
        Assert.AreEqual(2, c.Shape.Length(0));
        Assert.AreEqual(2, c.Shape.Length(1));
        Assert.AreEqual(Math.Abs(-1), c[0, 0]);
        Assert.AreEqual(Math.Abs(2), c[0, 1]);
        Assert.AreEqual(Math.Abs(3), c[1, 0]);
        Assert.AreEqual(Math.Abs(-4), c[1, 1]);
    }

    [TestMethod]
    public void TestSum()
    {
        var data = new float[]
        {
            1, 2, 3, 4,     // N=0, C=0
            5, 6, 7, 8,     // N=0, C=1
            9,10,11,12,     // N=1, C=0
            13,14,15,16     // N=1, C=1
        };
        Tensor<float> tensor = Tensor<float>.FromFlattenedArray(new TensorShape(2, 2, 2, 2), data);

        var totalA = tensor.Sum();                  // Should be [136]
        var totalB = tensor.Sum([1, 2, 3], keepdim: false);     // Should be [36, 100] (same as Sum() I believe)

        Assert.AreEqual(136, totalA);
        Assert.AreEqual(1, totalB.Rank);
        Assert.AreEqual(2, totalB.ElementCount);
        Assert.AreEqual(36, totalB[0]);
        Assert.AreEqual(100, totalB[1]);

        var SubSum = tensor.Sum([0, 2, 3], keepdim: false); // Should return [36, 100]
        Assert.AreEqual(1, totalB.Rank);
        Assert.AreEqual(2, totalB.ElementCount);
        Assert.AreEqual(52, SubSum[0]);
        Assert.AreEqual(84, SubSum[1]);
    }

    [TestMethod]
    public void TestMatMulCompatible()
    {
        // Arrange: Define two matrices to multiply
        Tensor<double> A = Tensor<double>.FromRectangularArray(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });

        Tensor<double> B = Tensor<double>.FromRectangularArray(new double[,] {
            { 5, 6 },
            { 7, 8 }
        });

        // Expected result of A * B
        Tensor<double> expected = Tensor<double>.FromRectangularArray(new double[,] {
            { 19, 22 },
            { 43, 50 }
        });

        // Act: Multiply matrices A and B (use both methods)
        {
            var result = A.MatMul(B);

            // Assert: Verify the result matches the expected output
            Assert.AreEqual(expected.Shape.Length(0), result.Shape.Length(0));
            Assert.AreEqual(expected.Shape.Length(1), result.Shape.Length(1));

            var enumerable = expected.AsSpan().ToArray().Zip(result.AsSpan().ToArray());
            foreach (var pair in enumerable)
            {
                Assert.AreEqual(pair.First, pair.Second, 0.01);
            }
        }
        {
            var result = A.BatchedMatMul(B);

            // Assert: Verify the result matches the expected output
            Assert.AreEqual(expected.Shape.Length(0), result.Shape.Length(0));
            Assert.AreEqual(expected.Shape.Length(1), result.Shape.Length(1));

            var enumerable = expected.AsSpan().ToArray().Zip(result.AsSpan().ToArray());
            foreach (var pair in enumerable)
            {
                Assert.AreEqual(pair.First, pair.Second, 0.01);
            }
        }
    }

    // TODO 
    public void TestMatMulBatchedCompatible()
    {
        /*
        1. 2D × 2D matrix multiplication (no batch). DONE ^^

        2. [1, 3, 4] × [4, 5]: broadcast second tensor to match first.

        3. [2, 1, 4, 6] × [3, 6, 7]: both require broadcasting.

        4. Zero-size tensors, e.g., [0, 3, 4] × [4, 5] should produce [0, 3, 5].

        5. Mismatched shapes that can’t broadcast (e.g. [2,3,4] × [5,4,6]) — should throw.

        6. Same shape, already batched, i.e. [B, M, K] × [B, K, N].
        
        7: [1, 4, 3], B: [10, 3, 2] → Should broadcast A over 10 batches
        */
    }

    [TestMethod]
    public void TestMatMulIncompatible()
    {
        // Arrange: Define two incompatible matrices
        Tensor<double> A = Tensor<double>.FromRectangularArray(new double[,] {
            { 1, 2, 3 }
        }); // 1x3 matrix

        Tensor<double> B = Tensor<double>.FromRectangularArray(new double[,] {
            { 4, 5 },
            { 6, 7 },
            { 8, 9 }
        }); // 3x2 matrix

        // Act & Assert: Try to multiply the matrices (this should throw an exception)
        {
            Assert.ThrowsException<InvalidOperationException>(() => B.MatMul(A));
        }
        {
            Assert.ThrowsException<InvalidOperationException>(() => B.BatchedMatMul(A));
        }
    }

    [TestMethod]
    public void TestMatMulVectorCompatible()
    {
        // Arrange: Define two matrices to multiply
        Tensor<double> A = Tensor<double>.FromRectangularArray(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });

        Tensor<double> B = Tensor<double>.Vec(new double[]{
            5,
            6
        });

        // Expected result of A * B
        Tensor<double> expected = Tensor<double>.Vec(new double[] {
            17,
            39
        });

        // Act: Multiply matrices A and B
        var result = A.MatMulVector(B.AsSpan());
        Assert.AreEqual(true, result.Equals(expected));

        result = A.MatMulEachVector(^1, B);
        Assert.AreEqual(true, result.Equals(expected));

        B = B.ReshapeShared(new TensorShape(2, 1)); // Convert to column vector
        result = A.MatMulEachVector(^2, B);
        Assert.AreEqual(true, result.Equals(expected));
    }

    [TestMethod]
    public void TestMatMulBatchedVectorCompatible()
    {
        // Arrange: Define two matrices to multiply
        Tensor<double> A = Tensor<double>.FromRectangularArray(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });

        Tensor<double> B = Tensor<double>.FromRectangularArray(new double[,]{
            { 5, 6 },
            { 8, 3 }
        });

        // Expected result of A * B
        Tensor<double> expected = Tensor<double>.FromRectangularArray(new double[,]{
            { 17, 39 },
            { 14, 36 }
        });

        // Act: Multiply matrices A and B
        var result = A.MatMulEachVector(^1, B);
        Assert.AreEqual(true, result.Equals(expected));
    }

    [TestMethod]
    public void TestConvolve2D_1()
    {
        var kernels = Tensor<double>.FromRectangularArray(new double[,] {
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1}
            });
        var input = Tensor<double>.FromRectangularArray(new double[,]{
            {1, 1, 1, 0, 0},
            {0, 1, 1, 1, 0},
            {0, 0, 1, 1, 1},
            {0, 0, 1, 1, 0},
            {0, 1, 1, 0, 0},
        });
        var output = input.Convolve2D(kernels);

        var result = Tensor<double>.FromRectangularArray(new double[,] {
            {4, 3, 4},
            {2, 4, 3},
            {2, 3, 4}
        });

        Assert.AreEqual(result.Shape.Length(^2), output.Shape.Length(^2));
        Assert.AreEqual(result.Shape.Length(^1), output.Shape.Length(^1));
        for (var r = 0; r < result.Shape.Length(^2); r++)
        {
            for (var c = 0; c < result.Shape.Length(^1); c++)
            {
                Assert.AreEqual(result[r, c], output[0, 0, r, c], $"Element mismatch @ row {r}, column {c}. Expected {result}, got {output}");
            }
        }
    }

    [TestMethod]
    public void TestTransposeConvolve2D_InPadding0OutPadding0Stride1()
    {
        Tensor<double> input = Tensor<double>.FromRectangularArray(new double[,]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });

        Tensor<double> kernel = Tensor<double>.FromRectangularArray(new double[,]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });

        Tensor<double> result_truth = Tensor<double>.FromRectangularArray(new double[,]{
            {1, 4, 10, 12, 9},
            {8, 26, 56, 54, 36},
            {30, 84, 165, 144, 90},
            {56, 134, 236, 186, 108},
            {49, 112, 190, 144, 81}
        });
        var result_predicted = input.TransposeConvolve2D(kernel);
        Assert.AreEqual(result_truth.Shape.Length(^2), result_predicted.Shape.Length(^2));
        Assert.AreEqual(result_truth.Shape.Length(^1), result_predicted.Shape.Length(^1));

        foreach (var (predicted, truth) in result_predicted.AsArray().Zip(result_truth.AsArray()))
        {
            Assert.AreEqual(truth, predicted, 0.0001);
        }
    }

    [TestMethod]
    public void TestSoftmax()
    {
        double[][][] input = [[[ 0.0637, -0.5338],
         [ 0.0342,  0.9319],
         [ 0.1470,  0.3923]],

        [[ 0.4529, -0.3350],
         [-0.3405, -0.2906],
         [ 0.1697,  1.5471]],

        [[ 1.2464,  0.9875],
         [-1.2910, -1.0184],
         [-0.4568,  1.8662]],

        [[ 0.7523, -1.0616],
         [ 0.3214,  0.8427],
         [-0.8794,  1.3009]]];

        Tensor<double> inputTensor = Tensor<double>.FromJaggedArray(input);

        double[][][] o1 = [[[0.1294, 0.1354],
         [0.3043, 0.4253],
         [0.3415, 0.0907]],

        [[0.1909, 0.1651],
         [0.2092, 0.1252],
         [0.3494, 0.2880]],

        [[0.4222, 0.6197],
         [0.0809, 0.0605],
         [0.1867, 0.3962]],

        [[0.2576, 0.0799],
         [0.4056, 0.3890],
         [0.1224, 0.2251]]];
        double[][][] o2 = [[[0.3270, 0.1273],
         [0.3175, 0.5513],
         [0.3554, 0.3214]],

        [[0.4534, 0.1161],
         [0.2051, 0.1214],
         [0.3416, 0.7625]],

        [[0.7929, 0.2823],
         [0.0627, 0.0380],
         [0.1444, 0.6797]],

        [[0.5419, 0.0546],
         [0.3522, 0.3663],
         [0.1060, 0.5792]]];
        double[][][] o3 = [[[0.6451, 0.3549],
         [0.2895, 0.7105],
         [0.4390, 0.5610]],

        [[0.6874, 0.3126],
         [0.4875, 0.5125],
         [0.2014, 0.7986]],

        [[0.5644, 0.4356],
         [0.4323, 0.5677],
         [0.0892, 0.9108]],

        [[0.8598, 0.1402],
         [0.3726, 0.6274],
         [0.1015, 0.8985]]];

        Assert.AreEqual(true, inputTensor.Softmax(0).Equals(Tensor<double>.FromJaggedArray(o1), 0.001f));
        Assert.AreEqual(true, inputTensor.Softmax(1).Equals(Tensor<double>.FromJaggedArray(o2), 0.001f));
        Assert.AreEqual(true, inputTensor.Softmax(2).Equals(Tensor<double>.FromJaggedArray(o3), 0.001f));
    }

    [TestMethod]
    public void TestSerialization()
    {
        var random = Random.Shared;
        Tensor<float> something = Tensor<float>.Generate(new TensorShape(3, 2, 4, 4), () => (float)random.NextDouble());

        using (var writer = new StreamWriter("Tensor.Test.TestSerialization.json"))
        {
            something.SaveJson(writer);
        }
        using (var stream = File.Open("Tensor.Test.TestSerialization.json", FileMode.Open))
        {
            var read = TensorExport.FromJson<float>(stream);
            if (!read.Shape.Equals(something.Shape))
                Assert.Fail("Failed to obtain correct shape from read tensor");

            var sarray = something.AsArray();
            var rarray = read.AsArray();
            if (sarray.Length != rarray.Length)
                Assert.Fail("Internal arrays are of differing lengths");
            
            foreach (var pair in sarray.Zip(rarray))
            {
                Assert.AreEqual(pair.First, pair.Second, 0.001f);
            }
        }

        using (var writer = new StreamWriter("Tensor.Test.TestSerialization.xml"))
        {
            something.SaveSpreadsheetML(writer);
        }

        using (var writer = new BinaryWriter(File.Open("Tensor.Test.TestSerialization.npy", FileMode.Create)))
        {
            something.SaveNpy(writer);
        }
    }

}
