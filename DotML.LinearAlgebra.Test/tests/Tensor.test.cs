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
        var result_predicted = input.TransposeConvolve2D_OutputDriven(kernel);
        Assert.AreEqual(result_truth.Shape.Length(^2), result_predicted.Shape.Length(^2));
        Assert.AreEqual(result_truth.Shape.Length(^1), result_predicted.Shape.Length(^1));

        foreach (var (predicted, truth) in result_predicted.AsArray().Zip(result_truth.AsArray())) {
            Assert.AreEqual(truth, predicted, 0.0001);
        }
    }

}
