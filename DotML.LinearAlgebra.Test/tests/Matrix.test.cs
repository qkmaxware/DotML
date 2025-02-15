namespace DotML.Test;

[TestClass]
public class BasicMatrix {
    [TestMethod]
    public void TestCreation() {
        Matrix<double> matrix = new Matrix<double>(5, 2);
        Assert.AreEqual(5, matrix.Rows);
        Assert.AreEqual(2, matrix.Columns);

        Matrix<double> m2 = new Matrix<double>(new double[,]{
            {1, 2, 3},
            {4, 5, 6}
        });
        Assert.AreEqual(2, m2.Rows);
        Assert.AreEqual(3, m2.Columns);
        Assert.AreEqual(1, m2[0, 0]);
        Assert.AreEqual(3, m2[0, 2]);
        Assert.AreEqual(4, m2[1, 0]);
        Assert.AreEqual(6, m2[1, 2]);

        var zero = Matrix<double>.Zeros(3);
        Assert.AreEqual(3, zero.Rows);
        Assert.AreEqual(3, zero.Columns);
        foreach (var element in zero)
            Assert.AreEqual(0, element);

        var ones = Matrix<double>.Ones(4);
        Assert.AreEqual(4, ones.Rows);
        Assert.AreEqual(4, ones.Columns);
        foreach (var element in ones)
            Assert.AreEqual(1, element);
    }

    [TestMethod]
    public void TestTranspose() {
        Matrix<double> m2 = new Matrix<double>(new double[,]{
            {1, 2, 3},
            {4, 5, 6}
        });
        Assert.AreEqual(2, m2.Rows);
        Assert.AreEqual(3, m2.Columns);
        Assert.AreEqual(1, m2[0, 0]);
        Assert.AreEqual(3, m2[0, 2]);
        Assert.AreEqual(4, m2[1, 0]);
        Assert.AreEqual(6, m2[1, 2]);
        var m2shape = m2.Shape;
        Assert.AreEqual(2, m2shape.Rows);
        Assert.AreEqual(3, m2shape.Columns);

        var transposed = m2.Transpose();
        Assert.AreEqual(3, transposed.Rows);
        Assert.AreEqual(2, transposed.Columns);
        Assert.AreEqual(1, transposed[0, 0]);
        Assert.AreEqual(4, transposed[0, 1]);
        Assert.AreEqual(3, transposed[2, 0]);
        Assert.AreEqual(6, transposed[2, 1]);
        var tshape = transposed.Shape;
        Assert.AreEqual(3, tshape.Rows);
        Assert.AreEqual(2, tshape.Columns);
    }

    [TestMethod]
    public void TestMap() {
        Matrix<double> A = new Matrix<double>(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });

        var B = A.Transform((x) => x * x);
        Assert.AreEqual(1, B[0,0]);
        Assert.AreEqual(2*2, B[0,1]);
        Assert.AreEqual(3*3, B[1,0]);
        Assert.AreEqual(4*4, B[1,1]);

        var C = A.Transform<float>((x) => (float)(x * x));
    }

    [TestMethod]
    public void TestFlatten() {
        Matrix<double> A = new Matrix<double>(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });

        var row_order = A.FlattenRows().ToArray();
        Assert.AreEqual(A.Size, row_order.Length);
        Assert.AreEqual(A[0,0], row_order[0]);  // 1
        Assert.AreEqual(A[0,1], row_order[1]);  // 2
        Assert.AreEqual(A[1,0], row_order[2]);  // 3
        Assert.AreEqual(A[1,1], row_order[3]);  // 4

        var col_order = A.FlattenColumns().ToArray();
        Assert.AreEqual(A.Size, col_order.Length);
        Assert.AreEqual(A[0,0], col_order[0]);  // 1
        Assert.AreEqual(A[1,0], col_order[1]);  // 3
        Assert.AreEqual(A[0,1], col_order[2]);  // 2
        Assert.AreEqual(A[1,1], col_order[3]);  // 4
    }

    [TestMethod]
    public void TestExtract() {
        Matrix<double> A = new Matrix<double>(new double[,] {
            { 1, 2, 3},
            { 4, 5, 6}
        });

        var row0 = A.ExtractRowVector(0);
        var row1 = A.ExtractRowVector(1);
        Assert.AreEqual(3, row0.Dimensionality);
        Assert.AreEqual(3, row1.Dimensionality);
        Assert.AreEqual(1, row0[0]); Assert.AreEqual(2, row0[1]); Assert.AreEqual(3, row0[2]);
        Assert.AreEqual(4, row1[0]); Assert.AreEqual(5, row1[1]); Assert.AreEqual(6, row1[2]);

        var col0 = A.ExtractColumnVector(0);
        var col1 = A.ExtractColumnVector(1);
        var col2 = A.ExtractColumnVector(2);
        Assert.AreEqual(2, col0.Dimensionality);
        Assert.AreEqual(2, col1.Dimensionality);
        Assert.AreEqual(2, col2.Dimensionality);
        Assert.AreEqual(1, col0[0]); Assert.AreEqual(4, col0[1]); 
        Assert.AreEqual(2, col1[0]); Assert.AreEqual(5, col1[1]); 
        Assert.AreEqual(3, col2[0]); Assert.AreEqual(6, col2[1]); 
    }

    [TestMethod]
    public void TestElementWise() {
        Matrix<double> A = new Matrix<double>(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });
        Matrix<double> B = new Matrix<double>(new double[,] {
            { 5, 6 },
            { 7, 8 }
        });

        var C = A.ElementWise(B, (a, b) => a + b);
        Assert.AreEqual(1+5, C[0,0]);
        Assert.AreEqual(2+6, C[0,1]);
        Assert.AreEqual(3+7, C[1,0]);
        Assert.AreEqual(4+8, C[1,1]);
    }

    [TestMethod]
    public void TestReshape() {
        Matrix<double> A = new Matrix<double>(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });
        var row = A.Reshape(new Shape2D(rows: 1, columns: 4)).ToArray();
        var col = A.Reshape(new Shape2D(rows: 4, columns: 1)).ToArray();

        Assert.AreEqual(1, row.Length);
        Assert.AreEqual(1, row[0].Rows);
        Assert.AreEqual(4, row[0].Columns);
        
        Assert.AreEqual(1, col.Length);
        Assert.AreEqual(4, col[0].Rows);
        Assert.AreEqual(1, col[0].Columns);
    }

    [TestMethod]
    public void TestCompatibleMultiply() {
        // Arrange: Define two matrices to multiply
        Matrix<double> A = new Matrix<double>(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });

        Matrix<double> B = new Matrix<double>(new double[,] {
            { 5, 6 },
            { 7, 8 }
        });

        // Expected result of A * B
        Matrix<double> expected = new Matrix<double>(new double[,] {
            { 19, 22 },
            { 43, 50 }
        });

        // Act: Multiply matrices A and B
        var result = A * B;

        // Assert: Verify the result matches the expected output
        Assert.AreEqual(expected.Rows, result.Rows);
        Assert.AreEqual(expected.Columns, result.Columns);
        foreach (var pair in expected.Zip(result)) {
            Assert.AreEqual(pair.First, pair.Second, 0.01);
        }
    }

    [TestMethod]
    public void TestCompatibleMultiplyVector() {
        // Arrange: Define two matrices to multiply
        Matrix<double> A = new Matrix<double>(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });

        Vec<double> B = new double[] {
            5,
            6 
        };

        // Expected result of A * B
        Vec<double> expected = new double[] {
            17,
            39
        };

        // Act: Multiply matrices A and B
        var result = A * B;

        // Assert: Verify the result matches the expected output
        Assert.AreEqual(expected.Dimensionality, result.Dimensionality);
        foreach (var pair in expected.Zip(result)) {
            Assert.AreEqual(pair.First, pair.Second, 0.01);
        }
    }

    [TestMethod]
    public void TestCompatibleMultiplyTransposed() {
        // Arrange: Define two matrices to multiply
        Matrix<double> A = new Matrix<double>(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });

        Matrix<double> B = new Matrix<double>(new double[,] {
            { 5, 6 },
            { 7, 8 }
        });

        // Expected result of A * B
        Matrix<double> expected = A.Transpose() * B;

        // Act: Multiply matrices A and B
        var result = A.MultiplyTransposedWith(B);

        // Assert: Verify the result matches the expected output
        Assert.AreEqual(expected.Rows, result.Rows);
        Assert.AreEqual(expected.Columns, result.Columns);
        foreach (var pair in expected.Zip(result)) {
            Assert.AreEqual(pair.First, pair.Second, 0.01);
        }
    }

    [TestMethod]
    public void TestIncompatibleMultiply() {
        // Arrange: Define two incompatible matrices
        Matrix<double> A = new Matrix<double>(new double[,] {
            { 1, 2, 3 }
        }); // 1x3 matrix

        Matrix<double> B = new Matrix<double>(new double[,] {
            { 4, 5 },
            { 6, 7 },
            { 8, 9 }
        }); // 3x2 matrix

        // Act & Assert: Try to multiply the matrices (this should throw an exception)
        Assert.ThrowsException<ArithmeticException>(() => B*A);
    }

    [TestMethod]
    public void TestConvolve() {
        var kernels = new Matrix<double>(new double[,] {
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1}
            });
        var input = new Matrix<double>(new double[,]{
            {1, 1, 1, 0, 0},
            {0, 1, 1, 1, 0},
            {0, 0, 1, 1, 1},
            {0, 0, 1, 1, 0},
            {0, 1, 1, 0, 0},
        });
        var output = input.Convolve(kernels);

        Matrix<double> result = new Matrix<double>(new double[,] {
            {4, 3, 4},
            {2, 4, 3},
            {2, 3, 4}
        });

        Assert.AreEqual(result.Rows, output.Rows);
        Assert.AreEqual(result.Columns, output.Columns);
        for (var r = 0; r < result.Rows; r++) {
            for (var c = 0; c < result.Columns; c++) {
                Assert.AreEqual(result[r, c], output[r, c], $"Element mismatch @ row {r}, column {c}. Expected {result}, got {output}");
            }
        }
    }

    [TestMethod]
    public void TestTransposeConvolve() {
        Matrix<double> input = new Matrix<double>(new double[,]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });

        Matrix<double> kernel = new Matrix<double>(new double[,]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });

        Matrix<double> result_truth = new Matrix<double>(new double[,]{
            {1, 4, 10, 12, 9},
            {8, 26, 56, 54, 36},
            {30, 84, 165, 144, 90},
            {56, 134, 236, 186, 108},
            {49, 112, 190, 144, 81}
        });
        var result_predicted = input.TransposeConvolve(kernel);
        Assert.AreEqual(result_truth.Rows, result_predicted.Rows);
        Assert.AreEqual(result_truth.Columns, result_predicted.Columns);

        foreach (var (predicted, truth) in result_predicted.Zip(result_truth)) {
            Assert.AreEqual(truth, predicted, 0.0001);
        }
    }

    [TestMethod]
    public void TestTransposeConvolve2() {
        Matrix<double> input = Matrix<double>.FromFlattened(3, 3, [
            1.2390981912612915,
            -0.27579402923583984,
            -1.463151216506958,
            0.8923978209495544,
            -1.5148130655288696,
            0.35458904504776,
            0.7004590630531311,
            1.1337686777114868,
            -0.1760110855102539
        ]);

        Matrix<double> kernel = Matrix<double>.FromFlattened(3, 3, [
            0.16329312324523926,
            0.09592697024345398,
            0.3095523416996002,
            0.244705468416214,
            -0.004788994789123535,
            0.1564023196697235,
            -0.33284634351730347,
            -0.2813574969768524,
            0.1707456409931183
        ]);

        Matrix<double> result_truth = Matrix<double>.FromFlattened(5, 5, [
            -0.005934034939855337,
            0.12630951404571533,
            0.0013207761803641915,
            -0.40117594599723816,
            0.007007023319602013,
            -0.26302453875541687,
            0.3322529196739197,
            -0.06771471351385117,
            0.028901897370815277,
            0.44568321108818054,
            -0.004273688420653343,
            -0.23110996186733246,
            0.007254431955516338,
            -0.15015040338039398,
            -0.0016981250373646617,
            -0.18388989567756653,
            1.0585384368896484,
            0.5349630117416382,
            -0.05445203185081482,
            -0.11665049195289612,
            -0.0033544946927577257,
            0.38699281215667725,
            -0.005429612472653389,
            0.13425317406654358,
            0.0008429161971434951
        ]);
        var result_predicted = input.TransposeConvolve(kernel, 
            flip_kernel: false, 
            outputStrideX: 2, outputStrideY: 2, 
            outputPaddingX: 1,
            outputPaddingY: 1,
            bias: 0
        );
        Assert.AreEqual(result_truth.Rows, result_predicted.Rows);
        Assert.AreEqual(result_truth.Columns, result_predicted.Columns);

        foreach (var (predicted, truth) in result_predicted.Zip(result_truth)) {
            Assert.AreEqual(truth, predicted, 0.0001);
        }
    }

}