using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test;

[TestClass]
public class BasicMatrix {
    [TestMethod]
    public void TestCreation() {
        Matrix<double> matrix = new Matrix<double>(5, 2);
        Assert.AreEqual(5, matrix.Rows);
        Assert.AreEqual(2, matrix.Columns);

        Matrix<double> m2 = new double[,]{
            {1, 2, 3},
            {4, 5, 6}
        };
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
        Matrix<double> m2 = new double[,]{
            {1, 2, 3},
            {4, 5, 6}
        };
        Assert.AreEqual(false, m2.IsTransposed);
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
        Assert.AreEqual(true, transposed.IsTransposed);
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
        Matrix<double> A = new double[,] {
            { 1, 2 },
            { 3, 4 }
        };

        var B = A.Map((x) => x * x);
        Assert.AreEqual(1, B[0,0]);
        Assert.AreEqual(2*2, B[0,1]);
        Assert.AreEqual(3*3, B[1,0]);
        Assert.AreEqual(4*4, B[1,1]);

        var C = A.Map<float>((x) => (float)(x * x));
    }

    [TestMethod]
    public void TestFlatten() {
        Matrix<double> A = new double[,] {
            { 1, 2 },
            { 3, 4 }
        };

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
        Matrix<double> A = new double[,] {
            { 1, 2, 3},
            { 4, 5, 6}
        };

        var row0 = A.ExtractRow(0);
        var row1 = A.ExtractRow(1);
        Assert.AreEqual(3, row0.Dimensionality);
        Assert.AreEqual(3, row1.Dimensionality);
        Assert.AreEqual(1, row0[0]); Assert.AreEqual(2, row0[1]); Assert.AreEqual(3, row0[2]);
        Assert.AreEqual(4, row1[0]); Assert.AreEqual(5, row1[1]); Assert.AreEqual(6, row1[2]);

        var col0 = A.ExtractColumn(0);
        var col1 = A.ExtractColumn(1);
        var col2 = A.ExtractColumn(2);
        Assert.AreEqual(2, col0.Dimensionality);
        Assert.AreEqual(2, col1.Dimensionality);
        Assert.AreEqual(2, col2.Dimensionality);
        Assert.AreEqual(1, col0[0]); Assert.AreEqual(4, col0[1]); 
        Assert.AreEqual(2, col1[0]); Assert.AreEqual(5, col1[1]); 
        Assert.AreEqual(3, col2[0]); Assert.AreEqual(6, col2[1]); 
    }

    [TestMethod]
    public void TestElementWise() {
        Matrix<double> A = new double[,] {
            { 1, 2 },
            { 3, 4 }
        };
        Matrix<double> B = new double[,] {
            { 5, 6 },
            { 7, 8 }
        };

        var C = A.ElementWise(B, (a, b) => a + b);
        Assert.AreEqual(1+5, C[0,0]);
        Assert.AreEqual(2+6, C[0,1]);
        Assert.AreEqual(3+7, C[1,0]);
        Assert.AreEqual(4+8, C[1,1]);
    }

    [TestMethod]
    public void TestReshape() {
        Matrix<double> A = new double[,] {
            { 1, 2 },
            { 3, 4 }
        };
        var row = A.Reshape(new Shape(rows: 1, columns: 4)).ToArray();
        var col = A.Reshape(new Shape(rows: 4, columns: 1)).ToArray();

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
        Matrix<double> A = new double[,] {
            { 1, 2 },
            { 3, 4 }
        };

        Matrix<double> B = new double[,] {
            { 5, 6 },
            { 7, 8 }
        };

        // Expected result of A * B
        Matrix<double> expected = new double[,] {
            { 19, 22 },
            { 43, 50 }
        };

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
    public void TestIncompatibleMultiply() {
        // Arrange: Define two incompatible matrices
        Matrix<double> A = new double[,] {
            { 1, 2, 3 }
        }; // 1x3 matrix

        Matrix<double> B = new double[,] {
            { 4, 5 },
            { 6, 7 },
            { 8, 9 }
        }; // 3x2 matrix

        // Act & Assert: Try to multiply the matrices (this should throw an exception)
        Assert.ThrowsException<ArithmeticException>(() => B*A);
    }
}