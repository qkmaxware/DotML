namespace DotML.Test;

[TestClass]
public class SafetensorsVec {
    [TestMethod]
    public void TestSingleMatrix() {
        var filename = "TestSingleMatrix.safetensors";
        var key = "identity";
        var matrix = Matrix<double>.Identity(5);

        Safetensors sb = new Safetensors();
        sb.Add(key, matrix);
        sb.WriteToFile(filename);

        sb = Safetensors.ReadFromFile(filename);
        Assert.AreEqual(1, sb.Keys().Count());
        Assert.AreEqual(true, sb.ContainsKey(key));

        var tensor = sb.GetTensor<double>(key);
        Assert.AreEqual(matrix.Rows, tensor.Rows);
        Assert.AreEqual(matrix.Columns, tensor.Columns);
        for (var i = 0; i < matrix.Size; i++) {
            Assert.AreEqual(matrix[i], tensor[i], 0.001, "Loaded matrix differs from the source");
        }
    }

    [TestMethod]
    public void TestMultipleMatrix() {
        var filename = "TestMultipleMatrix.safetensors";
        var key = "identity";
        var matrix = Matrix<double>.Identity(3);
        var key2 = "ordered";
        var matrix2 = Matrix<double>.FromFlattened(4, 4, [
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        ]);

        Safetensors sb = new Safetensors();
        sb.Add(key, matrix);
        sb.Add(key2, matrix2);
        sb.WriteToFile(filename);

        sb = Safetensors.ReadFromFile(filename);
        Assert.AreEqual(2, sb.Keys().Count());
        Assert.AreEqual(true, sb.ContainsKey(key));
        Assert.AreEqual(true, sb.ContainsKey(key2));

        var tensor = sb.GetTensor<double>(key);
        Assert.AreEqual(matrix.Rows, tensor.Rows);
        Assert.AreEqual(matrix.Columns, tensor.Columns);
        for (var i = 0; i < matrix.Size; i++) {
            Assert.AreEqual(matrix[i], tensor[i], 0.001, "Loaded matrix differs from the source");
        }

        tensor = sb.GetTensor<double>(key2);
        Assert.AreEqual(matrix2.Rows, tensor.Rows);
        Assert.AreEqual(matrix2.Columns, tensor.Columns);
        for (var i = 0; i < matrix2.Size; i++) {
            Assert.AreEqual(matrix2[i], tensor[i], 0.001, "Loaded matrix differs from the source");
        }
    }
}