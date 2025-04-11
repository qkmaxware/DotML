using DotML.Network;
using DotML.Network.Initialization;

namespace DotML.Test;

[TestClass]
public class TestSafetensors {
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

    [TestMethod]
    public void TestWholeModel() {
        var filename = "lenet.safetensors";
        var lenet = LeNet.Make(LeNet.Version.V5, 10);
        var initializer = new NormalXavierInitialization();
        lenet.Initialize(initializer);

        var saved = lenet.ToSafetensor();
        saved.WriteToFile(filename);

        var loaded = Safetensors.ReadFromFile(filename);
        Assert.AreNotEqual(0, loaded.Keys().Count());
        Assert.AreEqual(saved.Keys().Count(), loaded.Keys().Count());
        foreach (var key in saved.Keys()) {
            Assert.AreEqual(true, loaded.ContainsKey(key));
        }

        foreach (var key in saved.Keys()) {
            var saved_tensor = saved.GetTensor<double>(key);
            var loaded_tensor = loaded.GetTensor<double>(key);

            Assert.AreEqual(saved_tensor.Rows, loaded_tensor.Rows);
            Assert.AreEqual(saved_tensor.Columns, loaded_tensor.Columns);
            for (var i = 0; i < loaded_tensor.Size; i++) {
                Assert.AreEqual(saved_tensor[i], loaded_tensor[i], 0.001, $"Loaded matrix differs from the source for key '{key}' at position {i}");
            }
        }
    }
}