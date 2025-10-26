using DotML.Network;
using DotML.Network.Initialization;
using DotML.Network.IO;
using DotML.Network.Templates;

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

        var tensor = sb.GetMatrix<double>(key);
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

        var tensor = sb.GetMatrix<double>(key);
        Assert.AreEqual(matrix.Rows, tensor.Rows);
        Assert.AreEqual(matrix.Columns, tensor.Columns);
        for (var i = 0; i < matrix.Size; i++) {
            Assert.AreEqual(matrix[i], tensor[i], 0.001, "Loaded matrix differs from the source");
        }

        tensor = sb.GetMatrix<double>(key2);
        Assert.AreEqual(matrix2.Rows, tensor.Rows);
        Assert.AreEqual(matrix2.Columns, tensor.Columns);
        for (var i = 0; i < matrix2.Size; i++) {
            Assert.AreEqual(matrix2[i], tensor[i], 0.001, "Loaded matrix differs from the source");
        }
    }

    [TestMethod]
    public void TestWholeModel() {
        var filename = "lenet.safetensors";

        var factory = new LeNetFactory();
        var lenet = (ArchitectureBlock)factory.Make(new LeNetFactory.BuildSettingsV5 { OutputClasses = 10 });
        var initializer = new NormalXavierInitialization();
        lenet.Initialize(initializer);

        var writer = new SafetensorSerializer();
        lenet.Accept(writer);
        var saved = writer.ToSafetensors();
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

            Assert.AreEqual(saved_tensor.Rank, loaded_tensor.Rank);
            for (var r = 0; r < saved_tensor.Rank; r++) {
                Assert.AreEqual(saved_tensor.GetDimension(r), loaded_tensor.GetDimension(r));
            }
            var i = 0;
            foreach (var (saved_x, loaded_x) in saved_tensor.EnumerateElements().Zip(loaded_tensor.EnumerateElements())) {
                Assert.AreEqual(saved_x, loaded_x, 0.001, $"Loaded matrix differs from the source for key '{key}' at position {i}");
                i++;
            }
        }
    }
}