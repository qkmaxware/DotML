using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test;

[TestClass]
public class DataSizeTest {

    [TestMethod]
    public void TestCreation() {
        var size = new DataSize(1000, DataUnit.Bytes);

        Assert.AreEqual(1000, size.ValueAs(DataUnit.Bytes));
        Assert.AreEqual(100, size.ValueAs(DataUnit.Decabyte));
        Assert.AreEqual(10, size.ValueAs(DataUnit.Hectobyte));
        Assert.AreEqual(1, size.ValueAs(DataUnit.Kilobyte));

        Assert.AreEqual("1000b", size.ToString());
    }

    [TestMethod]
    public void TestCreationFromItemCount() {
        var count = 1500;

        var size = DataSize.FromValues32(count);

        Assert.AreEqual("6Kb", size.ToString());
    }   
}