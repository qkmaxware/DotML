namespace DotML.Test;

[TestClass]
public class TestVec {

    [TestMethod]
    public void TestCreate() {
        var v0 = new Vec<double>();
        Assert.AreEqual(0, v0.Dimensionality);

        var v1 = new Vec<double>(50);
        Assert.AreEqual(50, v1.Dimensionality);
        for (var i = 0; i < 50; i++) {
            Assert.AreEqual(0.0, v1[i]);
        }

        var v2 = new Vec<double>(25, 5.0);
        Assert.AreEqual(25, v2.Dimensionality);
        for (var i = 0; i < 25; i++) {
            Assert.AreEqual(5.0, v2[i]);
        }

        var v3 = new Vec<double>(1, 2, 3, 4, 5, 6);
        Assert.AreEqual(6, v3.Dimensionality);
        for (var i = 0; i < 6; i++) {
            Assert.AreEqual(i + 1.0, v3[i]);
        }
    }

    [TestMethod]
    public void TestClone() {
        var v0 = new Vec<double>(0, -1, 5, 0);
        var v1 = v0.Clone();

        v1[2] = 7;
        Assert.AreEqual(v0.Dimensionality, v1.Dimensionality);
        Assert.AreEqual(5.0, v0[2]);
        Assert.AreEqual(7.0, v1[2]);
    }

    [TestMethod]
    public void TestBounds() {
        var v0 = new Vec<double>(0, -1, 5, 0);
        Assert.AreEqual(-1, v0.MinValue);
        Assert.AreEqual(5, v0.MaxValue);

        Assert.AreEqual(1, v0.IndexOfMinValue());
        Assert.AreEqual(2, v0.IndexOfMaxValue());
    }

    [TestMethod]
    public void TestLength() {
        var v0 = new Vec<double>(5.0, 0.0);
        Assert.AreEqual(25, v0.SqrLength());
        Assert.AreEqual(5, v0.Length());

        var v1 = new Vec<double>(0.0, -3.0);
        Assert.AreEqual(9, v1.SqrLength());
        Assert.AreEqual(3, v1.Length());

        var v2 = new Vec<double>(3.0, 6.0);
        Assert.AreEqual(45, v2.SqrLength());
        Assert.AreEqual(Math.Sqrt(45), v2.Length(), 0.0001);
    }

    [TestMethod]
    public void TestDot() {
        var a = new Vec<double>(1.0, 2.0, 3.0);
        var b = new Vec<double>(2.0, 4.0, 6.0);
        var c = new Vec<double>(1.0, 2.0);

        Assert.AreEqual(1 * 2 + 2 * 4 + 3 * 6, a.Dot(b));
        Assert.ThrowsException<ArithmeticException>(() => a.Dot(c));
    }

    [TestMethod]
    public void TestHadamard() {
        var a = new Vec<double>(1.0, 2.0, 3.0);
        var b = new Vec<double>(2.0, 4.0, 6.0);
        var c = new Vec<double>(1.0, 2.0);

        var ab = a.Hadamard(b);
        Assert.AreEqual(a.Dimensionality, ab.Dimensionality);
        Assert.AreEqual(1 * 2, ab[0]);
        Assert.AreEqual(2 * 4, ab[1]);
        Assert.AreEqual(3 * 6, ab[2]);
        Assert.ThrowsException<ArithmeticException>(() => a.Dot(c));
    }

    [TestMethod]
    public void TestTransform() {
        var a = new Vec<double>(1.0, 2.0, 3.0);
        var b = a.Transform((v) => v * 2);
        Assert.AreEqual(a.Dimensionality, b.Dimensionality);
        for (var i = 0; i < a.Dimensionality; i++) 
            Assert.AreEqual(a[i] * 2, b[i]);

        b = a.Transform((ind, v) => ind.Value == 1 ? 5.0 : 2*v);
        Assert.AreEqual(a.Dimensionality, b.Dimensionality);
        for (var i = 0; i < a.Dimensionality; i++) 
            if (i == 1)
                Assert.AreEqual(5.0, b[i]);
            else 
                Assert.AreEqual(a[i] * 2, b[i]);
    }

    [TestMethod]
    public void TestElementWise() {
        var a = new Vec<double>(1.0, 2.0, 3.0);
        var b = new Vec<double>(2.0, 4.0, 6.0);
        var c = new Vec<double>(1.0, 2.0);

        var ab = a.ElementWise(b, (aa, bb) => aa * bb);
        Assert.AreEqual(a.Dimensionality, ab.Dimensionality);
        Assert.AreEqual(1 * 2, ab[0]);
        Assert.AreEqual(2 * 4, ab[1]);
        Assert.AreEqual(3 * 6, ab[2]);
        Assert.ThrowsException<ArithmeticException>(() => a.ElementWise(c, (aa, cc) => aa * cc));
    }

    [TestMethod]
    public void TestApply() {
        var a = new Vec<double>(1.0, 2.0, 3.0);
        var b = a.Clone(); b.Apply((v) => v * 2);
        Assert.AreEqual(a.Dimensionality, b.Dimensionality);
        for (var i = 0; i < a.Dimensionality; i++) 
            Assert.AreEqual(a[i] * 2, b[i]);

        b = a.Clone(); b.Apply((ind, v) => ind.Value == 1 ? 5.0 : 2*v);
        Assert.AreEqual(a.Dimensionality, b.Dimensionality);
        for (var i = 0; i < a.Dimensionality; i++) 
            if (i == 1)
                Assert.AreEqual(5.0, b[i]);
            else 
                Assert.AreEqual(a[i] * 2, b[i]);
    }

    [TestMethod]
    public void TestScale() {
        var a = new Vec<double>(1.0, 2.0, 3.0);
        var b = 4 * a;

        for (var i = 0; i < a.Dimensionality; i++)
            Assert.AreEqual(a[i] * 4, b[i]);
    }

    private static Random rng = new Random();

    [TestMethod]
    public void TestAdd() {
        var v0 = new Vec<double>(10, () => rng.Next(10));
        var v1 = new Vec<double>(10, () => rng.Next(10));
        var v2 = new Vec<double>(5, () => rng.Next(10));

        var res = v0.AddedWith(v1);
        Assert.AreEqual(v0.Dimensionality, res.Dimensionality);
        for(var i = 0; i < v0.Dimensionality; i++) {
            Assert.AreEqual(v0[i] + v1[i], res[i]);
        }
        Assert.ThrowsException<ArithmeticException>(() => v0.AddedWith(v2));
    }

    [TestMethod]
    public void TestSub() {
        var v0 = new Vec<double>(10, () => rng.Next(10));
        var v1 = new Vec<double>(10, () => rng.Next(10));
        var v2 = new Vec<double>(5, () => rng.Next(10));

        var res = v0.SubtractWith(v1);
        Assert.AreEqual(v0.Dimensionality, res.Dimensionality);
        for(var i = 0; i < v0.Dimensionality; i++) {
            Assert.AreEqual(v0[i] - v1[i], res[i]);
        }
        Assert.ThrowsException<ArithmeticException>(() => v0.SubtractWith(v2));
    }

}