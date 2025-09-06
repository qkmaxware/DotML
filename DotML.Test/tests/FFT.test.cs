using System.Numerics;
using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test;

[TestClass]
public class FFTTest {
   /* 
    [TestMethod]
    public void TestFFT() {
        Complex[] series = [1, 2, 3, 4]; // Must have length be a power of 2
        Complex[] expected = [new Complex(10, 0), new Complex(-2, 2), new Complex(-2, 0), new Complex(-2, -2)]; 

        var computed = new Complex[series.Length];
        Array.Copy(series, computed, series.Length);
        CooleyTukey.FFT(computed);

        Assert.AreEqual(expected.Length, computed.Length);
        for (var i = 0; i < expected.Length; i++) {
            Assert.AreEqual(expected[i].Real, computed[i].Real, 0.001, string.Join(',', expected));
            Assert.AreEqual(expected[i].Imaginary, computed[i].Imaginary, 0.001, string.Join(',', expected));
        }
    }

    public void TestInverseFFT() {
        Complex[] expected = [1, 2, 3, 4]; // Must have length be a power of 2
        Complex[] series = [new Complex(10, 0), new Complex(-2, 2), new Complex(-2, 0), new Complex(-2, -2)]; 
    
        var computed = new Complex[series.Length];
        Array.Copy(series, computed, series.Length);
        CooleyTukey.iFFT(computed);

        Assert.AreEqual(expected.Length, computed.Length);
        for (var i = 0; i < expected.Length; i++) {
            Assert.AreEqual(expected[i].Real, computed[i].Real, 0.001, string.Join(',', expected));
            Assert.AreEqual(expected[i].Imaginary, computed[i].Imaginary, 0.001, string.Join(',', expected));
        }
    }*/
}