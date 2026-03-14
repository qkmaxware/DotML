using DotML;

public static class AssertExt {

    public static void AreEqual(Vec<double> y_truth, Vec<double> y_projected, double epsilon = 0.001) {
        Assert.AreEqual(y_truth.Dimensionality, y_projected.Dimensionality);
        for (var i = 0; i < y_truth.Dimensionality; i++) {
            Assert.AreEqual(y_truth[i], y_projected[i], epsilon, $"Element {i}");
        }
    }

    public static void AreEqual(Matrix<double> y_truth, Matrix<double> y_projected, double epsilon = 0.001) {
        Assert.AreEqual(y_truth.Size, y_projected.Size);
        for (var i = 0; i < y_truth.Size; i++) {
            Assert.AreEqual(y_truth[i], y_projected[i], epsilon, $"Element {i}");
        }
    }

    public static void AreEqual(Vec<float> y_truth, Vec<float> y_projected, float epsilon = 0.001f) {
        Assert.AreEqual(y_truth.Dimensionality, y_projected.Dimensionality);
        for (var i = 0; i < y_truth.Dimensionality; i++) {
            Assert.AreEqual(y_truth[i], y_projected[i], epsilon, $"Element {i}");
        }
    }

    public static void AreEqual(Matrix<float> y_truth, Matrix<float> y_projected, float epsilon = 0.001f) {
        Assert.AreEqual(y_truth.Size, y_projected.Size);
        for (var i = 0; i < y_truth.Size; i++) {
            Assert.AreEqual(y_truth[i], y_projected[i], epsilon, $"Element {i}");
        }
    }

    public static void AreEqual(Vec<double> y_truth, Vec<float> y_projected, float epsilon = 0.001f) {
        Assert.AreEqual(y_truth.Dimensionality, y_projected.Dimensionality);
        for (var i = 0; i < y_truth.Dimensionality; i++) {
            Assert.AreEqual(y_truth[i], y_projected[i], epsilon, $"Element {i}");
        }
    }

    public static void AreEqual(Matrix<double> y_truth, Matrix<float> y_projected, float epsilon = 0.001f) {
        Assert.AreEqual(y_truth.Size, y_projected.Size);
        for (var i = 0; i < y_truth.Size; i++) {
            Assert.AreEqual(y_truth[i], y_projected[i], epsilon, $"Element {i}");
        }
    }
}