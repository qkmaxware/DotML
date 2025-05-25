namespace DotML;

public static class FeatureSetExtensions {

    public static BatchedFeatureSet<double> ToDoubleSet(this BatchedFeatureSet<float> floats) => floats.Transform(x => (double)x);
    public static FeatureSet<double> ToDoubleSet(this FeatureSet<float> floats) => floats.Transform(x => (double)x);
    public static Matrix<double> ToDoubleSet(this Matrix<float> floats) => floats.Transform(x => (double)x);


    public static BatchedFeatureSet<float> ToFloatSet(this BatchedFeatureSet<double> floats) => floats.Transform(x => (float)x);
    public static FeatureSet<float> ToFloatSet(this FeatureSet<double> floats) => floats.Transform(x => (float)x);
    public static Matrix<float> ToFloatSet(this Matrix<double> floats) => floats.Transform(x => (float)x);
}