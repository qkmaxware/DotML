namespace DotML.Network;

/// <summary>
/// Delegate representing a vector loss function
/// </summary>
/// <param name="predicted">The predicted value, such as the result from a network</param>
/// <param name="true">The true value, such as the result from a training set</param>
/// <returns>loss</returns>
public delegate double LossFunction(Vec<double> predicted, Vec<double> @true);

/// <summary>
/// Container with some standard loss functions
/// </summary>
public static class LossFunctions {
    /// <summary>
    /// Mean squared error (MSE) loss function
    /// </summary>
    /// <param name="predicted">The predicted value, such as the result from a network</param>
    /// <param name="true">The true value, such as the result from a training set</param>
    /// <returns>Mean squared error</returns>
    public static double MeanSquaredError(Vec<double> predicted, Vec<double> @true) {
        double mse = 0.0;
        var N = Math.Max(predicted.Dimensionality, @true.Dimensionality);

        for (var i = 0; i < N; i++) {
            var to_square = (predicted[i] - @true[i]);
            mse += to_square * to_square;
        }

        return mse/N;
    }

    /// <summary>
    /// Mean absolute error (MAE) loss function
    /// </summary>
    /// <param name="predicted">The predicted value, such as the result from a network</param>
    /// <param name="true">The true value, such as the result from a training set</param>
    /// <returns>Mean absolute error</returns>
    public static double MeanAbsoluteError(Vec<double> predicted, Vec<double> @true) {
        double mae = 0.0;
        var N = Math.Max(predicted.Dimensionality, @true.Dimensionality);

        for (var i = 0; i < N; i++) {
            mae += Math.Abs(predicted[i] - @true[i]);
        }

        return mae/N;
    }

    /// <summary>
    /// Cross-Entropy loss function
    /// </summary>
    /// <param name="predicted">The predicted value, such as the result from a network</param>
    /// <param name="true">The true value, such as the result from a training set</param>
    /// <returns>Cross entropy loss</returns>
    public static double CrossEntropy (Vec<double> predicted, Vec<double> @true) {
        if (!IsLikelyAProbability(predicted))
            predicted = predicted.SoftmaxNormalized();

        // -SUM(exp_i * log(actual_i))
        var sum = 0.0;
        var m = Math.Max(predicted.Dimensionality, @true.Dimensionality);
        for (var i = 0; i < m; i++) {
            sum += @true[i] * Math.Log(Math.Max(predicted[i], 1e-8));
        }
        return -(1.0/m)*sum;
    }

    private static bool IsLikelyAProbability(Vec<double> predicted) {
        const double epsilon = 1e-8;
        
        var sum = 0.0;
        foreach (var p in predicted) {
            if (p < 0.0 || p > 1.0) {
                return false;
            }
            sum += p;
        }
        
        return Math.Abs(sum - 1.0) < epsilon;
    }
}