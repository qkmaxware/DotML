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
    /// <returns>Mean squared erro</returns>
    public static double MeanSquaredError(Vec<double> predicted, Vec<double> @true) {
        double mse = 0.0;
        var N = Math.Max(predicted.Dimensionality, @true.Dimensionality);

        for (var i = 0; i < N; i++) {
            var to_square = (predicted[i] - @true[i]);
            mse += to_square * to_square;
        }

        return mse/N;
    }
}