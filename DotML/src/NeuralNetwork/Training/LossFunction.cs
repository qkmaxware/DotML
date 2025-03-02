namespace DotML.Network.Training;

/// <summary>
/// A loss function between computed output vectors (predicted) and their expected values (true)
/// <see href="https://en.wikipedia.org/wiki/Loss_function"/>
/// </summary>
public abstract class LossFunction : DelegateObject<Vec<double>, Vec<double>, double> {
    /// <summary>
    /// Loss function name
    /// </summary>
    public string Name => this.GetType().Name;

    /// <summary>
    /// Compute the gradient of the loss function with respect to the predicted output
    /// </summary>
    /// <param name="predicted">The predicted vector as output from forward-propagation</param>
    /// <param name="true">The true vector expected as output</param>
    /// <returns>Gradient for use in backpropagation</returns>
    public abstract Vec<double> Gradient(Vec<double> predicted, Vec<double> @true);
}

/// <summary>
/// Container with some standard loss functions
/// </summary>
public static class LossFunctions {
    /// <summary>
    /// Enumerate over all loss functions in this container
    /// </summary>
    /// <returns>Enumerable of loss functions</returns>
    public static IEnumerable<LossFunction> EnumerateAll() {
        return typeof(LossFunctions)
            .GetProperties(System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public)
            .Where(prop => prop.CanRead && prop.PropertyType.IsAssignableTo(typeof(LossFunction)))
            .Select(prop => prop.GetValue(null))
            .OfType<LossFunction>();
    }

    /// <summary>
    /// Mean squared error (MSE) loss function
    /// </summary>
    public static LossFunction MeanSquaredError {get; private set;} = new Training.MeanSquaredError();

    /// <summary>
    /// Mean squared error (RMSE) loss function
    /// </summary>
    public static LossFunction RootMeanSquaredError {get; private set;} = new Training.RootMeanSquaredError();

    /// <summary>
    /// Mean absolute error (MAE) loss function
    /// </summary>
    public static LossFunction MeanAbsoluteError {get; private set;} = new Training.MeanAbsoluteError();

    /// <summary>
    /// Categorical cross-entropy loss function
    /// </summary>
    public static LossFunction CategoricalCrossEntropy {get; private set;} = new Training.CategoricalCrossEntropy();
}

/// <summary>
/// Mean squared error (MSE) loss function
/// </summary>
public class MeanSquaredError : LossFunction {
    public override double Invoke(Vec<double> predicted, Vec<double> @true) {
        double mse = 0.0;
        var N = Math.Min(predicted.Dimensionality, @true.Dimensionality);

        for (var i = 0; i < N; i++) {
            var to_square = predicted[i] - @true[i];
            mse += to_square * to_square;
        }

        return mse/N;
    }
    
    public override Vec<double> Gradient(Vec<double> predicted, Vec<double> @true) {
        return /*scalar * */ predicted - @true;
    }
}

/// <summary>
/// Mean squared error (RMSE) loss function
/// </summary>
public class RootMeanSquaredError : LossFunction {
    public override double Invoke(Vec<double> predicted, Vec<double> @true) {
        double mse = 0.0;
        var N = Math.Min(predicted.Dimensionality, @true.Dimensionality);

        for (var i = 0; i < N; i++) {
            var to_square = (predicted[i] - @true[i]);
            mse += to_square * to_square;
        }

        return Math.Sqrt(mse/N);
    }
    
    public override Vec<double> Gradient(Vec<double> predicted, Vec<double> @true) {
        return /*scalar * */ predicted - @true;
    }
}

/// <summary>
/// Mean absolute error (MAE) loss function
/// </summary>
public class MeanAbsoluteError : LossFunction {
    public override double Invoke(Vec<double> predicted, Vec<double> @true) {
        double mae = 0.0;
        var N = Math.Min(predicted.Dimensionality, @true.Dimensionality);

        for (var i = 0; i < N; i++) {
            mae += Math.Abs(predicted[i] - @true[i]);
        }

        return mae/N;
    }
    
    public override Vec<double> Gradient(Vec<double> predicted, Vec<double> @true) {
        return (predicted - @true).Transform(x => /*scalar * */ (double)Math.Sign(x));
    }
}

/// <summary>
/// Categorical cross-entropy loss function
/// </summary>
public class CategoricalCrossEntropy : LossFunction {

    private const double epsilon = 1e-15;

    /// <summary>
    /// Compute the loss of between a predicted and true vector
    /// </summary>
    /// <param name="predicted">The predicted vector as output from forward-propagation</param>
    /// <param name="true">The true vector expected as output</param>
    /// <returns>The computed loss between the predicted and true vectors</returns>
    public override double Invoke(Vec<double> predicted, Vec<double> @true) {
        if (predicted.Dimensionality != @true.Dimensionality) {
            throw new ArgumentException("Predicted and true vectors must have the same length.");
        }

        // Predicted must be a softmax distribution
        var predictedNormalized = predicted.IsLikelyAProbabilityDistribution() ? predicted : predicted.SoftmaxNormalized();

        // -SUM(exp_i * log(actual_i))
        var sum = 0.0;
        var M = predictedNormalized.Dimensionality; // Each dimension is a class
        for (var i = 0; i < M; i++) { 
            // @true is a class label, predicted is the predicted probability
            sum += @true[i] * Math.Log(Math.Max(predictedNormalized[i], epsilon));
        }
        return -(1.0/M)*sum;
    }
    
    /// <summary>
    /// Compute the gradient of the loss function with respect to the predicted output
    /// </summary>
    /// <param name="predicted">The predicted vector as output from forward-propagation</param>
    /// <param name="true">The true vector expected as output</param>
    /// <returns>Gradient for use in backpropagation</returns>
    public override Vec<double> Gradient(Vec<double> predicted, Vec<double> @true) {
        if (predicted.Dimensionality != @true.Dimensionality) {
            throw new ArgumentException("Predicted and true vectors must have the same length.");
        }

        // Predicted must be a softmax distribution
        var predictedNormalized = predicted.IsLikelyAProbabilityDistribution() ? predicted : predicted.SoftmaxNormalized();

        return predictedNormalized - @true; //Is it or isn't it what's written below?

        // dL/dYhat_i = - Ytrue_i / Yhat_u
        var grad = new Vec<double>(predicted.Dimensionality);
        for (var i = 0; i < grad.Dimensionality; i++) {
            var v = -(@true[i]/predictedNormalized[i]);
            grad[i] = double.IsNaN(v) ? 0.0 : v; // NaN's are bad! Do everything I can to avoid them.
        }

        return grad;
    }
}