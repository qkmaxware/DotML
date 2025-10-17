using System.Numerics;

namespace DotML.Network.Training;

/// <summary>
/// A loss function between computed output vectors (predicted) and their expected values (true)
/// <see href="https://en.wikipedia.org/wiki/Loss_function"/>
/// </summary>
public abstract class LossFunction
{
    /// <summary>
    /// Loss function name
    /// </summary>
    public string Name => this.GetType().Name;

    /// <summary>
    /// Compute the loss of the predicted output compared against the ground truth values.
    /// </summary>
    /// <param name="predicted">The predicted vector as output from forward-propagation</param>
    /// <param name="true">The true vector expected as output</param>
    /// <returns>computed loss</returns>
    public abstract float Invoke(ReadOnlySpan<float> predicted, ReadOnlySpan<float> @true);

    /// <summary>
    /// Compute the loss of the predicted output compared against the ground truth values.
    /// </summary>
    /// <param name="predicted">The predicted vector as output from forward-propagation</param>
    /// <param name="true">The true vector expected as output</param>
    /// <returns>computed loss</returns>
    public float Invoke(Vec<float> predicted, Vec<float> @true) => Invoke(predicted.AsSpan(), @true.AsSpan());

    /// <summary>
    /// Compute the gradient of the loss function with respect to the predicted output
    /// </summary>
    /// <param name="gradient">Span to insert the gradient computation into</param>
    /// <param name="predicted">The predicted vector as output from forward-propagation</param>
    /// <param name="true">The true vector expected as output</param>
    public abstract void Gradient(Span<float> gradient, ReadOnlySpan<float> predicted, ReadOnlySpan<float> @true);

    /// <summary>
    /// Compute the gradient of the loss function with respect to the predicted output
    /// </summary>
    /// <param name="predicted">The predicted vector as output from forward-propagation</param>
    /// <param name="true">The true vector expected as output</param>
    /// <returns>Gradient for use in backpropagation</returns>
    public Vec<float> Gradient(Vec<float> predicted, Vec<float> @true)
    {
        float[] vs = new float[predicted.Dimensionality];
        Gradient(vs, predicted.AsSpan(), @true.AsSpan());
        return Vec<float>.Wrap(vs);
    }
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
    public override float Invoke(ReadOnlySpan<float> predicted, ReadOnlySpan<float> @true) {
        float mse = 0.0f;
        var N = Math.Min(predicted.Length, @true.Length);

        for (var i = 0; i < N; i++) {
            var to_square = predicted[i] - @true[i];
            mse += to_square * to_square;
        }

        return mse/N;
    }
    
    public override void Gradient(Span<float> gradient, ReadOnlySpan<float> predicted, ReadOnlySpan<float> @true) {
        for (var i = 0; i < predicted.Length; i++)
            gradient[i] = predicted[i] - @true[i];
    }
}

/// <summary>
/// Mean squared error (RMSE) loss function
/// </summary>
public class RootMeanSquaredError : LossFunction {
    public override float Invoke(ReadOnlySpan<float> predicted, ReadOnlySpan<float> @true) {
        float mse = 0.0f;
        var N = Math.Min(predicted.Length, @true.Length);

        for (var i = 0; i < N; i++) {
            var to_square = (predicted[i] - @true[i]);
            mse += to_square * to_square;
        }

        return MathF.Sqrt(mse/N);
    }
    
    public override void Gradient(Span<float> gradient, ReadOnlySpan<float> predicted, ReadOnlySpan<float> @true) {
        for (var i = 0; i < predicted.Length; i++)
            gradient[i] = predicted[i] - @true[i];
    }
}

/// <summary>
/// Mean absolute error (MAE) loss function
/// </summary>
public class MeanAbsoluteError : LossFunction {
    public override float Invoke(ReadOnlySpan<float> predicted, ReadOnlySpan<float> @true) {
        float mae = 0.0f;
        var N = Math.Min(predicted.Length, @true.Length);

        for (var i = 0; i < N; i++) {
            mae += MathF.Abs(predicted[i] - @true[i]);
        }

        return mae/N;
    }
    
    public override void Gradient(Span<float> gradient, ReadOnlySpan<float> predicted, ReadOnlySpan<float> @true) {
         for (var i = 0; i < predicted.Length; i++)
            gradient[i] = Math.Sign(predicted[i] - @true[i]);
    }
}

/// <summary>
/// Categorical cross-entropy loss function
/// </summary>
public class CategoricalCrossEntropy: LossFunction
{
    /// <summary>
    /// Computes the categorical cross-entropy loss for a set of predictions (logits).
    /// </summary>
    /// <param name="predicted">Predicted logits (raw network outputs)</param>
    /// <param name="truth">True labels in one-hot encoding</param>
    /// <returns>The computed cross-entropy loss.</returns>
    public override float Invoke(ReadOnlySpan<float> predicted, ReadOnlySpan<float> truth)
    {
        // Step 1: Apply softmax to the logits to get the probabilities
        int numClasses = predicted.Length;
        float maxLogit = predicted[0];
        float sumExp = 0f;

        // Find max logit to improve numerical stability
        for (int i = 1; i < numClasses; i++)
        {
            if (predicted[i] > maxLogit) maxLogit = predicted[i];
        }

        // Compute the softmax values (numerically stable)
        for (int i = 0; i < numClasses; i++)
        {
            sumExp += MathF.Exp(predicted[i] - maxLogit);
        }

        // Softmax and compute the log of probabilities
        float logProb = 0f;
        for (int i = 0; i < numClasses; i++)
        {
            float prob = MathF.Exp(predicted[i] - maxLogit) / sumExp;
            if (truth[i] == 1f)
            {
                logProb = MathF.Log(prob); // Only compute log for the true class
                break;
            }
        }

        // Step 2: Return the negative log-likelihood for the true class
        return -logProb;
    }

    /// <summary>
    /// Computes the gradient of the Categorical Cross-Entropy loss w.r.t the logits.
    /// </summary>
    /// <param name="gradient">Gradient will be stored in this span</param>
    /// <param name="predicted">Predicted logits (raw network outputs)</param>
    /// <param name="truth">True labels in one-hot encoding</param>
    public override void Gradient(Span<float> gradient, ReadOnlySpan<float> predicted, ReadOnlySpan<float> truth)
    {
        int numClasses = predicted.Length;

        // Step 1: Apply softmax to the logits to get the probabilities
        float maxLogit = predicted[0];
        float sumExp = 0f;

        // Find max logit to improve numerical stability
        for (int i = 1; i < numClasses; i++)
        {
            if (predicted[i] > maxLogit) maxLogit = predicted[i];
        }

        // Compute the softmax values (numerically stable)
        for (int i = 0; i < numClasses; i++)
        {
            sumExp += MathF.Exp(predicted[i] - maxLogit);
        }

        // Compute softmax probabilities
        for (int i = 0; i < numClasses; i++)
        {
            gradient[i] = MathF.Exp(predicted[i] - maxLogit) / sumExp;
        }

        // Step 2: Subtract the truth vector (one-hot encoded) from the probabilities
        for (int i = 0; i < numClasses; i++)
        {
            gradient[i] -= truth[i];
        }
    }
}