namespace DotML.Network.Training;

/// <summary>
/// Regularization function used in network learning
/// </summary>
public abstract class RegularizationFunction {
    /// <summary>
    /// Invoke the activation function with the given input
    /// </summary>
    /// <param name="x">function input</param>
    /// <returns>function result</returns>
    public abstract double Invoke(double x);    
}

/// <summary>
/// Container for various regularization functions
/// </summary>
public static class Regularization {
    /// <summary>
    /// Enumerate over all learning rate optimizers
    /// </summary>
    /// <returns>enumerable of learning rate optimizers</returns>
    public static IEnumerable<RegularizationFunction> EnumerateAll() {
        return typeof(Regularization)
            .GetProperties(System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public)
            .Where(prop => prop.CanRead && prop.PropertyType.IsAssignableTo(typeof(RegularizationFunction)))
            .Select(prop => prop.GetValue(null))
            .OfType<RegularizationFunction>();
    }

    /// <summary>
    /// No regularization
    /// </summary>
    public static RegularizationFunction None {get; private set;} = new NoRegularization();

    /// <summary>
    /// L1 regularization
    /// </summary>
    public static RegularizationFunction L1 {get; private set;} = new L1Regularization();

    /// <summary>
    /// L2 regularization
    /// </summary>
    public static RegularizationFunction L2 {get; private set;} = new L2Regularization();
}

/// <summary>
/// No Regularization function
/// </summary>
public class NoRegularization : RegularizationFunction {

    public NoRegularization() { }

    public override double Invoke(double x) => 0.0d;

    //public double InvokeDerivative(double y) {
        //return y < 0 ? -1 : (y > 0 ? 1 : 0);
    //}
}

/// <summary>
/// L1 Regularization function
/// </summary>
public class L1Regularization : RegularizationFunction {

    public double Hyperparameter {get; set;}

    public L1Regularization(double lambda = 0.01) {
        this.Hyperparameter = lambda;
    }

    public override double Invoke(double x) {
        return Hyperparameter * Math.Abs(x);
    }

    //public double InvokeDerivative(double y) {
        //return y < 0 ? -1 : (y > 0 ? 1 : 0);
    //}
}

/// <summary>
/// L2 Regularization function
/// </summary>
public class L2Regularization : RegularizationFunction {

    public double Hyperparameter {get; set;}

    public L2Regularization(double lambda = 0.01) {
        this.Hyperparameter = lambda;
    }

    public override double Invoke(double x) {
        return Hyperparameter * x * x;
    }

    //public double InvokeDerivative(double y) {
        //return y;
    //}
}