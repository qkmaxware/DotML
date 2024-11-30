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