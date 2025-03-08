namespace DotML.Network.Initialization;

/// <summary>
/// Interface describing a network initialization strategy
/// </summary>
public interface IInitializer {
    /// <summary>
    /// Generate a random weight for a given neuron
    /// </summary>
    /// <param name="input_count">Number of input neurons</param>
    /// <param name="output_count">Number of output neurons</param>
    /// <param name="parameterCount">Total number of parameters in the layer</param>
    /// <returns>Random weight</returns>
    public double RandomWeight(int input_count, int output_count, int parameterCount);

    /// <summary>
    /// Generate a random bias for a given neuron
    /// </summary>
    /// <param name="input_count">Number of input neurons</param>
    /// <param name="output_count">Number of output neurons</param>
    /// <param name="parameterCount">Total number of parameters in the layer</param>
    /// <returns>Random bias</returns>
    public double RandomBias(int input_count, int output_count, int parameterCount);
}

/// <summary>
/// Container storing various initialization strategies
/// </summary>
public static class Initializers {
    /// <summary>
    /// Enumerate over all initialization strategies
    /// </summary>
    /// <returns>enumerable of initialization strategies</returns>
    public static IEnumerable<IInitializer> EnumerateAll() {
        return typeof(Initializers)
            .GetProperties(System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public)
            .Where(prop => prop.CanRead && prop.PropertyType.IsAssignableTo(typeof(IInitializer)))
            .Select(prop => prop.GetValue(null))
            .OfType<IInitializer>();
    }

    /// <summary>
    /// Initialize everything to zero
    /// </summary>
    public static IInitializer Zero {get; private set;} = new ConstantInitialization(0);

    /// <summary>
    /// Initialize everything to one
    /// </summary>
    public static IInitializer One {get; private set;} = new ConstantInitialization(1);

    /// <summary>
    /// Initialize everything to a random value between -1 and 1
    /// </summary>
    public static IInitializer Random {get; private set;} = new RandomInitialization(-1.0, 1.0);

    /// <summary>
    /// Initialize everything using Xavier or Glorot with a normal distribution
    /// </summary>
    public static IInitializer NormalXavier {get; private set;} = new NormalXavierInitialization();

    /// <summary>
    /// Initialize everything using Xavier or Glorot with a uniform distribution
    /// </summary>
    public static IInitializer UniformXavier {get; private set;} = new UniformXavierInitialization();

    /// <summary>
    /// Initialize everything using He initialization
    /// </summary>
    public static IInitializer He {get; private set;} = new HeInitialization();

    /// <summary>
    /// Initialize everything using LeCun initialization
    /// </summary>
    public static IInitializer LeCun {get; private set;} = new LeCunInitialization();
}