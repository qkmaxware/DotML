namespace DotML.Network.IO.Netbuild;

internal class ActivationFunctionMapper {

    private static Type[] Empty = [];
    private static object[] NoArgs = new object[0];
    private static Type[] Alpha = [typeof(double)];

    public ActivationFunction Decode(string name, float alpha) => DecodeStatic(name, alpha);

    public static ActivationFunction DecodeStatic(string name, float alpha)
    {
        // General strategy
        foreach (var function in ActivationFunctions.EnumerateAll())
        {
            var func_type = function.GetType();
            var func_name = func_type.Name;
            if (func_name.Contains(name, StringComparison.CurrentCultureIgnoreCase))
            {
                ActivationFunction? instance = null;
                var parameter_constructor = func_type.GetConstructor(System.Reflection.BindingFlags.Public, Alpha);
                if (parameter_constructor is not null)
                {
                    instance = (ActivationFunction?)Activator.CreateInstance(func_type, [alpha]);
                }
                else
                {
                    var default_constructor = func_type.GetConstructor(System.Reflection.BindingFlags.Public, Empty);
                    if (default_constructor is not null)
                    {
                        instance = (ActivationFunction?)Activator.CreateInstance(func_type, NoArgs);
                    }
                }
                if (instance is not null)
                    return instance;
            }
        }

        // Nickname strategy
        return name.ToLower() switch
        {
            "step" => BinaryStep.Instance,
            "binarystep" => BinaryStep.Instance,
            "elu" => new ExponentialLU(alpha),
            "exponentiallu" => new ExponentialLU(alpha),
            "tanh" => HyperbolicTangent.Instance,
            "hyperbolictangent" => HyperbolicTangent.Instance,
            "id" => IdentityFunction.Instance,
            "identity" => IdentityFunction.Instance,
            "leaky-relu" => LeakyReLU.Instance,
            "leakyrelu" => LeakyReLU.Instance,
            "prelu" => new PReLU(alpha),
            "relu" => ReLU.Instance,
            "sigmoid" => Sigmoid.Instance,
            "telu" => TeLU.Instance,
            _ => throw new ArgumentException($"Unknown activation function {name}")
        };
    }
}