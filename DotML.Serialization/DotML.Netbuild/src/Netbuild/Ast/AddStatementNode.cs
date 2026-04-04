using System.ComponentModel;
using System.Reflection;
using DotML.Network.Embedding.Text;

namespace DotML.Network.IO.Netbuild;

public class AddStatement : Statement {
    string layer_name;
    string? ident;
    public Dictionary<string, Literal> arguments = new Dictionary<string, Literal>();

    public AddStatement(string layer_name,List<(Token<string>, Literal)> args, string? alias) {
        this.layer_name = layer_name;
        foreach (var pair in args) {
            arguments[pair.Item1.Value] = pair.Item2;
        } 
        this.ident = alias;
    }

    public override void ModuleAction(BuildEnvironment env)
    {
        var network = env.NetworkBlock;
        if (network is null)
            return;

        var output_shape = network.LayerCount > 0 ? network.ForwardShape(env.InputShape) : env.InputShape;
        INetworkModule? module = makeLayer(this.layer_name, output_shape, new ArgumentMap(env, arguments));
        if (module is null)
            return;

        var index = network.LayerCount;
        network.Add(module);
        
        if (!string.IsNullOrEmpty(ident))
        {
            env.LayerAliases[ident] = index;
        }
    }

    internal static INetworkModule? makeLayer(string layer_name, Shape ishape, ArgumentMap args)
    {
        return layer_name switch
        {
            nameof(DenseLinear) => makeLayer<DenseLinear>(ishape, args),
            nameof(Conv2D) => makeLayer<Conv2D>(ishape, args),
            nameof(TransposeConv2D) => makeLayer<TransposeConv2D>(ishape, args),
            nameof(Flatten) => makeLayer<Flatten>(ishape, args),
            nameof(Activation) => makeLayer<Activation>(ishape, args),
            nameof(MaxPool2D) => makeLayer<MaxPool2D>(ishape, args),
            nameof(MinPool2D) => makeLayer<MinPool2D>(ishape, args),
            nameof(AvgPool2D) => makeLayer<AvgPool2D>(ishape, args),
            nameof(Dropout) => makeLayer<Dropout>(ishape, args),
            nameof(SoftmaxOutput) => makeLayer<SoftmaxOutput>(ishape, args),
            nameof(SelfAttention) => makeLayer<SelfAttention>(ishape, args),
            nameof(LearnedEmbedding) => makeLayer<LearnedEmbedding>(ishape, args),
            _ => throw new FormatException("Layer type '{layer_name}' is not supported")
        };
    }

    private static INetworkModule? makeLayer<T>(Shape ishape, ArgumentMap args)
    where T : INetworkModule
    {
        var cons = typeof(T).GetConstructors();

        var alpha = 0.0f;
        if (args.ContainsKey("alpha"))
        {
            alpha = args["alpha"].AsFloat();
        }

        foreach (var con in cons)
        {
            bool useCon = true;
            var param = con.GetParameters();
            foreach (var p in param)
            {
                if (p.Name is null)
                    continue;
                if (!args.ContainsKey(p.Name))
                {
                    useCon = false;
                    break;
                }
            }

            if (!useCon)
            {
                continue;
            }

            var objs = new object?[param.Length];
            for (var i = 0; i < objs.Length; i++)
            {
                ParameterInfo p = param[i];
                string s = args[p.Name ?? string.Empty].AsString();
                objs[i] = ChangeTypeToParameter(s, p, alpha);
            }
            var mod = con.Invoke(objs) as INetworkModule;
            if (mod is null)
                continue;

            return mod;
        }
        return null;
    }

    private static object? ChangeTypeToParameter(string value, ParameterInfo param, float alpha = 0.0f)
    {
        var targetType = param.ParameterType;
        
        // For enums first try to match off case-insensitive parsing
        if (targetType.IsEnum)
        {
            if (Enum.TryParse(targetType, value, ignoreCase: true, out var parsed))
            {
                return parsed;
            }
        }

        // If the target is a activation function, handle that specially
        if (targetType == typeof(ActivationFunction))
        {
            ActivationFunctionMapper.DecodeStatic(value, alpha);
        }

        // If the target type implements IParsable<T>, use that
        if (targetType.IsAssignableTo(typeof(IParsable<>)))
        {
            var parseMethod = targetType.GetMethod("Parse", BindingFlags.Public | BindingFlags.Static, null, new Type[] { typeof(string), typeof(IFormatProvider) }, null);
            if (parseMethod is not null)
            {
                var obj = parseMethod.Invoke(null, new object?[] { value, null });
                if (obj is not null && obj.GetType().IsAssignableTo(targetType))
                {
                    return obj;
                }
            }
        }

        // TODO handle Index types

        // Get the type converter
        var converter = TypeDescriptor.GetConverter(targetType);
        if (converter is null || !converter.CanConvertFrom(typeof(string)))
        {
            if (param.HasDefaultValue)
                return param.DefaultValue; // If the constructor provided a default value, just use that
            else
                throw new ArgumentException($"Value cannot be converted to object of type {targetType}", nameof(value));
        }

        // Convert
        var def = param.HasDefaultValue ? param.DefaultValue : null;
        var converted = converter.ConvertFromInvariantString(value);
        return converted ?? def;
	}

    public override string ToString()
    {
        if (ident is not null)
        {
            return $"ADD {layer_name} {string.Join(' ', arguments.Select(kv => $"{kv.Key}={kv.Value}"))} AS {ident}";
        }
        else
        {
            return $"ADD {layer_name} {string.Join(' ', arguments.Select(kv => $"{kv.Key}={kv.Value}"))}";
        }
    }
}