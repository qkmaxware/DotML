using System.ComponentModel;
using System.Reflection;
using DotML.NetDot.Dot;
using DotML.Network;

namespace DotML.NetDot;

// Basic pattern for convertinging nodes representing single (standalone) layers into modules IE dense-linear or conv2d
public class LayerPattern<T>
: IModulePattern
where T:INetworkModule
{
    private string type;
    private ConstructorInfo[] cons;

    public LayerPattern()
    {
        var type = typeof(T);
        this.type = type.Name;

        this.cons = type.GetConstructors(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
    }

    public bool TryReplacePattern(DotGraph graph)
    {
        for (var nodeIndex = 0; nodeIndex < graph.VertexCount; nodeIndex++)
        {
            var node = graph.GetVertex(nodeIndex);
            if (node.IsInput() || node.IsOutput())
                continue;
            var attrs = node.Attributes;

            if (!attrs.TryGetValue("type", out var type_name) || type_name != this.type)
            {
                continue;
            }

            foreach (var con in cons)
            {
                bool useCon = true;
                var param = con.GetParameters();
                foreach (var p in param)
                {
                    if (p.Name is null)
                        continue;
                    if (attrs.ContainsKey(p.Name))
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
                    string s = attrs[p.Name ?? string.Empty];
                    objs[i] = ChangeTypeToParameter(s, p);
                }
                var mod = con.Invoke(objs) as INetworkModule;
                if (mod is null)
                    continue;

                graph.Replace(node, new CompiledModuleNode(mod));
                return true;
            }
        }

        return false;
    }
	
	private static object? ChangeTypeToParameter(string value, ParameterInfo param)
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
}
