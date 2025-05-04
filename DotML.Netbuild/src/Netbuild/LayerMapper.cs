using System.Data;
using System.Reflection;

namespace DotML.Network.IO.Netbuild;

public class ArgumentMap : System.Collections.Generic.Dictionary<string, DotML.Network.IO.Netbuild.Literal> {
    public BuildEnvironment Env {get; private set;}
    public ArgumentMap(BuildEnvironment environment) {
        this.Env = environment;
    }
    public ArgumentMap(BuildEnvironment environment, System.Collections.Generic.Dictionary<string, DotML.Network.IO.Netbuild.Literal> args) : base(args) {
        this.Env = environment;
    }
}

public static class ArgumentMapExtensions {
    public static Literal FirstOf(this ArgumentMap mapping, Literal @default, params string[] names) {
        foreach (var token in names) {
            if (mapping.TryGetValue(token, out var literal)) {
                return literal;
            }
        }

        return @default;
    }
}

public class LayerMapper : ILayerInputOutputVisitor<LayerMapper.LayerConstructionArgs,IFeedforwardNetworkLayer> {

    private List<MethodInfo> decoders;

    private ActivationFunctionMapper activations;

    private static Literal One = new ObjectLiteral(1);
    private static Literal Zero = new ObjectLiteral(0);
    private static Literal Same = new ObjectLiteral("same");

    public LayerMapper(ActivationFunctionMapper activations) {
        this.activations = activations;
        this.decoders = typeof(LayerMapper)
            .GetMethods(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)
            .Where(m => IsCompatibleMethod(m))
            .ToList();
    }

    public class LayerConstructionArgs {
        public Shape3D InputShape;
        public ArgumentMap Arguments;

        public LayerConstructionArgs(Shape3D ishape, ArgumentMap args) {
            this.InputShape = ishape;
            this.Arguments = args;
        }
    }

    private static bool IsCompatibleMethod(MethodInfo info) {
        if (info.IsAbstract)
            return false;                                                           // Method must be implemented and callable
        
        if (!info.ReturnType.IsAssignableTo(typeof(IFeedforwardNetworkLayer)))      
            return false;                                                           // Check that the return type matches the return type for the visitor pattern

        var param = info.GetParameters();
        if (param.Length != 2)
            return false;                                                           // Check that this has the correct argument count for the visitor pattern

        return 
            param[0].ParameterType.IsAssignableTo(typeof(IFeedforwardNetworkLayer)) // First arg is a valid layer
         && param[1].ParameterType.IsAssignableTo(typeof(LayerConstructionArgs));   // Second arg is my custom constructor parameters
    }

    public IFeedforwardNetworkLayer Decode(int position, string name, Shape3D input_shape, ArgumentMap arguments) {
        var decoder = decoders
            .Where(m => m.Name.Contains(name, StringComparison.CurrentCultureIgnoreCase) || m.GetParameters()[0].ParameterType.Name.Contains(name, StringComparison.CurrentCultureIgnoreCase))
            .FirstOrDefault();
        if (decoder is null) {
            throw new SyntaxErrorException($"Unknown layer type {name} at position {position}");
        }
        var layer = (IFeedforwardNetworkLayer?)decoder.Invoke(this, [ 
            null,                                               // No layer used as we are trying to create one from scratch; we are just exploiting the visitor interface here for type safety
            new LayerConstructionArgs(input_shape, arguments)   // The arguments that are actually used to create the new layer
        ]);
        if (layer is null) {
            // Should never happen but ... type safety
            throw new SyntaxErrorException($"Unknown layer type {name} at position {position}");
        }

        return layer;
    }

    public IFeedforwardNetworkLayer Visit(ConvolutionLayer layer, LayerConstructionArgs args) {
        (Shape3D ishape, ArgumentMap arguments) = (args.InputShape, args.Arguments);
        int? x_pad      = arguments.ContainsKey("padding-x") ? arguments["padding-x"].AsInt() : null;
        int? y_pad      = arguments.ContainsKey("padding-y") ? arguments["padding-y"].AsInt() : null;
        Padding pad     = Enum.Parse<Padding>(arguments.FirstOf(Same, "padding").AsString(), true);
        var x_stride    = arguments.FirstOf(One, "stride-x", "stride").AsInt();
        var y_stride    = arguments.FirstOf(One, "stride-y", "stride").AsInt();
        var filters     = arguments["filters"].AsInt();
        var kernel_size = arguments["kernel"].AsInt();

        if (x_pad.HasValue && y_pad.HasValue) {
            return new ConvolutionLayer(
                input_size: ishape,
                rowsPadding: y_pad.Value,
                columnsPadding: x_pad.Value,
                strideX: x_stride,
                strideY: y_stride,
                filters: ConvolutionFilter.Make(
                    filters: filters,
                    kernels_per_filter: ishape.Channels,
                    kernel_size: kernel_size
                ) 
            );
        } else {
            return new ConvolutionLayer(
                input_size: ishape,
                padding: pad,
                strideX: x_stride,
                strideY: y_stride,
                filters: ConvolutionFilter.Make(
                    filters: filters,
                    kernels_per_filter: ishape.Channels,
                    kernel_size: kernel_size
                ) 
            );
        }
    }

    public IFeedforwardNetworkLayer Visit(DepthwiseConvolutionLayer? layer, LayerConstructionArgs args) {
        (Shape3D ishape, ArgumentMap arguments) = (args.InputShape, args.Arguments);
        var padding     = Enum.Parse<Padding>(arguments.FirstOf(Same, "padding").AsString(), true);
        var x_stride    = arguments.FirstOf(One, "stride-x", "stride").AsInt();
        var y_stride    = arguments.FirstOf(One, "stride-y", "stride").AsInt();
        var kernel_size = arguments["kernel"].AsInt();

        return new DepthwiseConvolutionLayer(
            input_size: ishape,
            padding: padding,
            strideX: x_stride,
            strideY: y_stride,
            filter: ConvolutionFilter.Make(
                filters: ishape.Channels,
                kernels_per_filter: 1,
                kernel_size: kernel_size
            ) 
        );
    }

    public IFeedforwardNetworkLayer Visit(TransposeConvolutionLayer layer, LayerConstructionArgs args) {
        (Shape3D ishape, ArgumentMap arguments) = (args.InputShape, args.Arguments);
        int? x_pad      = arguments.ContainsKey("padding-x") ? arguments["padding-x"].AsInt() : null;
        int? y_pad      = arguments.ContainsKey("padding-y") ? arguments["padding-y"].AsInt() : null;
        int? x_expand   = arguments.ContainsKey("expand-x") ? arguments["expand-x"].AsInt() : null;
        int? y_expand   = arguments.ContainsKey("expand-y") ? arguments["expand-y"].AsInt() : null;
        Padding pad     = Enum.Parse<Padding>(arguments.FirstOf(Same, "padding").AsString(), true);
        Expansion expand= Enum.Parse<Expansion>(arguments.FirstOf(Same, "expand").AsString(), true);
        var x_stride    = arguments.FirstOf(One, "stride-x", "stride").AsInt();
        var y_stride    = arguments.FirstOf(One, "stride-y", "stride").AsInt();
        var filters     = arguments["filters"].AsInt(); // Not really filters, but outputs
        var kernel_size = arguments["kernel"].AsInt();

        if (x_pad.HasValue && y_pad.HasValue && x_expand.HasValue && y_expand.HasValue) {
            return new TransposeConvolutionLayer(
                input_size: ishape,
                inputPaddingX: x_pad.Value, inputPaddingY: x_pad.Value,
                outputPaddingX: x_expand.Value, outputPaddingY: y_expand.Value,
                strideX: x_stride,
                strideY: y_stride,
                filters: ConvolutionFilter.Make(
                    filters: filters,
                    kernels_per_filter: ishape.Channels,
                    kernel_size: kernel_size
                ) 
            );
        } else {
            return new TransposeConvolutionLayer(
                input_size: ishape,
                padding: pad,
                expansion: expand,
                strideX: x_stride,
                strideY: y_stride,
                filters: ConvolutionFilter.Make(
                    filters: filters,
                    kernels_per_filter: ishape.Channels,
                    kernel_size: kernel_size
                ) 
            );
        }
    }

    public IFeedforwardNetworkLayer Visit(PixelShuffle? layer, LayerConstructionArgs args) {
        (Shape3D ishape, ArgumentMap arguments) = (args.InputShape, args.Arguments);
        var upscale = arguments.FirstOf(One, "upscale", "upscale-by").AsInt();

        return new PixelShuffle(
            input_size: ishape,
            upscale_factor: upscale
        );
    }

    public IFeedforwardNetworkLayer Visit(PoolingLayer? layer, LayerConstructionArgs args) {
        // Don't use this. Use the LocalMaxPoolingLayer and LocalAvgPoolingLayer versions below
        // I think I want to fix this in the visitor pattern anyways 
        // This acts as a fallback method
        throw new NotImplementedException();
    }

    public IFeedforwardNetworkLayer Visit(LocalMaxPoolingLayer? layer, LayerConstructionArgs args) {
        (Shape3D ishape, ArgumentMap arguments) = (args.InputShape, args.Arguments);
        var x_stride = arguments.FirstOf(One, "stride-x", "stride").AsInt();
        var y_stride = arguments.FirstOf(One, "stride-y", "stride").AsInt();
        var x_pad = arguments.FirstOf(Zero, "padding-x", "padding").AsInt();
        var y_pad = arguments.FirstOf(Zero, "padding-y", "padding").AsInt();
        var kernel = arguments["kernel"].AsInt();

        return new LocalMaxPoolingLayer(
            input_size: ishape,
            width: kernel,
            height: kernel,
            strideX: x_stride,
            strideY: y_stride,
            paddingX: x_pad,
            paddingY: y_pad
        );
    }

    public IFeedforwardNetworkLayer Visit(LocalAvgPoolingLayer? layer, LayerConstructionArgs args) {
        (Shape3D ishape, ArgumentMap arguments) = (args.InputShape, args.Arguments);
        var x_stride = arguments.FirstOf(One, "stride-x", "stride").AsInt();
        var y_stride = arguments.FirstOf(One, "stride-y", "stride").AsInt();
        var x_pad = arguments.FirstOf(Zero, "padding-x", "padding").AsInt();
        var y_pad = arguments.FirstOf(Zero, "padding-y", "padding").AsInt();
        var kernel = arguments["kernel"].AsInt();

        return new LocalAvgPoolingLayer(
            input_size: ishape,
            width: kernel,
            height: kernel,
            strideX: x_stride,
            strideY: y_stride,
            paddingX: x_pad,
            paddingY: y_pad
        );
    }

    public IFeedforwardNetworkLayer Visit(FlatteningLayer? layer, LayerConstructionArgs args) {
        (Shape3D ishape, _) = (args.InputShape, args.Arguments);
        return new FlatteningLayer(ishape);
    }

    public IFeedforwardNetworkLayer Visit(DropoutLayer? layer, LayerConstructionArgs args) {
        (Shape3D ishape, ArgumentMap argument) = (args.InputShape, args.Arguments);
        var dropout = argument["percent"].AsDouble();

        return new DropoutLayer(ishape, dropout);
    }

    public IFeedforwardNetworkLayer Visit(LayerNorm? layer, LayerConstructionArgs args) {
        (Shape3D ishape, _) = (args.InputShape, args.Arguments);
        return new LayerNorm(ishape);
    }

    public IFeedforwardNetworkLayer Visit(BatchNorm? layer, LayerConstructionArgs args) {
        (Shape3D ishape, _) = (args.InputShape, args.Arguments);
        return new BatchNorm(ishape);
    }

    public IFeedforwardNetworkLayer Visit(DenseLinearLayer? layer, LayerConstructionArgs args) {
        (Shape3D ishape, ArgumentMap argument) = (args.InputShape, args.Arguments);
        var neurons = argument["neurons"].AsInt();

        return new DenseLinearLayer(
            input_size: ishape.Count,
            neurons: neurons
        );
    }

    public IFeedforwardNetworkLayer Visit(ActivationLayer? layer, LayerConstructionArgs args) {
        (Shape3D ishape, ArgumentMap arguments) = (args.InputShape, args.Arguments);
        var activation = activations.Decode(
            arguments["fn"].AsString().ToLower(),
            arguments.ContainsKey("alpha") ? arguments["alpha"].AsDouble() : 0.0
        );

        return new ActivationLayer(ishape, activation);
    }

    public IFeedforwardNetworkLayer Visit(SoftmaxLayer? layer, LayerConstructionArgs args) {
        (Shape3D ishape, _) = (args.InputShape, args.Arguments);
        return new SoftmaxLayer(ishape.Count);
    }

    public IFeedforwardNetworkLayer Visit(InputCapture? capture, LayerConstructionArgs args) {
        (Shape3D ishape, _) = (args.InputShape, args.Arguments);
        return new InputCapture(ishape);
    }

    public IFeedforwardNetworkLayer Visit(AdditionSkipConnection? skip, LayerConstructionArgs args) {
        (Shape3D ishape, ArgumentMap arguments) = (args.InputShape, args.Arguments);
        var layer_name = arguments["residual"].AsString();
        InputCapture? capture = (InputCapture?)arguments.Env.GetLayer(layer_name);
        if (capture is null)
            throw new ArgumentException($"Layer '{layer_name}' either doesn't exist or is not an input capturing layer.");
        return new AdditionSkipConnection(ishape, capture);
    }
}