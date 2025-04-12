namespace DotML.Network;

[WorkInProgress]
public class ResidualConnection : AdditionSkipConnection {
    public ResidualConnection(Shape3D input_shape, InputCapture captureSource) : base(input_shape, captureSource) { }

    public override void Visit(ILayerVisitor visitor) {
        throw new NotImplementedException();
    }

    public override void Visit<TIn>(ILayerInputVisitor<TIn> visitor, TIn args) {
        throw new NotImplementedException();
    }

    public override T Visit<T>(ILayerOutputVisitor<T> visitor) {
        throw new NotImplementedException();
    }

    public override TOut Visit<TIn, TOut>(ILayerInputOutputVisitor<TIn, TOut> visitor, TIn args) {
        throw new NotImplementedException();
    }
}