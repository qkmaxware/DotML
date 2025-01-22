namespace DotML.Network;

[WorkInProgress]
public abstract class ResidualConnection : AdditionSkipConnection {
    public ResidualConnection(InputCapture captureSource) : base(captureSource) { }  

    // Combine behavior is the same as in the default AdditionSkipConnection
}