namespace DotML.Network.Initialization;

/// <summary>
/// An initializer which initializes everything to a constant value
/// </summary>
public class ConstantInitialization: IInitializer {
    private float constant;

    public ConstantInitialization(float constant) {
        this.constant = constant;
    }

    public float RandomBias(int input_count, int output_count, int parameterCount) {
        return constant;
    }

    public float RandomWeight(int input_count, int output_count, int parameterCount) {
        return constant;
    }
}

/// <summary>
/// An initializer which initializes everything to 0
/// </summary>
/// <typeparam name="TNetwork">Network type</typeparam>
public class ZeroInitialization: ConstantInitialization {
    
    public ZeroInitialization(): base(0) { }

}