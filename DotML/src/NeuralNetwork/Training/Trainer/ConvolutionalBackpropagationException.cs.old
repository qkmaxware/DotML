namespace DotML.Network.Training;

/// <summary>
/// Exception thrown when an error happens during backpropagation training for a CNN
/// </summary>
public class ConvolutionalBackpropagationException : System.ArithmeticException {

    /// <summary>
    /// The network being trained
    /// </summary>
    public FeedforwardNetwork Network {get; init;} 

    public ConvolutionalBackpropagationException(FeedforwardNetwork network, Exception inner) : base("An exception ocurred during backpropagation training", inner) {
        this.Network = network;
    }

    public Safetensors DumpMatrices() => Network.ToSafetensor();

    public void DumpMatrices(BinaryWriter writer) => Network.ToSafetensor(writer);
}