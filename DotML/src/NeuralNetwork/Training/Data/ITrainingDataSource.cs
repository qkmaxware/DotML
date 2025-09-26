using System.Numerics;

namespace DotML.Network.Training;

public interface ITrainingDataSource<TType> 
: IEnumerable<(Tensor<TType> Input, Tensor<TType> Output)>
where TType:INumber<TType>
{
    public int Count { get; }
    public TensorShape InputShape { get; }
    public TensorShape OutputShape { get; }
    public (Tensor<TType> Input, Tensor<TType> Output) this[int index] {get;}
}

public class ListTrainingDataSource<TType> 
: List<(Tensor<TType> Input, Tensor<TType> Output)>, ITrainingDataSource<TType> 
where TType:INumber<TType>
{
    public TensorShape InputShape { get; init; }
    public TensorShape OutputShape { get; init; }
    
    public ListTrainingDataSource(TensorShape ishape, TensorShape oshape) {
        this.InputShape = ishape;
        this.OutputShape = oshape;
        
    }
}