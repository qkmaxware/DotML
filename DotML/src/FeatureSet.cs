using System.Collections;
using System.Numerics;

namespace DotML;

/// <summary>
/// Represents a batch of multiple feature sets
/// </summary>
/// <typeparam name="T">feature value type</typeparam>
public class BatchedFeatureSet<T> :
    IEnumerable<FeatureSet<T>>
    where T:INumber<T>,IExponentialFunctions<T>,IRootFunctions<T> 
{
    private FeatureSet<T>[] batches;

    public BatchedFeatureSet() {
        batches = Array.Empty<FeatureSet<T>>();
    }

    public BatchedFeatureSet(BatchedFeatureSet<T> other) {
        this.batches = (FeatureSet<T>[])other.batches.Clone();
    }

    public BatchedFeatureSet(params FeatureSet<T>[] batches) {
        this.batches = batches;
    }

    /// <summary>
    /// Fetch a given feature by channel index
    /// </summary>
    /// <param name="batch">batch index</param>
    /// <returns>feature set</returns>
    public FeatureSet<T> this[int batch] => batches[batch];

    /// <summary>
    /// Fetch a feature matrix
    /// </summary>
    /// <param name="batch">batch index</param>
    /// <param name="channel">channel index</param>
    /// <returns>feature</returns>
    public Matrix<T> this[int batch, int channel] => batches[batch][channel];

    /// <summary>
    /// Fetch a feature value
    /// </summary>
    /// <param name="batch">batch index</param>
    /// <param name="channel">channel index</param>
    /// <param name="row">row index</param>
    /// <param name="col">column index</param>
    /// <returns>feature</returns>
    public T this[int batch, int channel, int row, int col] => batches[batch][channel][row, col];

    /// <summary>
    /// Number of batches of feature sets
    /// </summary>
    public int Batches => batches.Length;

    /// <summary>
    /// Number of channels in the feature set
    /// </summary>
    public int Channels => batches.Length > 0 ? batches[0].Channels : 0;

    /// <summary>
    /// Number of rows in the feature set
    /// </summary>
    public int Rows => batches.Length > 0 ? batches[0].Rows : 0;

    /// <summary>
    /// Number of columns in the feature set
    /// </summary>
    public int Columns => batches.Length > 0 ? batches[0].Columns : 0;

    /// <summary>
    /// Shape of the feature set
    /// </summary>
    public Shape4D Shape => new Shape4D(Batches, Channels, Rows, Columns);

    public IEnumerator<FeatureSet<T>> GetEnumerator() => ((IEnumerable<FeatureSet<T>>)batches).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => batches.GetEnumerator();
}

/// <summary>
/// Feature set (basically a wrapper for a matrix array with additional semantics)
/// </summary>
/// <typeparam name="T">feature value type</typeparam>
public class FeatureSet<T> :  
    IEnumerable<Matrix<T>>
    where T:INumber<T>,IExponentialFunctions<T>,IRootFunctions<T> 
{
    private Matrix<T>[] channels; // The feature channels

    #region Constructors

    /// <summary>
    /// Create an empty feature set
    /// </summary>
    public FeatureSet() {
        this.channels = Array.Empty<Matrix<T>>();
    }

    /// <summary>
    /// Create a copy of a feature set
    /// </summary>
    /// <param name="other">feature set to copy</param>
    public FeatureSet(FeatureSet<T> other) {
        this.channels = (Matrix<T>[])other.channels.Clone();
    }

    /// <summary>
    /// Create a feature set from the given channels
    /// </summary>
    /// <param name="channels">feature channels</param>
    public FeatureSet(params Matrix<T>[] channels) {
        this.channels = channels;
    }

    #endregion

    #region Shape

    /// <summary>
    /// Number of channels in the feature set
    /// </summary>
    public int Channels => channels.Length;

    /// <summary>
    /// Number of rows in the feature set
    /// </summary>
    public int Rows => channels.Length > 0 ? channels[0].Rows : 0;

    /// <summary>
    /// Number of columns in the feature set
    /// </summary>
    public int Columns => channels.Length > 0 ? channels[0].Columns : 0;

    /// <summary>
    /// Shape of the feature set
    /// </summary>
    public Shape3D Shape => new Shape3D(Channels, Rows, Columns);

    #endregion

    #region Indexers

    /// <summary>
    /// Fetch a given feature by channel index
    /// </summary>
    /// <param name="channel">channel index</param>
    /// <returns>feature matrix</returns>
    public Matrix<T> this[int channel] => channels[channel];

    /// <summary>
    /// Fetch a feature value
    /// </summary>
    /// <param name="channel">channel index</param>
    /// <param name="row">row index</param>
    /// <param name="col">column index</param>
    /// <returns>feature value</returns>
    public T this[int channel, int row, int col] => channels[channel][row, col];

    #endregion

    #region Channel Shortcuts

    /// <summary>
    /// The X channel (channel 0)
    /// </summary>
    public Matrix<T> X => channels.ElementAt(0);
    /// <summary>
    /// The Y channel (channel 1)
    /// </summary>
    public Matrix<T> Y => channels.ElementAt(1);
    /// <summary>
    /// The Z channel (channel 2)
    /// </summary>
    public Matrix<T> Z => channels.ElementAt(2);

    /// <summary>
    /// The Red image channel (channel 0)
    /// </summary>
    public Matrix<T> Red => channels.ElementAt(0);
    /// <summary>
    /// The Green image channel (channel 0)
    /// </summary>
    public Matrix<T> Green => channels.ElementAt(1);
    /// <summary>
    /// The Blue image channel (channel 0)
    /// </summary>
    public Matrix<T> Blue => channels.ElementAt(2);
    /// <summary>
    /// The Alpha image channel (channel 0)
    /// </summary>
    public Matrix<T> Alpha => channels.ElementAt(2);

    #endregion

    /// <summary>
    /// Concatenate additional feature sets to create a new feature set
    /// </summary>
    /// <param name="other">feature set to concatenate</param>
    /// <returns>new feature set containing the features of both</returns>
    public FeatureSet<T> Concat(FeatureSet<T> other) {
        Matrix<T>[] results = new Matrix<T>[this.channels.Length + other.channels.Length];
        this.channels.CopyTo(results, 0);
        other.channels.CopyTo(results, this.channels.Length);
        return new FeatureSet<T>(results);
    }

    public IEnumerator<Matrix<T>> GetEnumerator() => ((IEnumerable<Matrix<T>>)channels).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => channels.GetEnumerator();

    /// <summary>
    /// Convert a matrix array to feature set
    /// </summary>
    /// <param name="channels">array of feature channels</param>
    public static explicit operator FeatureSet<T>(Matrix<T>[] channels) => new FeatureSet<T>(channels);

    /// <summary>
    /// Convert a feature set to a matrix array
    /// </summary>
    /// <param name="features">feature set</param>
    public static explicit operator Matrix<T>[](FeatureSet<T> features) => features.channels;

    /// <summary>
    /// Fetch the underlying matrix array interpretation of the data
    /// </summary>
    /// <returns>matrix array</returns>
    public Matrix<T>[] AsArray() => this.channels;

    /// <summary>
    /// Convert the feature set matrices to a flattened vector
    /// </summary>
    /// <returns></returns>
    public Vec<T> ToVector() => Vec<T>.Wrap(this.SelectMany(x => x.FlattenRows()).ToArray());
}