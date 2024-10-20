namespace DotML.Sandbox.Data;

/// <summary>
/// A heap boxed value allowing multiple spots in the code to point to the same value
/// </summary>
/// <typeparam name="T">Type of item being boxed</typeparam>
public class Box<T> {
    public T? Value {get; set;}

    public Box() {
        this.Value = default(T);
    }

    public Box(T value) {
        this.Value = value;
    }
}