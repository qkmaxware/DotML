namespace Qkmaxware.Parsing
{

public abstract class Maybe<T>
{
    private Maybe() { }

    public abstract bool TryGetValue(out T? value);

    public abstract O Match<O>(Func<Some, O> some, Func<None, O> none);

    public sealed class Some : Maybe<T>
    {
        public Some(T value) => Value = value;
        public T Value { get; }

        public override bool TryGetValue(out T? value)
        {
            value = this.Value;
            return true;
        }

        public override O Match<O>(Func<Some, O> some, Func<None, O> none) => some(this);
    }

        public sealed class None : Maybe<T>
        {
        public override bool TryGetValue(out T? value)
        {
            value = default(T);
            return false;
        }
        
        public override O Match<O>(Func<Some, O> some, Func<None, O> none) => none(this);
    }
}

}