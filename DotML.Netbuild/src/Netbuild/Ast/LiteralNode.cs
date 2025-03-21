namespace DotML.Network.IO.Netbuild;

public abstract class Literal : AstNode {
    public abstract object ValueOf();

    public string AsString() => Convert.ToString(ValueOf()) ?? string.Empty;
    public int AsInt() => Convert.ToInt32(ValueOf());
    public long AsLong() => Convert.ToInt64(ValueOf());
    public float AsFloat() => Convert.ToSingle(ValueOf());
    public double AsDouble() => Convert.ToDouble(ValueOf());
}

public class ObjectLiteral : Literal {
    private object obj;
    public ObjectLiteral(object o) {
        this.obj = o;
    }
    public override object ValueOf() => obj;

    public override string ToString() {
        return obj?.ToString() ?? string.Empty;
    }
}