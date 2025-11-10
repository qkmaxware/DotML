namespace DotML.Network.IO.Netbuild;

public class Token {
    public Lexeme Type {get; private set;}
    public int Position {get; private set;}
    public int Length {get; private set;}

    public Token(Lexeme type, int position, int length) {
        this.Type = type;
        this.Position = position;
        this.Length = length;
    }
}

public class Token<T> : Token {

    public T Value {get; private set;}

    public Token(Lexeme type, int position, int length, T value) : base(type, position, length) {
        this.Value = value;
    }

    public override string ToString() => Value?.ToString() ?? string.Empty;
}