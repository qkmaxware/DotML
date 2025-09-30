using System;

namespace Qkmaxware.Parsing {

public struct InputString : IInput {
    private string data;
    public int Position {get; private set;}
    public bool EndOfStream => Position >= data.Length;

    public InputString(string data) {
        this.data = data;
        this.Position = 0;
    }

    public InputString (string data, int position) {
        this.data = data;
        this.Position = position;
    }

    public Result<char> NextChar() {
        if (data == null)
            return new Result<char>(new NullReferenceException(), this);

        if (!EndOfStream) {
            return new Result<char>(data[Position], new InputString(this.data, this.Position + 1));
        } else {
            return new Result<char>(new IndexOutOfRangeException(), this);
        }
    } 

    public static implicit operator InputString(string data) {
        return new InputString(data);
    }
}

}