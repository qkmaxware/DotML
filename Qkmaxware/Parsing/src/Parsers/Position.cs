using System;

namespace Qkmaxware.Parsing {

/// <summary>
/// Static parsers for positional data
/// </summary>
public static class Position {
    /// <summary>
    /// Parser matching the start of the input sequence
    /// </summary>
    public static Parser<int> Start() { 
        return input => input.Position == 0 ? new Result<int>(0, input) : new Result<int>(new ArgumentOutOfRangeException(), input); 
    }

    /// <summary>
    /// Parser matching the a specific location in the input sequence
    /// </summary>
    public static Parser<int> At(int position) { 
        return input => input.Position == position ? new Result<int>(position, input) : new Result<int>(new ArgumentOutOfRangeException(), input); 
    }

    /// <summary>
    /// Parser matching the end of the input sequence
    /// </summary>
    public static Parser<int> End() { 
        return input => input.EndOfStream ? new Result<int>(input.Position, input) : new Result<int>(new ArgumentOutOfRangeException(), input); 
    }
}

}