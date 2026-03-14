using System;

namespace Qkmaxware.Parsing {

public interface IInput {
    int Position {get;}
    bool EndOfStream {get;}
    Result<char> NextChar();
} 

}