using System;
using System.Linq;
using System.Collections.Generic;

namespace Qkmaxware.Parsing {

public static class Constant {
    /// <summary>
    /// Constant value parser
    /// </summary>
    public static Parser<T> Value<T>(T value) {
        return input => new Result<T>(value, input);
    }
}

}

