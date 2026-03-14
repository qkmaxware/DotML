using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class OperatorAssign : Operator {
    public OperatorAssign() : base(@"\G\s*(?<value>=)\s*", "value") { }
}