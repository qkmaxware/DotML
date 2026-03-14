using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class OperatorColon : Operator {
    public OperatorColon() : base(@"\G\s*(?<value>:)\s*", "value") { }
}