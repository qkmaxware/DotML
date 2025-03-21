using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class OperatorColon : RegexLexeme {
    public OperatorColon() : base(@"\G\s*(?<value>:)\s*", RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}