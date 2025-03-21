using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class OperatorAssign : RegexLexeme {
    public OperatorAssign() : base(@"\G\s*(?<value>=)\s*", RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}