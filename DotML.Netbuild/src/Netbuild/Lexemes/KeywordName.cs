using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordName : RegexLexeme {
    public KeywordName() : base(@"\G\s*\b(?<value>NAME|LABEL)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}