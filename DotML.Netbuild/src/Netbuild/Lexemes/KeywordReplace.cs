using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordReplace : RegexLexeme {
    public KeywordReplace() : base(@"\G\s*\b(?<value>REPLACE)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}