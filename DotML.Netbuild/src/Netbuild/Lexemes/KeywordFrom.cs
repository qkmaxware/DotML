using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordFrom : RegexLexeme {
    public KeywordFrom() : base(@"\G\s*\b(?<value>FROM)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}