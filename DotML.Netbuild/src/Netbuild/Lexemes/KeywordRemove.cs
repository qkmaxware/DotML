using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordRemove : RegexLexeme {
    public KeywordRemove() : base(@"\G\s*\b(?<value>REMOVE)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}