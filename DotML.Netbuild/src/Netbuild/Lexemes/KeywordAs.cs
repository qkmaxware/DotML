using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordAs : RegexLexeme {
    public KeywordAs() : base(@"\G\s*\b(?<value>AS)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}