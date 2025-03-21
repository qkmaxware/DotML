using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordArg : RegexLexeme {
    public KeywordArg() : base(@"\G\s*\b(?<value>ARG)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}