using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordAdd : RegexLexeme {
    public KeywordAdd() : base(@"\G\s*\b(?<value>ADD)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}