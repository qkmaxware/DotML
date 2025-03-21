using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordInput : RegexLexeme {
    public KeywordInput() : base(@"\G\s*\b(?<value>INPUT)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}