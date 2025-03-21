using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordWith : RegexLexeme {
    public KeywordWith() : base(@"\G\s*\b(?<value>WITH)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}