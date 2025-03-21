using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordPretrain : RegexLexeme {
    public KeywordPretrain() : base(@"\G\s*\b(?<value>PRETRAIN)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}