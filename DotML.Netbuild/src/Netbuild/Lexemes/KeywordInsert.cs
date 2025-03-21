using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordInsert : RegexLexeme {
    public KeywordInsert() : base(@"\G\s*\b(?<value>INSERT)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}

internal class KeywordAfter : RegexLexeme {
    public KeywordAfter() : base(@"\G\s*\b(?<value>AFTER)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}

internal class KeywordBefore : RegexLexeme {
    public KeywordBefore() : base(@"\G\s*\b(?<value>BEFORE)\b\s*", RegexOptions.IgnoreCase | RegexOptions.Compiled) {
        this.GroupName = "value";
    }
}