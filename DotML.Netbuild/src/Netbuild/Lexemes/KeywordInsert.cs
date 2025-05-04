using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordInsert : Keyword {
    public KeywordInsert() : base(@"\G\s*\b(?<value>INSERT)\b\s*", "value") { }
}

internal class KeywordAfter : Keyword {
    public KeywordAfter() : base(@"\G\s*\b(?<value>AFTER)\b\s*", "value") { }
}

internal class KeywordBefore : Keyword {
    public KeywordBefore() : base(@"\G\s*\b(?<value>BEFORE)\b\s*", "value") { }
}