namespace DotML;

/// <summary>
/// NCHW layer indices
/// </summary>
public static class NCHW
{
    /// <summary>
    /// Reference to the column dimension in NCHW format
    /// </summary>
    public static readonly Index Columns = ^1;
    /// <summary>
    /// Reference to the row dimension in NCHW format
    /// </summary>
    public static readonly Index Rows = ^2;
    /// <summary>
    /// Reference to the channel dimension in NCHW format
    /// </summary>
    public static readonly Index Channels = ^3;
    /// <summary>
    /// Reference to the batch dimension in NCHW format
    /// </summary>
    public static readonly Index Batches = ^4;
}

/// <summary>
/// BSH layer indices
/// </summary>
public static class BSH
{
    /// <summary>
    /// Reference to the hidden dimension in BSH format
    /// </summary>
    public static readonly Index Hidden = ^1;
    /// <summary>
    /// Reference to the sequence length dimension in BSH format
    /// </summary>
    public static readonly Index Sequence = ^2;
    /// <summary>
    /// Reference to the batch dimension in BSH format
    /// </summary>
    public static readonly Index Batches = ^3;
}