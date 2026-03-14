namespace DotML;

/// <summary>
/// NCHW layer indices (Batch, Channels, Height, Width)
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
/// BSH layer indices (Batch, Sequence, Hidden)
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

/// <summary>
/// NHWC layer indices (Batch, Height, Width, Channels)
/// </summary>
public static class NHWC
{
    /// <summary>
    /// Reference to the column dimension in NHWC format
    /// </summary>
    public static readonly Index Columns = ^2;
    /// <summary>
    /// Reference to the row dimension in NHWC format
    /// </summary>
    public static readonly Index Rows = ^3;
    /// <summary>
    /// Reference to the channel dimension in NHWC format
    /// </summary>
    public static readonly Index Channels = ^1;
    /// <summary>
    /// Reference to the batch dimension in NHWC format
    /// </summary>
    public static readonly Index Batches = ^4;
}

/// <summary>
/// BHWC is an alias for NHWC (Batch, Height, Width, Channels)
/// </summary>
public static class BHWC
{
    public static readonly Index Columns = NHWC.Columns;
    public static readonly Index Rows = NHWC.Rows;
    public static readonly Index Channels = NHWC.Channels;
    public static readonly Index Batches = NHWC.Batches;
}

/// <summary>
/// CHW layer indices (Channels, Height, Width)
/// </summary>
public static class CHW
{
    /// <summary>
    /// Reference to the column dimension in CHW format
    /// </summary>
    public static readonly Index Columns = ^1;
    /// <summary>
    /// Reference to the row dimension in CHW format
    /// </summary>
    public static readonly Index Rows = ^2;
    /// <summary>
    /// Reference to the channel dimension in CHW format
    /// </summary>
    public static readonly Index Channels = ^3;
}

/// <summary>
/// HWC layer indices (Height, Width, Channels)
/// </summary>
public static class HWC
{
    /// <summary>
    /// Reference to the column dimension in HWC format
    /// </summary>
    public static readonly Index Columns = ^2;
    /// <summary>
    /// Reference to the row dimension in HWC format
    /// </summary>
    public static readonly Index Rows = ^3;
    /// <summary>
    /// Reference to the channel dimension in HWC format
    /// </summary>
    public static readonly Index Channels = ^1;
}

/// <summary>
/// NCDHW layer indices (Batch, Channels, Depth, Height, Width)
/// </summary>
public static class NCDHW
{
    /// <summary>
    /// Reference to the column (width) dimension in NCDHW format
    /// </summary>
    public static readonly Index Columns = ^1;
    /// <summary>
    /// Reference to the row (height) dimension in NCDHW format
    /// </summary>
    public static readonly Index Rows = ^2;
    /// <summary>
    /// Reference to the depth dimension in NCDHW format
    /// </summary>
    public static readonly Index Depth = ^3;
    /// <summary>
    /// Reference to the channel dimension in NCDHW format
    /// </summary>
    public static readonly Index Channels = ^4;
    /// <summary>
    /// Reference to the batch dimension in NCDHW format
    /// </summary>
    public static readonly Index Batches = ^5;
}

/// <summary>
/// NDHWC layer indices (Batch, Depth, Height, Width, Channels)
/// </summary>
public static class NDHWC
{
    /// <summary>
    /// Reference to the column (width) dimension in NDHWC format
    /// </summary>
    public static readonly Index Columns = ^2;
    /// <summary>
    /// Reference to the row (height) dimension in NDHWC format
    /// </summary>
    public static readonly Index Rows = ^3;
    /// <summary>
    /// Reference to the depth dimension in NDHWC format
    /// </summary>
    public static readonly Index Depth = ^4;
    /// <summary>
    /// Reference to the channel dimension in NDHWC format
    /// </summary>
    public static readonly Index Channels = ^1;
    /// <summary>
    /// Reference to the batch dimension in NDHWC format
    /// </summary>
    public static readonly Index Batches = ^5;
}