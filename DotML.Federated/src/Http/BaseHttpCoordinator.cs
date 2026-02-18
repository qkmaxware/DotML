namespace DotML.Federated.Http;

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;

public abstract class BaseHttpCoordinator<TModel, TWeights> : Coordinator<TModel, TWeights>
{
 
    private readonly string[] urls;

    public BaseHttpCoordinator(TModel initialModel, params string[] urls) : base(initialModel)
    {
        if (urls.Length == 0)
            throw new ArgumentException("At least one URL must be provided.", nameof(urls));

        this.urls = urls;
    }

    public async Task<WebApplication> StartAsync(CancellationToken token)
    {
        var builder = WebApplication.CreateBuilder();

        builder.WebHost.UseUrls(urls);

        // Allow derived classes to configure services if needed
        ConfigureServices(builder.Services);

        var _app = builder.Build();

        ConfigureEndpoints(_app);

        await _app.StartAsync(token);
        return _app;
    }

    /// <summary>
    /// Override to add DI services (logging, auth, etc.)
    /// </summary>
    protected virtual void ConfigureServices(IServiceCollection services)
    {
        
    }

    /// <summary>
    /// Derived classes must map HTTP endpoints.
    /// </summary>
    protected virtual void ConfigureEndpoints(WebApplication app)
    {
        app.MapGet("/model", HandleGetModel);
        app.MapPost("/model", HandlePostModel);
    }

    private async Task HandleGetModel(HttpContext context)
    {
        var weights = FetchWeights(out ModelVersion version);

        context.Response.Headers["X-Model-Version"] = version.ToString();
        context.Response.ContentType = "application/octet-stream";

        await EncodeWeightsAsync(context.Response.Headers, weights, context.Response.Body, context.RequestAborted);
    }

    private async Task<IResult> HandlePostModel(HttpRequest request, CancellationToken cancellationToken)
    {
        if (!request.Headers.TryGetValue("X-Model-Version", out var versionHeader) ||
            !request.Headers.TryGetValue("X-Samples-Trained", out var samplesHeader))
        {
            return Results.BadRequest("Missing required headers.");
        }

        var version = ModelVersion.Parse(versionHeader.First(), System.Globalization.CultureInfo.InvariantCulture);
        var samples = int.Parse(samplesHeader!);

        await using var bodyStream = request.Body;

        var weights = DecodeWeights(bodyStream);
        PushWeights(version, samples, weights);

        return Results.Ok();
    }

    protected abstract Task EncodeWeightsAsync(IHeaderDictionary headers, TWeights weights, Stream stream, CancellationToken cancel);

    protected abstract TWeights DecodeWeights(Stream stream);
}