namespace DotML.Federated.Http;

using System.Diagnostics.CodeAnalysis;
using System.Net.Http.Headers;

public abstract class HttpTrainer<TModel, TWeights>: TrainerWorker<TModel, TWeights>, IDisposable
{
    private readonly string coordinatorUrl;
    protected readonly HttpClient Client;


    public HttpTrainer(string coordinatorUrl, TModel model): this(coordinatorUrl, model, new HttpClient()) {}

    protected HttpTrainer(string coordinatorUrl, TModel model, HttpClient client): base(model)
    {
        this.coordinatorUrl = coordinatorUrl ?? throw new ArgumentNullException(nameof(coordinatorUrl));
        Client = client ?? throw new ArgumentNullException(nameof(client));
        ConfigureClient(Client);
    }

    public void Dispose()
    {
        Client.Dispose();
    }

    protected virtual void ConfigureClient(HttpClient client)
    {
        // Allow the derived classes to configure the HttpClient if needed, like with default request headers, timeouts, etc.
    }

    protected override bool TryPullGlobalWeights(out ModelVersion version, [NotNullWhen(true)]out TWeights? weights)
    {
        var req = Client
            .GetAsync(coordinatorUrl, HttpCompletionOption.ResponseHeadersRead)
            .GetAwaiter()
            .GetResult();

        version = new ModelVersion(0);
        weights = default;

        if(!req.IsSuccessStatusCode)
            return false;

        if (!req.Headers.TryGetValues("X-Model-Version", out var versionHeaders))
            return false;

        version = ModelVersion.Parse(versionHeaders.First(), System.Globalization.CultureInfo.InvariantCulture);
   
        using var stream = req.Content
            .ReadAsStreamAsync()
            .GetAwaiter()
            .GetResult();

        weights = DecodeWeights(stream);
        return weights != null;
    }

    protected override void PushLocalWeights(ModelVersion version, int samplesTrained, TWeights weights)
    {
        using var req = new HttpRequestMessage(HttpMethod.Post, coordinatorUrl);

        using var stream = EncodeWeights(weights);
        req.Content = new StreamContent(stream);
        req.Content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");

        req.Headers.Add("X-Model-Version", version.ToString());
        req.Headers.Add("X-Samples-Trained", samplesTrained.ToString(System.Globalization.CultureInfo.InvariantCulture));

        Client.Send(req);
    }

    protected abstract Stream EncodeWeights(TWeights weights);
    protected abstract TWeights DecodeWeights(Stream stream);
}