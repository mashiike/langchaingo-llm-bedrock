package bedrock

import (
	"context"
	"io"
	"log/slog"
	"os"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

type BedrockClient interface {
	InvokeModel(ctx context.Context, params *bedrockruntime.InvokeModelInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error)
}

type options struct {
	region         string
	embeddingModel string
	model          string
	awsCfg         *aws.Config
	client         BedrockClient
	numWorkers     int
	logger         *slog.Logger
	maxTokens      int
	temperature    float64
	topP           float64
	topK           int
	stopWords      []string
}

func newOptions() *options {
	return &options{
		region:         os.Getenv("AWS_REGION"),
		embeddingModel: TitanEmbeddingG1Text,
		model:          ClaudeInstant,
		numWorkers:     10,
		logger:         slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{})),
		maxTokens:      1000,
		temperature:    0.7,
		topK:           50,
		topP:           0.9,
		stopWords:      []string{"Human:"},
	}
}

func (o *options) apply(opts ...Option) {
	for _, opt := range opts {
		opt(o)
	}
}

func (o *options) newBedrockClient(ctx context.Context) (BedrockClient, error) {
	if o.client != nil {
		return o.client, nil
	}

	if o.awsCfg == nil {
		awsCfg, err := config.LoadDefaultConfig(ctx)
		if err != nil {
			return nil, err
		}
		o.awsCfg = &awsCfg
	}

	bedrockOpts := []func(*bedrockruntime.Options){}
	if o.region != "" {
		bedrockOpts = append(bedrockOpts, func(bo *bedrockruntime.Options) {
			bo.Region = o.region
		})
	}
	o.client = bedrockruntime.NewFromConfig(*o.awsCfg, bedrockOpts...)
	return o.client, nil
}

type Option func(*options)

func WithRegion(region string) Option {
	return func(o *options) {
		o.region = region
	}
}

func WithModel(model string) Option {
	return func(o *options) {
		o.model = model
	}
}

func WithEmbeddingModel(embeddingModel string) Option {
	return func(o *options) {
		o.embeddingModel = embeddingModel
	}
}

func WithAWSConfig(cfg aws.Config) Option {
	return func(o *options) {
		o.awsCfg = &cfg
	}
}

func WithClient(client BedrockClient) Option {
	return func(o *options) {
		o.client = client
	}
}

func WithNumWorkers(workers int) Option {
	return func(o *options) {
		o.numWorkers = workers
	}
}

func WithLogger(logger *slog.Logger) Option {
	return func(o *options) {
		o.logger = logger
	}
}

func WithMaxTokens(maxTokens int) Option {
	return func(o *options) {
		o.maxTokens = maxTokens
	}
}

func WithTemperature(temperature float64) Option {
	return func(o *options) {
		o.temperature = temperature
	}
}

func WithTopP(topP float64) Option {
	return func(o *options) {
		o.topP = topP
	}
}

func WithTopK(topK int) Option {
	return func(o *options) {
		o.topK = topK
	}
}

func WithStopWords(stopWords []string) Option {
	return func(o *options) {
		o.stopWords = stopWords
	}
}
