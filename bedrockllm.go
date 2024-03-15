package bedrock

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms"
)

type LLM struct {
	CallbacksHandler callbacks.Handler
	client           BedrockClient
	logger           *slog.Logger
	numWorkers       int
	model            string
	embeddingModel   string
	maxTokens        int
	topK             int
	topP             float64
	temperature      float64
	stopWords        []string
}

var _ llms.Model = (*LLM)(nil)
var _ embeddings.EmbedderClient = (*LLM)(nil)

// New returns a new Bedrock LLM.
func New(opts ...Option) (*LLM, error) {
	o := newOptions()
	o.apply(opts...)
	client, err := o.newBedrockClient(context.Background())
	if err != nil {
		return nil, err
	}
	return &LLM{
		client:         client,
		logger:         o.logger,
		numWorkers:     o.numWorkers,
		model:          o.model,
		embeddingModel: o.embeddingModel,
		maxTokens:      o.maxTokens,
		topK:           o.topK,
		topP:           o.topP,
		temperature:    o.temperature,
		stopWords:      o.stopWords,
	}, nil
}

func (l *LLM) CreateEmbedding(ctx context.Context, texts []string) ([][]float32, error) {
	l.logger.Debug("bedrock.LLM.CreateEmbedding called", "texts", texts)
	switch l.embeddingModel {
	case TitanEmbeddingG1Text:
		return l.createEmbeddingWithTaitan(ctx, texts)
	default:
		return nil, fmt.Errorf("embedding model `%s` not supported", l.embeddingModel)
	}
}

func (l *LLM) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
	l.logger.Debug("bedrock.LLM.GenerateContent called", "messages", messages)
	if l.CallbacksHandler != nil {
		l.CallbacksHandler.HandleLLMGenerateContentStart(ctx, messages)
	}

	opts := &llms.CallOptions{
		Model: l.model,
	}
	for _, opt := range options {
		opt(opts)
	}
	var resp *llms.ContentResponse
	var err error
	switch l.model {
	case ClaudeV2, ClaudeInstant:
		resp, err = l.generateContentWithClaudeV2(ctx, messages, opts)
	default:
		err = fmt.Errorf("model `%s` not supported", l.model)
	}
	if err != nil {
		if l.CallbacksHandler != nil {
			l.CallbacksHandler.HandleLLMError(ctx, err)
		}
		return nil, err
	}
	if l.CallbacksHandler != nil {
		l.CallbacksHandler.HandleLLMGenerateContentEnd(ctx, resp)
	}
	return resp, nil
}

func (l *LLM) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	l.logger.Debug("bedrock.LLM.Call called", "prompt", prompt)
	return llms.GenerateFromSinglePrompt(ctx, l, prompt, options...)
}
