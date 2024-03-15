package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/tmc/langchaingo/llms"
)

type Claude2Request struct {
	Prompt            string   `json:"prompt"`
	MaxTokensToSample int      `json:"max_tokens_to_sample"`
	Temperature       float64  `json:"temperature,omitempty"`
	TopP              float64  `json:"top_p,omitempty"`
	TopK              int      `json:"top_k,omitempty"`
	StopSequences     []string `json:"stop_sequences,omitempty"`
}

type Claude2Response struct {
	Completion string `json:"completion"`
}

func (l *LLM) generateContentWithClaude2(ctx context.Context, messages []llms.MessageContent, opts *llms.CallOptions) (*llms.ContentResponse, error) {
	if len(messages) != 1 {
		return nil, errors.New("only one message is supported")
	}
	if opts.MaxTokens == 0 {
		opts.MaxTokens = l.maxTokens
	}
	if opts.Temperature == 0 {
		opts.Temperature = l.temperature
	}
	if opts.TopP == 0 {
		opts.TopP = l.topP
	}
	if opts.TopK == 0 {
		opts.TopK = l.topK
	}
	if opts.StopWords == nil {
		opts.StopWords = l.stopWords
	}
	msg := messages[0]
	if len(msg.Parts) != 1 {
		return nil, errors.New("only one part is supported")
	}
	textPart, ok := msg.Parts[0].(llms.TextContent)
	if !ok {
		return nil, errors.New("only text content is supported")
	}
	prompt := textPart.Text
	if !strings.Contains(prompt, "\n\nHuman:") {
		prompt = "\n\nHuman:" + prompt
	}
	if !strings.Contains(prompt, "\n\nAssistant:") {
		prompt = prompt + "\n\nAssistant:"
	}

	payload := Claude2Request{
		Prompt:            prompt,
		MaxTokensToSample: opts.MaxTokens,
		Temperature:       opts.Temperature,
		TopP:              opts.TopP,
		TopK:              opts.TopK,
		StopSequences:     opts.StopWords,
	}
	l.logger.Debug("generate content with claude v2", "payload", payload)
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}
	output, err := l.client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(opts.Model),
		Body:        payloadBytes,
		ContentType: aws.String("application/json"),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to invoke model: %w", err)
	}
	var resp Claude2Response
	if err := json.Unmarshal(output.Body, &resp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}
	llmResponse := &llms.ContentResponse{
		Choices: []*llms.ContentChoice{
			{
				Content: resp.Completion,
				GenerationInfo: map[string]interface{}{
					"model": opts.Model,
				},
			},
		},
	}
	return llmResponse, nil
}
