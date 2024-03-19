package bedrock

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/schema"
)

type Claude3Request struct {
	Temperature      float64                  `json:"temperature,omitempty"`
	TopP             float64                  `json:"top_p,omitempty"`
	TopK             int                      `json:"top_k,omitempty"`
	StopSequences    []string                 `json:"stop_sequences,omitempty"`
	System           string                   `json:"system,omitempty"`
	MaxTokens        int                      `json:"max_tokens,omitempty"`
	Messages         []*Claude3RequestMessage `json:"messages,omitempty"`
	AnthropicVersion string                   `json:"anthropic_version,omitempty"`
}

type Claude3RequestMessage struct {
	Role    string                         `json:"role"`
	Content []Claude3RequestMessageContent `json:"content"`
}

type Claude3RequestMessageContent interface {
	thisIslaudeV3RequestMessageContent()
}

type Claude3RequestMessageTextContent struct {
	Type string `json:"type,omitempty"`
	Text string `json:"text,omitempty"`
}

func (Claude3RequestMessageTextContent) thisIslaudeV3RequestMessageContent() {}

type Claude3RequestMessageImageContent struct {
	Type   string                                     `json:"type,omitempty"`
	Source claoudelV3RequestMessageImageContentSource `json:"source,omitempty"`
}

func (Claude3RequestMessageImageContent) thisIslaudeV3RequestMessageContent() {}

type claoudelV3RequestMessageImageContentSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type Claude3Response struct {
	Content      []Claude3ResponseContent `json:"content"`
	ID           string                   `json:"id"`
	Model        string                   `json:"model"`
	Role         string                   `json:"role"`
	StopReason   string                   `json:"stop_reason"`
	StopSequence any                      `json:"stop_sequence"`
	Type         string                   `json:"type"`
	Usage        Claude3ResponseUsage     `json:"usage"`
}

type Claude3ResponseContent struct {
	Text string `json:"text"`
	Type string `json:"type"`
}

type Claude3ResponseUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

func convertMessageForClaude3(message llms.MessageContent) (*Claude3RequestMessage, error) {
	var role string
	switch message.Role {
	case schema.ChatMessageTypeHuman:
		role = "user"
	case schema.ChatMessageTypeAI:
		role = "assistant"
	case schema.ChatMessageTypeSystem:
		role = "system"
	default:
		return nil, fmt.Errorf("unsupported role: %s", message.Role)
	}
	var content []Claude3RequestMessageContent
	for _, part := range message.Parts {
		switch p := part.(type) {
		case llms.TextContent:
			content = append(content, Claude3RequestMessageTextContent{
				Type: "text",
				Text: p.Text,
			})
		case llms.ImageURLContent:
			resp, err := http.DefaultClient.Get(p.URL)
			if err != nil {
				return nil, fmt.Errorf("failed to get image: %w", err)
			}
			if resp.StatusCode >= http.StatusNoContent {
				return nil, fmt.Errorf("failed to get image: %s", resp.Status)
			}
			bs, err := io.ReadAll(resp.Body)
			if err != nil {
				return nil, fmt.Errorf("failed to read image: %w", err)
			}
			resp.Body.Close()
			mediaType := http.DetectContentType(bs)
			content = append(content, Claude3RequestMessageImageContent{
				Type: "image",
				Source: claoudelV3RequestMessageImageContentSource{
					Type:      "base64",
					MediaType: mediaType,
					Data:      base64.StdEncoding.EncodeToString(bs),
				},
			})
		case llms.BinaryContent:
			content = append(content, Claude3RequestMessageImageContent{
				Type: "image",
				Source: claoudelV3RequestMessageImageContentSource{
					Type:      "base64",
					MediaType: p.MIMEType,
					Data:      base64.StdEncoding.EncodeToString(p.Data),
				},
			})
		default:
			return nil, fmt.Errorf("unsupported content type: %T", p)
		}
	}
	return &Claude3RequestMessage{
		Role:    role,
		Content: content,
	}, nil
}

func convertMessagesForClaude3(messages []llms.MessageContent) ([]*Claude3RequestMessage, error) {
	var result []*Claude3RequestMessage
	for _, message := range messages {
		msg, err := convertMessageForClaude3(message)
		if err != nil {
			return nil, fmt.Errorf("failed to convert message: %w", err)
		}
		result = append(result, msg)
	}
	return result, nil
}

func (l *LLM) generateContentWithClaude3(ctx context.Context, messages []llms.MessageContent, opts *llms.CallOptions) (*llms.ContentResponse, error) {
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
	payload := Claude3Request{
		AnthropicVersion: "bedrock-2023-05-31",
		Temperature:      opts.Temperature,
		TopP:             opts.TopP,
		TopK:             opts.TopK,
		StopSequences:    opts.StopWords,
		MaxTokens:        opts.MaxTokens,
	}
	msgs, err := convertMessagesForClaude3(messages)
	if err != nil {
		return nil, fmt.Errorf("failed to convert messages: %w", err)
	}
	if len(msgs) == 0 {
		return nil, errors.New("no messages")
	}
	if msgs[0].Role == "system" {
		var builder strings.Builder
		for _, content := range msgs[0].Content {
			if textContent, ok := content.(Claude3RequestMessageTextContent); ok {
				builder.WriteString(textContent.Text)
			}
		}
		payload.System = builder.String()
		msgs = msgs[1:]
	}
	if len(msgs) == 1 && len(msgs[0].Content) == 1 {
		if textContent, ok := msgs[0].Content[0].(Claude3RequestMessageTextContent); ok {
			text := textContent.Text
			if strings.Contains(text, "\n\nHuman:") {
				splits := strings.SplitN(text, "\n\nHuman:", 2)
				if systemPrompt := strings.TrimSpace(splits[0]); systemPrompt != "" {
					payload.System += systemPrompt
				}
				text = strings.TrimPrefix(splits[1], "\n\nHuman:")
			}
			var assistantSection *Claude3RequestMessage
			splits := strings.SplitN(text, "\n\nAssistant:", 2)
			if len(splits) == 2 {
				if assistantText := strings.TrimSpace(strings.TrimPrefix(splits[1], "\n\nAssistant:")); assistantText != "" {
					assistantSection = &Claude3RequestMessage{
						Role: "assistant",
						Content: []Claude3RequestMessageContent{
							Claude3RequestMessageTextContent{
								Type: "text",
								Text: assistantText,
							},
						},
					}
				}
			}
			payload.Messages = []*Claude3RequestMessage{
				{
					Role: "user",
					Content: []Claude3RequestMessageContent{
						Claude3RequestMessageTextContent{
							Type: "text",
							Text: strings.TrimSpace(splits[0]),
						},
					},
				},
			}
			if assistantSection != nil {
				payload.Messages = append(payload.Messages, assistantSection)
			}
		}
	}
	if len(payload.Messages) == 0 {
		payload.Messages = msgs
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}
	l.logger.Debug("generate content with claude v3", "payload", string(payloadBytes))
	output, err := l.client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(opts.Model),
		Body:        payloadBytes,
		ContentType: aws.String("application/json"),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to invoke model: %w", err)
	}
	var resp Claude3Response
	if err := json.Unmarshal(output.Body, &resp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}
	l.logger.Debug("generate content with claude v3", "id", resp.ID, "role", resp.Role, "stop_reason", resp.StopReason, "stop_sequence", resp.StopSequence, "type", resp.Type, "usage", resp.Usage)
	var builder strings.Builder
	for _, content := range resp.Content {
		builder.WriteString(content.Text)
	}
	llmResponse := &llms.ContentResponse{
		Choices: []*llms.ContentChoice{
			{
				Content:    builder.String(),
				StopReason: resp.StopReason,
				GenerationInfo: map[string]interface{}{
					"id":                  resp.ID,
					"model":               opts.Model,
					"usage.input_tokens":  resp.Usage.InputTokens,
					"usage.output_tokens": resp.Usage.OutputTokens,
					"stop_sequence":       resp.StopSequence,
					"role":                resp.Role,
				},
			},
		},
	}
	return llmResponse, nil
}
