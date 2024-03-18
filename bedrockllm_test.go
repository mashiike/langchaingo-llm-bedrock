package bedrock_test

import (
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	bedrock "github.com/mashiike/langchaingo-llm-bedrock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/schema"
)

type mockBedrockClient struct {
	mock.Mock
	t *testing.T
}

func newMockBedrockClient(t *testing.T) *mockBedrockClient {
	c := &mockBedrockClient{t: t}
	c.Test(t)
	return c
}

func (m *mockBedrockClient) InvokeModel(ctx context.Context, params *bedrockruntime.InvokeModelInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
	var args mock.Arguments
	if len(optFns) > 0 {
		args = m.Called(ctx, params, optFns)
	} else {
		args = m.Called(ctx, params)
	}
	output := args.Get(0)
	err := args.Error(1)
	if output == nil {
		return nil, err
	}
	if o, ok := output.(*bedrockruntime.InvokeModelOutput); ok {
		return o, err
	}
	m.t.Errorf("unexpected output type: %T", output)
	return nil, err
}

var flagUseRemote = flag.Bool("use-remote", false, "run tests with remote resources")

func TestMain(m *testing.M) {
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})))
	m.Run()
}

func TestCreateEmbeddingWithTaitan(t *testing.T) {
	if !*flagUseRemote {
		t.Skip("skipping test; use -use-remote to enable")
	}
	llm, err := bedrock.New(bedrock.WithLogger(slog.Default()))
	require.NoError(t, err)
	embeddings, err := llm.CreateEmbedding(context.Background(), []string{
		"this is a pen",
		"this is a apple",
		"i eat a apple",
	})
	require.NoError(t, err)
	calcCosDistance := func(a, b []float32) float32 {
		var dot, a2, b2 float32
		for i := range a {
			dot += a[i] * b[i]
			a2 += a[i] * a[i]
			b2 += b[i] * b[i]
		}
		return dot / (a2 * b2)
	}
	dis1 := calcCosDistance(embeddings[0], embeddings[1])
	t.Logf("`this is a pen` vs `this is a apple` distance: %f", dis1)
	dis2 := calcCosDistance(embeddings[0], embeddings[2])
	t.Logf("`this is a pen` vs `i eat a apple` distance: %f", dis2)
	require.Greater(t, dis1, dis2)
}

func TestMockCreateEmbeddingWithTaitan(t *testing.T) {
	m := newMockBedrockClient(t)
	m.On("InvokeModel", mock.Anything, mock.MatchedBy(
		func(input *bedrockruntime.InvokeModelInput) bool {
			if input.ModelId == nil {
				return false
			}
			if *input.ModelId != bedrock.TitanEmbeddingG1Text {
				return false
			}
			var payload map[string]interface{}
			if err := json.Unmarshal(input.Body, &payload); err != nil {
				return false
			}
			if text, ok := payload["inputText"].(string); ok {
				return text == "this is a pen"
			}
			return false
		}),
	).Return(&bedrockruntime.InvokeModelOutput{
		Body: []byte(`{"embedding": [0.1, 0.2, 0.3]}`),
	}, nil).Times(1)
	defer m.AssertExpectations(t)

	llm, err := bedrock.New(bedrock.WithClient(m))
	require.NoError(t, err)
	embeddings, err := llm.CreateEmbedding(context.Background(), []string{
		"this is a pen",
	})
	require.NoError(t, err)
	require.EqualValues(t, [][]float32{{0.1, 0.2, 0.3}}, embeddings)
}

func TestGenerateContentWithClaude2(t *testing.T) {
	if !*flagUseRemote {
		t.Skip("skipping test; use -use-remote to enable")
	}
	llm, err := bedrock.New(
		bedrock.WithLogger(slog.Default()),
		bedrock.WithTemperature(0.1),
		bedrock.WithModel(bedrock.Claude2),
	)
	require.NoError(t, err)
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(schema.ChatMessageTypeHuman,
			`答えを知っているか、あるいは十分な推測ができる場合のみ、以下の質問に答えてください。答えられない場合は『わかりません』を出力してください。
これまで記録された中で最も重いカバは？」`),
	})
	require.NoError(t, err)
	t.Log(resp.Choices[0].Content)
	require.Contains(t, resp.Choices[0].Content, "わかりません")
}

func TestGenerateContentWithClaude3Haiku(t *testing.T) {
	if !*flagUseRemote {
		t.Skip("skipping test; use -use-remote to enable")
	}
	llm, err := bedrock.New(
		bedrock.WithLogger(slog.Default()),
		bedrock.WithTemperature(0.1),
		bedrock.WithModel(bedrock.Claude3Haiku),
	)
	require.NoError(t, err)
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(schema.ChatMessageTypeHuman,
			`答えを知っているか、あるいは十分な推測ができる場合のみ、以下の質問に答えてください。答えられない場合は『わかりません』を出力してください。
これまで記録された中で最も重いカバは？」`),
	})
	require.NoError(t, err)
	t.Log(resp.Choices[0].Content)
	require.Contains(t, resp.Choices[0].Content, "わかりません")
}

//go:embed testdata/lgtm.png
var image []byte

func TestGenerateContentWithClaude3HaikuWithImage(t *testing.T) {
	if !*flagUseRemote {
		t.Skip("skipping test; use -use-remote to enable")
	}
	llm, err := bedrock.New(
		bedrock.WithLogger(slog.Default()),
		bedrock.WithTemperature(0.1),
		bedrock.WithModel(bedrock.Claude3Haiku),
	)
	require.NoError(t, err)
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		{
			Role: schema.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.BinaryPart("image/png", image),
				llms.TextPart("この画像に書かれてる単語は何？"),
			},
		},
	})
	require.NoError(t, err)
	t.Log(resp.Choices[0].Content)
	require.Contains(t, resp.Choices[0].Content, "LGTM")
}

func TestMockGenerateContentWithClaude3HaikuWithImage(t *testing.T) {
	m := newMockBedrockClient(t)
	m.On("InvokeModel", mock.Anything, mock.MatchedBy(
		func(input *bedrockruntime.InvokeModelInput) bool {
			if input.ModelId == nil {
				return false
			}
			if *input.ModelId != bedrock.Claude3Haiku {
				return false
			}
			return assert.JSONEq(t, fmt.Sprintf(`{
	"temperature":0.1,
	"top_p":0.9,
	"top_k":50,
	"stop_sequences":["Human:"],
	"max_tokens":1000,
	"messages":[
		{
			"role":"user",
			"content":[
				{
					"type":"image",
					"source":{
						"type":"base64",
						"media_type":"image/png",
						"data":"%s"
					}
				},
				{
					"type":"text",
					"text":"この画像に書かれてる単語は何？"
				}
			]
		}
	],
	"anthropic_version":"bedrock-2023-05-31"
}`, base64.StdEncoding.EncodeToString(image)), string(input.Body))
		}),
	).Return(&bedrockruntime.InvokeModelOutput{
		Body: []byte(`{
	"id":"msg_000000000000000000000000",
	"type":"message",
	"role":"assistant",
	"content":[
		{"type":"text","text":"この画像に書かれている単語は \"LGTM\" です。"}
	],
	"model":"claude-3-haiku-48k-20240307",
	"stop_reason":"end_turn",
	"stop_sequence":null,
	"usage":{
		"input_tokens":140,
		"output_tokens":22
	}
}`)}, nil).Times(1)
	defer m.AssertExpectations(t)

	llm, err := bedrock.New(
		bedrock.WithClient(m),
		bedrock.WithLogger(slog.Default()),
		bedrock.WithTemperature(0.1),
		bedrock.WithModel(bedrock.Claude3Haiku),
	)
	require.NoError(t, err)
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		{
			Role: schema.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.BinaryPart("image/png", image),
				llms.TextPart("この画像に書かれてる単語は何？"),
			},
		},
	})
	require.NoError(t, err)
	t.Log(resp.Choices[0].Content)
	require.Contains(t, resp.Choices[0].Content, "LGTM")
}
