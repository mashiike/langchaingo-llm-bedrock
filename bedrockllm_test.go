package bedrock_test

import (
	"context"
	_ "embed"
	"flag"
	"log/slog"
	"os"
	"testing"

	bedrock "github.com/mashiike/langchaingo-llm-bedrock"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/schema"
)

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
