# langchaingo-llm-bedrock

for `github.com/tmc/langchaingo` Amazon Bedrock LLM interface implement

[![GoDoc](https://godoc.org/github.com/mashiike/ github.com/mashiike/langchaingo-llm-bedrock?status.svg)](https://godoc.org/github.com/mashiike/ github.com/mashiike/langchaingo-llm-bedrock)
[![Go Report Card](https://goreportcard.com/badge/github.com/mashiike/ github.com/mashiike/langchaingo-llm-bedrock)](https://goreportcard.com/report/github.com/mashiike/ github.com/mashiike/langchaingo-llm-bedrock)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Example

```go
package main

import (
	"context"
	_ "embed"
	"log"

	bedrock "github.com/mashiike/langchaingo-llm-bedrock"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/schema"
)

//go:embed image.png
var image []byte

func main() {
	llm, err := bedrock.New(
		bedrock.WithModel(bedrock.Claude3Haiku),
	)
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	resp, err := llm.GenerateContent(ctx, []llms.MessageContent{
		{
			Role: schema.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.BinaryPart("image/png", image),
				llms.TextPart("この画像に書かれている内容をテキストにしてください。"),
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	for _, choice := range resp.Choices {
		log.Println(choice.Content)
	}
}
```

## License

MIT License
