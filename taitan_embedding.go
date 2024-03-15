package bedrock

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

type titanEmbeddingRequest struct {
	InputText string `json:"inputText"`
}

type titanEmbeddingResponse struct {
	Embedding           []float64 `json:"embedding"`
	InputTextTokenCount int       `json:"inputTextTokenCount"`
}

type titanEmbdddingJob struct {
	index int
	text  string
}

func (l *LLM) createEmbeddingWithTaitan(ctx context.Context, texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	jobs := make(chan titanEmbdddingJob, l.numWorkers)
	var wg sync.WaitGroup
	cctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)

	for w := 0; w < l.numWorkers; w++ {
		wg.Add(1)
		go func(id int, ch <-chan titanEmbdddingJob) {
			defer wg.Done()
			l.logger.Debug("start embedding worker", "id", id)
			for j := range ch {
				select {
				case <-cctx.Done():
					l.logger.Debug("embedding worker cancelled", "id", id)
					return
				default:
				}
				l.logger.Debug("create embedding", "id", id, "index", j.index, "text", j.text)
				embedding, err := l.createEmbeddingWithTaitanImpl(cctx, j.text)
				if err != nil {
					l.logger.Debug("failed to create embedding", "id", id, "err", err)
					cancel(fmt.Errorf("failed to create embedding for text %d: %w", j.index, err))
					return
				}
				embeddings[j.index] = embedding
			}
			l.logger.Debug("finish embedding worker", "id", id)
		}(w, jobs)
	}
	for i, text := range texts {
		jobs <- titanEmbdddingJob{
			index: i,
			text:  text,
		}
	}
	close(jobs)
	wg.Wait()
	if err := context.Cause(cctx); err != nil {
		return nil, err
	}
	return embeddings, nil
}

func (l *LLM) createEmbeddingWithTaitanImpl(ctx context.Context, text string) ([]float32, error) {
	payload := &titanEmbeddingRequest{
		InputText: text,
	}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}

	output, err := l.client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(l.embeddingModel),
		Body:        payloadBytes,
		ContentType: aws.String("application/json"),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to invoke model: %w", err)
	}
	var resp titanEmbeddingResponse
	if err := json.Unmarshal(output.Body, &resp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}
	embedding := make([]float32, len(resp.Embedding))
	for i, v := range resp.Embedding {
		embedding[i] = float32(v)
	}
	l.logger.Debug("embedding created", "text", text, "token_count", resp.InputTextTokenCount)
	return embedding, nil
}
