# Architecture

This document explains the Retrieval-Augmented Generation (RAG) architecture used in this project.

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that enhances Large Language Model (LLM) responses by providing relevant context from your own data. Instead of relying solely on the LLM's training data, RAG retrieves relevant documents and includes them in the prompt.

## System Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   PDF File  │────▶│  PDF Reader  │────▶│  Text Splitter  │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                  │
                                                  ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   OpenAI    │◀────│   Embeddings │◀────│  Text Chunks    │
│  Embeddings │     │   API Call   │     │                 │
└──────┬──────┘     └──────────────┘     └─────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              PostgreSQL + PGVector                       │
│                  (Vector Store)                          │
└─────────────────────────────────────────────────────────┘
       │
       │  Similarity Search
       ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│    User     │────▶│   Chatbot    │────▶│  OpenAI Chat    │
│   Query     │     │  Component   │     │   Completion    │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                  │
                                                  ▼
                                         ┌─────────────────┐
                                         │    Response     │
                                         └─────────────────┘
```

## Components

### 1. PDF Document Reader

**Class:** `PagePdfDocumentReader` (Spring AI)

Reads PDF files and extracts text content page by page. Configuration options include:
- Skipping header/footer lines
- Pages per document chunk

```java
var config = PdfDocumentReaderConfig.builder()
    .withPageExtractedTextFormatter(new ExtractedTextFormatter.Builder()
        .withNumberOfBottomTextLinesToDelete(3)
        .withNumberOfTopPagesToSkipBeforeDelete(1)
        .build())
    .withPagesPerDocument(1)
    .build();
```

### 2. Text Splitter

**Class:** `TokenTextSplitter` (Spring AI)

Splits large documents into smaller chunks optimized for embedding generation. This is important because:
- Embedding models have token limits
- Smaller chunks provide more precise similarity matches
- Reduces noise in retrieved context

### 3. Vector Store

**Technology:** PostgreSQL with PGVector extension

Stores document chunks as vector embeddings. PGVector enables:
- Efficient similarity search using cosine distance
- SQL-based querying and management
- Scalable storage for large document collections

**Table:** `vector_store`
- Stores text content alongside its vector embedding
- Enables fast approximate nearest neighbor (ANN) search

### 4. Chatbot Component

**Class:** `Chatbot`

The core RAG implementation:

1. **Receives** user query
2. **Searches** vector store for similar documents
3. **Constructs** prompt with retrieved context
4. **Calls** OpenAI API with augmented prompt
5. **Returns** generated response

```java
public String chat(String message) {
    // 1. Similarity search
    var listOfSimilarDocuments = this.vectorStore.similaritySearch(message);

    // 2. Extract content from matched documents
    var documents = listOfSimilarDocuments.stream()
        .map(Document::getContent)
        .collect(Collectors.joining(System.lineSeparator()));

    // 3. Build prompt with context
    var systemMessage = new SystemPromptTemplate(this.template)
        .createMessage(Map.of("documents", documents));
    var userMessage = new UserMessage(message);
    var prompt = new Prompt(List.of(systemMessage, userMessage));

    // 4. Get response from LLM
    var aiResponse = aiClient.call(prompt);
    return aiResponse.getResult().getOutput().getContent();
}
```

### 5. Prompt Template

The system prompt instructs the LLM how to use the retrieved context:

```
You're a wikipedia expert specialized in the olympic games.
Provide accurate answers but act as if you knew this information innately.
If unsure, simply state that you don't know.
DOCUMENTS:
{documents}
```

## Data Flow

### Indexing Phase (Application Startup)

1. **Load PDF** - Read the PDF file from disk
2. **Extract Text** - Parse PDF pages into raw text
3. **Split Text** - Divide into manageable chunks
4. **Generate Embeddings** - Convert chunks to vectors via OpenAI
5. **Store Vectors** - Save embeddings in PostgreSQL

### Query Phase

1. **User Query** - Receive natural language question
2. **Query Embedding** - Convert question to vector
3. **Similarity Search** - Find closest matching document chunks
4. **Context Assembly** - Combine relevant chunks
5. **LLM Prompt** - Send question + context to OpenAI
6. **Response** - Return generated answer

## Key Spring AI Abstractions

| Component | Interface | Implementation |
|-----------|-----------|----------------|
| Chat Client | `ChatClient` | OpenAI Chat Completion API |
| Vector Store | `VectorStore` | PGVector (PostgreSQL) |
| Document Reader | `DocumentReader` | `PagePdfDocumentReader` |
| Text Splitter | `DocumentTransformer` | `TokenTextSplitter` |

## Why RAG?

| Without RAG | With RAG |
|-------------|----------|
| LLM can only use training data | LLM uses your custom documents |
| Knowledge cutoff date limits answers | Always up-to-date with your data |
| May hallucinate facts | Grounded in retrieved documents |
| Generic responses | Domain-specific accurate answers |

## Further Reading

- [Spring AI Documentation](https://spring.io/projects/spring-ai)
- [PGVector GitHub](https://github.com/pgvector/pgvector)
- [RAG Paper (Lewis et al.)](https://arxiv.org/abs/2005.11401)
